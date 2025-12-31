from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
from langchain_groq import ChatGroq
from PyPDF2 import PdfReader
from datetime import datetime
from gtts import gTTS
import os
import re
import logging
import sys

app = Flask(__name__)
CORS(app)

app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Use CPU on Render since it doesn't have GPU support
if os.getenv("RENDER"):
    device = torch.device("cpu")
    print("Running on Render - using CPU")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

# Initialize model variables to None
processor = None
model = None
model_loaded = False

# Attempt to load Vision Transformer model only if the model directory exists
if os.path.exists("vit_model"):
    try:
        vit_path = "vit_model"
        print("Loading Vision Transformer model...")
        processor = ViTImageProcessor.from_pretrained(vit_path)
        model = ViTForImageClassification.from_pretrained(vit_path).to(device)
        model_loaded = True
        print("✅ Vision Transformer model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load Vision Transformer model: {e}")
        model_loaded = False
else:
    print("⚠️  Vision Transformer model directory not found, model will not be loaded")
    print("   To use image prediction features, please add the 'vit_model' directory with the required model files")
    model_loaded = False


# Load LLM (Llama-3.3-70b-versatile via Groq)
try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY")  # Set this in Render dashboard
    )
    print("✅ LLM loaded successfully")
except Exception as e:
    print(f"❌ Failed to load LLM: {e}")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ------------------ Predict Route ------------------
@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded:
        return jsonify({'error': 'Vision model is not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided in the request'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or no selected file'}), 400

    temp_path = ""
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)

        image = Image.open(temp_path).convert('RGB')
        # Ensure model and processor are loaded before using them
        if processor is None or model is None:
            return jsonify({'error': 'Vision model is not properly loaded'}), 500
        
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            predicted_class_idx = outputs.logits.argmax(-1).item()
            predicted_class = model.config.id2label[predicted_class_idx]

        lang = request.form.get("lang", "en")
        part = request.form.get("plantPart", "leaf").lower()
        issue = predicted_class if part == "leaf" else f"a disease on the {part}"

        prompt_templates = {
            "mr": f"ही एका झाडाच्या {part} ची प्रतिमा आहे, जी रोगग्रस्त दिसते. संभाव्य रोग '{issue}' आहे. कृपया शेतकऱ्याला समजेल अशा सोप्या मराठीत सांगा: १. लक्षणे २. कारणे ३. उपाय.",
            "hi": f"यह पौधे के {part} की एक छवि है जो बीमार दिखती है। संभावित रोग '{issue}' है। कृपया किसान के लिए सरल हिंदी में समझाएं: 1. लक्षण 2. कारण 3. उपचार।",
            "en": f"This is an image of a plant's {part} that seems diseased. The likely issue is {issue}. Please explain in simple, farmer-friendly English: 1. Symptoms 2. Causes 3. Remedies.",
            "ml": f"ഇത് രോഗബാധിതമായി കാണപ്പെടുന്ന ഒരു ചെടിയുടെ {part}-ന്റെ ചിത്രമാണ്. '{issue}' എന്നതാണ് സാധ്യതയുള്ള പ്രശ്നം. ദയവായി ഒരു കർഷകന് മനസ്സിലാകുന്ന ലളിതമായ മലയാളത്തിൽ വിശദീകരിക്കുക: 1. ലക്ഷണങ്ങൾ 2. കാരണങ്ങൾ 3. പ്രതിവിധികൾ."
        }
        prompt = prompt_templates.get(lang, prompt_templates["en"])
        
        response = llm.invoke(prompt).content
        explanation = re.sub(r"[\*#]", "", response.strip())
        explanation = re.sub(r"\n{3,}", "\n\n", explanation)

        os.remove(temp_path)
        return jsonify({
            "diagnosis": predicted_class,
            "explanation": explanation
        })
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e), 'message': 'Error processing the image'}), 500


# ------------------ File Chat Route ------------------
@app.route('/file-chat', methods=['POST'])
def file_chat():
    try:
        prompt = request.form.get('prompt', '')
        lang = request.form.get('language', 'en')
        file = request.files.get('file')

        if not prompt and not file:
            return jsonify({"reply": "Please provide a question or upload a file."}), 400

        content = ""
        if file:
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            ext = filename.rsplit('.', 1)[1].lower()

            if ext in ['jpg', 'jpeg', 'png']:
                content = "An image of a plant or soil report has been uploaded."
            elif ext == 'pdf':
                # Extract text carefully
                reader = PdfReader(filepath)
                pages_text = []
                for i, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text and text.strip():
                        pages_text.append(text.strip())
                content = "\n\n".join(pages_text)
                if not content:
                    content = "PDF uploaded, but no readable text found."

            os.remove(filepath)

        default_user_question = {
            'en': "Please analyze this and give me advice.",
            'hi': "कृपया इसका विश्लेषण करें और मुझे सलाह दें।",
            'mr': "कृपया याचे विश्लेषण करा आणि मला सल्ला द्या.",
            'ml': "ദയവായി ഇത് വിശകലനം ചെയ്ത് എനിക്ക് ഉപദേശം നൽകുക."
        }

        user_query = prompt if prompt else default_user_question.get(lang, default_user_question['en'])
        file_context = content if content else "No information available."

        messages = []
        if lang == 'mr':
            messages = [
                {"role": "system", "content": "तुम्ही एक कृषी सहाय्यक आहात. फक्त मराठीत प्रतिसाद द्या."},
                {"role": "user", "content": f"शेतकऱ्याचा प्रश्न: '{user_query}'. फाइल माहिती: '{file_context}'"}
            ]
        elif lang == 'hi':
            messages = [
                {"role": "system", "content": "आप एक कृषि सहायक हैं। केवल हिंदी में उत्तर दें।"},
                {"role": "user", "content": f"किसान का सवाल: '{user_query}'. फ़ाइल जानकारी: '{file_context}'"}
            ]
        elif lang == 'ml':
            messages = [
                {"role": "system", "content": "നിങ്ങൾ ഒരു കാർഷിക സഹായിയാണ്. മലയാളത്തിൽ മാത്രം പ്രതികരിക്കുക."},
                {"role": "user", "content": f"കർഷകന്റെ ചോദ്യം: '{user_query}'. ഫയൽ വിവരങ്ങൾ: '{file_context}'"}
            ]
        else:
            messages = [
                {"role": "system", "content": "You are an agricultural assistant. Respond only in English."},
                {"role": "user", "content": f"The farmer's question: '{user_query}'. File context: '{file_context}'"}
            ]

        reply = llm.invoke(messages).content

        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/assistant', methods=['POST'])
def assistant():
    try:
        data = request.get_json() or {}
        prompt = data.get('prompt', '')
        lang = data.get('language', 'en')
        file = request.files.get('file')

        if not prompt and not file:
            return jsonify({"reply": "Please provide a question or upload a file."}), 400

        content = ""
        if file:
            filename = secure_filename(file.filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            ext = filename.rsplit('.', 1)[1].lower()
            if ext in ['jpg', 'jpeg', 'png']:
                content = "An image of a plant or soil report has been uploaded."
            elif ext == 'pdf':
                reader = PdfReader(filepath)
                content = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
            
            os.remove(filepath)

        default_user_question = {
            'en': "Please analyze this and give me advice.",
            'hi': "कृपया इसका विश्लेषण करें और मुझे सलाह दें।",
            'mr': "कृपया याचे विश्लेषण करा आणि मला सल्ला द्या.",
            'ml': "ദയവായി ഇത് വിശകലനം ചെയ്ത് എനിക്ക് ഉപദേശം നൽകുക."
        }
        
        user_query = prompt if prompt else default_user_question.get(lang, default_user_question['en'])
        file_context = content if content else "No information available."

        if lang == 'mr':
            messages = [
                {"role": "system", "content": "तुम्ही एक कृषी सहाय्यक आहात. फक्त मराठीत प्रतिसाद द्या."},
                {"role": "user", "content": f"शेतकऱ्याचा प्रश्न: '{user_query}'. फाइल माहिती: '{file_context}'"}
            ]
        elif lang == 'hi':
            messages = [
                {"role": "system", "content": "आप एक कृषि सहायक हैं। केवल हिंदी में उत्तर दें।"},
                {"role": "user", "content": f"किसान का सवाल: '{user_query}'. फ़ाइल जानकारी: '{file_context}'"}
            ]
        elif lang == 'ml':
            messages = [
                {"role": "system", "content": "നിങ്ങൾ ഒരു കാർഷിക സഹായിയാണ്. മലയാളത്തിൽ മാത്രം പ്രതികരിക്കുക."},
                {"role": "user", "content": f"കർഷകന്റെ ചോദ്യം: '{user_query}'. ഫയൽ വിവരങ്ങൾ: '{file_context}'"}
            ]
        else:
            messages = [
                {"role": "system", "content": "You are an agricultural assistant. Respond only in English."},
                {"role": "user", "content": f"The farmer's question: '{user_query}'. File context: '{file_context}'"}
            ]

        reply = llm.invoke(messages).content
        return jsonify({"reply": reply})

    except Exception as e:
        return jsonify({"reply": f"An unexpected error occurred: {str(e)}"}), 500



# ------------------ Sowing Advice Route ------------------
@app.route('/sowing-advice', methods=['POST'])
def sowing_advice():
    try:
        data = request.get_json()
        forecast = data.get('forecast', [])
        crop = data.get('crop', '').lower()

        if not forecast or not crop:
            return jsonify({'advice': 'Insufficient data for analysis.'}), 400

        avg_temp = sum(day['avgTemp'] for day in forecast) / len(forecast)
        avg_rain = sum(day['rain'] for day in forecast) / len(forecast)
        avg_humidity = sum(day['humidity'] for day in forecast) / len(forecast)
        current_month = datetime.now().month

        prompt = f"""
You are an agricultural expert assistant.
Give short sowing advice based on weather:
- Crop: {crop}
- Month: {current_month}
- Avg Temp: {avg_temp:.1f}°C
- Avg Rain: {avg_rain:.1f} mm
- Avg Humidity: {avg_humidity:.1f}%
Instructions: Mention if month is suitable, check weather, keep 2-3 lines, simple English.
"""

        response = llm.invoke(prompt).content.strip()
        return jsonify({'advice': response})

    except Exception as e:
        return jsonify({'advice': f'Error generating sowing advice: {str(e)}'}), 500


# ------------------ Speak Route ------------------
@app.route('/speak', methods=['POST'])
def speak():
    try:
        data = request.get_json()
        text = data.get("text", "")
        lang = data.get("lang", "ml")  # Malayalam by default

        if not text:
            return jsonify({"error": "No text provided"}), 400

        tts = gTTS(text=text, lang=lang)
        filename = os.path.join(app.config['UPLOAD_FOLDER'], "temp_voice.mp3")
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        tts.save(filename)

        return send_file(filename, mimetype="audio/mpeg")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Health check endpoint for Render
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'model_loaded': model_loaded})


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Log startup information
    logger.info(f"Starting server on port {port}")
    logger.info(f"Using device: {device}")
    
    app.run(host='0.0.0.0', port=port, debug=False)