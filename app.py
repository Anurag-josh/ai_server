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
import logging

# ------------------ App Setup ------------------
app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = "temp_uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "gif"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ------------------ Device ------------------
device = torch.device("cpu")  # Render has no GPU

# ------------------ Lazy Globals ------------------
processor = None
model = None
model_loaded = False
llm = None


# ------------------ Fast Root Route (CRITICAL) ------------------
@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "running",
        "message": "AI server is up"
    })


# ------------------ Health Check ------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model_loaded
    })


# ------------------ Helpers ------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]


def load_vit_model():
    global processor, model, model_loaded

    if model_loaded:
        return

    if not os.path.exists("vit_model"):
        print("⚠️ vit_model directory not found")
        return

    print("⏳ Lazy loading Vision Transformer...")
    processor = ViTImageProcessor.from_pretrained("vit_model")
    model = ViTForImageClassification.from_pretrained("vit_model").to(device)
    model_loaded = True
    print("✅ Vision Transformer loaded")


def get_llm():
    global llm
    if llm is None:
        print("⏳ Initializing Groq LLM...")
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY")
        )
        print("✅ LLM ready")
    return llm


# ------------------ Predict ------------------
@app.route("/predict", methods=["POST"])
def predict():
    load_vit_model()

    if not model_loaded:
        return jsonify({"error": "Vision model not available"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    try:
        image = Image.open(path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            idx = outputs.logits.argmax(-1).item()
            prediction = model.config.id2label[idx]

        os.remove(path)
        return jsonify({"diagnosis": prediction})

    except Exception as e:
        if os.path.exists(path):
            os.remove(path)
        return jsonify({"error": str(e)}), 500


# ------------------ Assistant ------------------
@app.route("/assistant", methods=["POST"])
def assistant():
    data = request.get_json() or {}
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"reply": "Prompt required"}), 400

    reply = get_llm().invoke(prompt).content
    return jsonify({"reply": reply})


# ------------------ File Chat ------------------
@app.route("/file-chat", methods=["POST"])
def file_chat():
    prompt = request.form.get("prompt", "")
    file = request.files.get("file")
    content = ""

    if file:
        filename = secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        if filename.lower().endswith(".pdf"):
            reader = PdfReader(path)
            content = "\n".join(
                page.extract_text()
                for page in reader.pages
                if page.extract_text()
            )

        os.remove(path)

    final_prompt = f"{prompt or 'Please analyze and advise.'}\n\nContext:\n{content}"
    reply = get_llm().invoke(final_prompt).content
    return jsonify({"reply": reply})


# ------------------ Sowing Advice ------------------
@app.route("/sowing-advice", methods=["POST"])
def sowing_advice():
    data = request.get_json() or {}
    forecast = data.get("forecast", [])
    crop = data.get("crop", "")

    if not forecast or not crop:
        return jsonify({"advice": "Insufficient data"}), 400

    avg_temp = sum(d["avgTemp"] for d in forecast) / len(forecast)
    avg_rain = sum(d["rain"] for d in forecast) / len(forecast)
    avg_humidity = sum(d["humidity"] for d in forecast) / len(forecast)
    month = datetime.now().month

    prompt = f"""
Crop: {crop}
Month: {month}
Avg Temp: {avg_temp:.1f}
Avg Rain: {avg_rain:.1f}
Avg Humidity: {avg_humidity:.1f}

Give short sowing advice.
"""

    advice = get_llm().invoke(prompt).content
    return jsonify({"advice": advice})


# ------------------ Speak ------------------
@app.route("/speak", methods=["POST"])
def speak():
    data = request.get_json() or {}
    text = data.get("text", "")
    lang = data.get("lang", "ml")

    if not text:
        return jsonify({"error": "Text required"}), 400

    filename = os.path.join(app.config["UPLOAD_FOLDER"], "speech.mp3")
    gTTS(text=text, lang=lang).save(filename)
    return send_file(filename, mimetype="audio/mpeg")


# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO)


