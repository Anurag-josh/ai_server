# Agricultural AI Assistant API

This is a Flask-based API for agricultural assistance that provides:
- Plant disease diagnosis using Vision Transformer models
- Agricultural advice using LLMs
- Weather-based sowing recommendations
- Text-to-speech functionality

## Deployment on Render

This application is ready for deployment on Render with the following configuration:

### Environment Variables Required:
- `GROQ_API_KEY`: Your Groq API key for LLM integration

### Build Process:
- Render will automatically install dependencies from `requirements.txt`
- The application will be served using Gunicorn as specified in the `Procfile`

### Port Configuration:
- The application automatically uses the port specified by Render's `PORT` environment variable

## API Endpoints

- `POST /predict`: Plant disease diagnosis from images
- `POST /file-chat`: File-based agricultural advice
- `POST /assistant`: General agricultural assistant
- `POST /sowing-advice`: Weather-based sowing recommendations
- `POST /speak`: Text-to-speech conversion

## Setup Instructions

### For Local Development:
1. Clone the repository:
   ```bash
   git clone https://github.com/Anurag-josh/ai_server.git
   cd ai_server
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the required model files:
   - Vision Transformer model: Place in `vit_model/` directory
   - Alternative ViT model: Place in `vit-model/` directory
   - Fine-tuned crop model: Place in `vit-finetuned-crops/` directory
   
   You can use the `download_blip2.py` script to download the BLIP-2 model, or download the ViT models from Hugging Face.

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your GROQ_API_KEY
   ```

5. Run the application:
   ```bash
   python app.py
   ```

### For Deployment on Render:
1. Create a new Web Service on Render
2. Connect to your GitHub repository
3. Set the environment variable:
   - `GROQ_API_KEY`: Your Groq API key
4. Use the following build settings:
   - Environment: `Python`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --keep-alive 5 app:app`
5. The application will be available at the URL provided by Render

## Notes for Production

- The application is configured to run on CPU only in production environments like Render
- All temporary files are stored in the `temp_uploads` directory and cleaned up after processing
- The Vision Transformer model is loaded at startup for optimal performance
