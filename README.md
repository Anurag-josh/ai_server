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

## Notes for Production

- The application is configured to run on CPU only in production environments like Render
- All temporary files are stored in the `temp_uploads` directory and cleaned up after processing
- The Vision Transformer model is loaded at startup for optimal performance# ai_server
