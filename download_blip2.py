from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download BLIP-2 model (Flan-T5 XL variant)
model_id = "Salesforce/blip2-opt-2.7b"

print("🔽 Downloading BLIP-2 model and processor...")
processor = Blip2Processor.from_pretrained(model_id)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
).to(device)
print("✅ BLIP-2 model downloaded and ready.")
