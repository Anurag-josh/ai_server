from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import torch

# Load processor and model (will download on first run)
model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto"
)

# Load your image
image_path = "your_image.jpg"
image = Image.open(image_path).convert("RGB")

# Your question about the image
question = "What disease might be affecting this plant?"

# Format input
inputs = processor(text=question, images=image, return_tensors="pt").to("cuda")

# Generate answer
output = model.generate(**inputs, max_new_tokens=200)
answer = processor.decode(output[0], skip_special_tokens=True)

print("Answer:", answer)

