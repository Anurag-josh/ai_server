# vit_predict.py
import sys
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTImageProcessor, ViTForImageClassification
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import os

# --- Load image path from CLI ---
if len(sys.argv) < 2:
    print("Usage: python vit_predict.py <image_path>")
    sys.exit(1)
image_path = sys.argv[1]

# --- Load class names ---
with open("class_names.txt", "r") as f:
    class_names = f.read().splitlines()

# --- Load ViT model ---
vit_model_path = "./vit_model"
processor = ViTImageProcessor.from_pretrained(vit_model_path)
model = ViTForImageClassification.from_pretrained(vit_model_path)

# --- Preprocess and predict ---
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
predicted_class = model.config.id2label[outputs.logits.argmax().item()]
print(f"\nPredicted class: {predicted_class}")

# --- Use LangChain + Groq to elaborate ---
# Ensure the GROQ API key is provided via environment variable `GROQ_API_KEY`.
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-70b-8192")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful plant disease expert."),
    ("human", "Explain what {label} means and how it affects tomato plants in simple language.")
])

chain = prompt | llm
response = chain.invoke({"label": predicted_class})

print("\nElaborated explanation:\n")
print(response.content)
