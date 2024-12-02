import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50
from clip import clip

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device)

# Load the image
image = Image.open("image.png")
image_input = transform(image).unsqueeze(0).to(device)

# Load the text
classes = ["carton", "dog", "cat", "human", "stairs", "staircase", "forklift", "person", "flower pot", "fire extinguisher"]
text = clip.tokenize(classes).to(device)

# Calculate features
with torch.no_grad():
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text)
    
# Calculate similarity
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# Print top 5 most similar labels
values, indices = similarity[0].topk(2)
print("\nTop predictions:\n")

for value, index in zip(values, indices):
    print(f"{classes[index]:<16} {100 * value.item():.2f}%")