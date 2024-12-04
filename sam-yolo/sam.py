from ultralytics import SAM

# Load a model
model = SAM("sam2.1_b.pt")

# Display model information (optional)
model.info()

# Run inference on a single image
# results = model("image2.png")

model.predict("image2.png", save=True)
