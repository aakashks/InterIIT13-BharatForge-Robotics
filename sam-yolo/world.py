from ultralytics import YOLOWorld

# Initialize a YOLO-World model
model = YOLOWorld("yolov8s-world.pt")

# Define custom classes
model.set_classes(["person", "bus", "fire extinguisher", "stairs", "extinguisher"])

# Execute prediction on an image
results = model.predict("image3.png")

# Show results
results[0].show()