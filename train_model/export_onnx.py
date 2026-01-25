from ultralytics import YOLO

# Load the trained model
model_path = r"model_path.pt"
model = YOLO(model_path)

# Export to ONNX
model.export(
    format="onnx",
    half=False,
    dynamic=True, 
)

print(f"\nModel exported to {model_path.replace('.pt', '.onnx')}")
