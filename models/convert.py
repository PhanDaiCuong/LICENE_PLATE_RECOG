from ultralytics import YOLO

model_path = 'best.pt'
output_format = 'onnx'

model = YOLO(model_path)  

model.export(format=output_format, opset=16)

