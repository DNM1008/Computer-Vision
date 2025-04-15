from ultralytics import YOLO

pt_path = "../data/results/big_model_11/weights/best_m.pt"

pt_model = YOLO(pt_path)
onnx_model = pt_model.export(format="onnx")
