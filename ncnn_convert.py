from ultralytics import YOLO

# # Load a YOLOv8n PyTorch model
# model = YOLO("./model/pose_model.pt")

# # Export the model to NCNN format
# model.export(format="ncnn")  # creates 'yolov8n_ncnn_model'

# Load the exported NCNN model
ncnn_model = YOLO("./model/pose_model_ncnn_model")
print(ncnn_model)
model = YOLO('model/pose_model.pt')
print(model)
# # Run inference
# results = ncnn_model("./video/falling_video.mp4")