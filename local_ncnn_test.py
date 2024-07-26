import cv2
import numpy as np
from base64 import b64encode
import os
import json
import threading

# Import firebase
import firebase_admin
from firebase_admin import credentials, db

# Import YOLOv8 dependencies
from ultralytics import YOLO


# Function to convert OpenCV image to base64 string
def image_to_base64(img):
    _, buffer = cv2.imencode(".jpg", img)
    img_bytes = buffer.tobytes()
    img_b64 = b64encode(img_bytes).decode("utf-8")
    return img_b64


# Firebase update function
def update_firebase(ref, labels):
    for label in labels:
        label_name = label.split(": ")[0]
        pos = label.split(": ")[1]
        if float(pos) > 0.8:
            if label_name == "fall":
                ref.update({"fall": True})
            elif label_name == "jump":
                ref.update({"jump": True})
            elif label_name == "sitting":
                ref.update({"sitting": True})
            elif label_name == "standing":
                ref.update({"standing": True})
            elif label_name == "walking":
                ref.update({"walking": True})


# Function to start the video stream and perform object detection
def start_video_and_detect():

    #######firebase Setting#################
    # Environment Setting for using firebase
    cred = credentials.Certificate(
        "/Users/sangwon_back/IoT_Project/json/silvercare-84496-firebase-adminsdk-tksu6-bac3439fd8.json"
    )
    # cred = credentials.Certificate('/Users/sangwon_back/Chrome_download/IoT_Project/local_execution/silvercare-84496-firebase-adminsdk-tksu6-bac3439fd8.json')

    app_name = "myApp"

    if app_name not in firebase_admin._apps:
        cur_app = firebase_admin.initialize_app(
            cred,
            {"databaseURL": "https://silvercare-84496-default-rtdb.firebaseio.com/"},
            name=app_name,
        )
    else:
        cur_app = firebase_admin.get_app(app_name)

    # Reference to the database
    ref = db.reference("/", cur_app)
    #######################################

    # Set confidence threshold and NMS threshold
    confidence_threshold = 0.5
    nms_threshold = 0.45

    # Load the YOLOv8 model
    model = YOLO("/Users/sangwon_back/IoT_Project/model/pose_model_ncnn_model")
    # model = YOLO("model/pose_model.pt")

    # webcam
    # cap = cv2.VideoCapture(0)

    # mp4
    cap = cv2.VideoCapture("/Users/sangwon_back/IoT_Project/video/falling_video.mp4")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # cv2.imshow("ff", frame)
        # cv2.waitKey(0)
        # Perform object detection using YOLOv8
        # results = model(frame, verbose=False, device='mps', conf=confidence_threshold, iou=nms_threshold)
        results = model(
            frame,
            verbose=False,
            device="mps",
            conf=confidence_threshold,
            iou=nms_threshold,
        )

        # Get the detected objects
        detections = results[0].boxes.data
        # print("@@", detections)
        # Generate labels for the detected objects
        labels = []
        for detection in detections:
            class_id = int(detection[5])
            print(f"Predicted class_id: {class_id}")  # 디버깅용 출력

            # class_id가 model.names에 존재하는지 확인
            if class_id in model.names:
                class_name = model.names[class_id]
            else:
                print(f"Error: class_id {class_id} not found in model.names")
                class_name = "Unknown"

            confidence = float(detection[4])
            if confidence >= confidence_threshold and class_id <= 10:
                x1, y1, x2, y2 = map(int, detection[:4])
                labels.append(f"{class_name}: {confidence:.2f}")
                # print("@@@@@@")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{class_name}: {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

        # Display the frame with detections
        cv2.imshow("YOLOv8 Object Detection", frame)

        # Update Firebase data in a separate thread
        if labels:
            threading.Thread(target=update_firebase, args=(ref, labels)).start()

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):  # Press 'q' to quit
            break
        elif key == ord("c"):  # Press 'c' to capture and save the image
            filename = "_".join(labels) + ".jpg"
            filename = filename.replace(":", "_")  # Replace colons with underscores
            filename = filename.replace(" ", "_")  # Replace spaces with underscores
            cv2.imwrite(filename, frame)

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()


# Main function
def main():
    # Start the video stream and perform object detection
    start_video_and_detect()


if __name__ == "__main__":
    main()
