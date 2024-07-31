import cv2
import numpy as np
from base64 import b64encode
import os
import json
import threading
import socket

# Import flask
from flask import Flask, render_template, Response
import firebase_admin
from firebase_admin import credentials, db
from ultralytics import YOLO


#######firebase Setting#################
# Environment Setting for using firebase
cred = credentials.Certificate(
    "./json/silvercare-84496-firebase-adminsdk-tksu6-bac3439fd8.json"
)
app_name = "myApp123"

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


# state of positions
isfall = False
issitting = False
iswalking = False
isstanding = False
isjump = False

port_value = 5000

# 프레임을 저장할 전역 변수
global_frame = None

# Flask 애플리케이션 설정
app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


def get_server_address():
    global port_value
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return f"http://{local_ip}:{port_value}"


# Firebase 업데이트 함수
def update_firebase(ref, labels):
    global isfall, issitting, iswalking, isstanding, isjump

    for label in labels:
        label_name = label.split(": ")[0]
        pos = label.split(": ")[1]
        if float(pos) > 0.8:
            if label_name == "fall":
                ref.update({"fall": True})
                isfall = True
            elif label_name == "jump":
                ref.update({"jump": True})
                isjump = True
            elif label_name == "sitting":
                ref.update({"sitting": True})
                issitting = True
            elif label_name == "standing":
                ref.update({"standing": True})
                isstanding = True
            elif label_name == "walking":
                ref.update({"walking": True})
                iswalking = True
        else:
            if isfall:
                ref.update({"fall": False})
                isfall = False
            if isjump:
                ref.update({"jump": False})
                isjump = False
            if isstanding:
                ref.update({"standing": False})
                isstanding = False
            if issitting:
                ref.update({"sitting": False})
                issitting = False
            if iswalking:
                ref.update({"walking": False})
                iswalking = False


# 비디오 스트림과 객체 감지를 시작하는 함수
def start_video_and_detect():
    global global_frame

    ####### Firebase 설정 ########
    cred = credentials.Certificate(
        "./json/silvercare-84496-firebase-adminsdk-tksu6-bac3439fd8.json"
    )
    app_name = "myApp123"

    if app_name not in firebase_admin._apps:
        cur_app = firebase_admin.initialize_app(
            cred,
            {"databaseURL": "https://silvercare-84496-default-rtdb.firebaseio.com/"},
            name=app_name,
        )
    else:
        cur_app = firebase_admin.get_app(app_name)

    ref = db.reference("/", cur_app)
    ################################

    confidence_threshold = 0.5
    nms_threshold = 0.45
    model = YOLO("./model/pose_model.pt")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 1)

    while True:
        for _ in range(5):
            cap.read()

        ret, frame = cap.read()
        if not ret:
            break

        results = model(
            frame,
            verbose=False,
            device="cpu",
            conf=confidence_threshold,
            iou=nms_threshold,
        )

        detections = results[0].boxes.data
        labels = []
        for detection in detections:
            class_id = int(detection[5])
            class_name = model.names[class_id]
            confidence = float(detection[4])
            if confidence >= confidence_threshold:
                x1, y1, x2, y2 = map(int, detection[:4])
                labels.append(f"{class_name}: {confidence:.2f}")
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

        if labels:
            threading.Thread(target=update_firebase, args=(ref, labels)).start()

        global_frame = frame  # 전역 변수에 프레임 저장

    cap.release()


# 비디오 피드를 웹 페이지로 스트리밍하는 함수
def generate_video_feed():
    global global_frame

    while True:
        if global_frame is not None:
            ret, buffer = cv2.imencode(".jpg", global_frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_video_feed(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    address = get_server_address()
    ref.update({"flask_url": address})
    print(f"Server running at {address}")

    # 비디오 감지 기능을 별도의 스레드로 실행
    threading.Thread(target=start_video_and_detect, daemon=True).start()
    # Flask 애플리케이션 실행
    try:
        app.run(debug=True, host="0.0.0.0", port=port_value)
    except:
        port_value += 100
        app.run(debug=True, host="0.0.0.0", port=port_value)