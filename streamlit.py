import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
import io,json, cv2, tempfile
import numpy as np

API_URL = "http://localhost:8000/recognize/"

st.title("License Plate Recognition App")

option = st.radio("Select input type:", ("Image", "Video", "Webcam"))

def draw_frame(frame, result):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for box in result:
        x1, y1, x2, y2 = box["bbox"]
        text = box["plate_text"]

        draw.rectangle([x1, y1, x2, y2], outline="green", width=5)
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 32)
        draw.text((x1, y1 - 40), str(text), fill=(0,255,0), font=font)
    return img

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        response = requests.post(
            API_URL,
            files={"file": (uploaded_file.name, file_bytes)},
            data={"input_type": "image"}
        )

        if response.status_code == 200:
            results = response.json().get("results") 

            # Hiển thị ảnh đầu ra đã vẽ bbox và text
            np_arr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            image = draw_frame(image, results)
            st.image(image, caption='Detected License Plate', use_container_width=False)
        else:
            st.error("Recognition error!")

elif option == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()

        # gửi video lên backend
        response = requests.post(
            API_URL,
            files={"file": (uploaded_file.name, file_bytes)},
            data={"input_type": "video"},
            stream=True,
            timeout=120  
        )

        if response.status_code == 200:
            results = response.json()["results"]

            # Lưu tạm video để cv2 đọc
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name

            cap = cv2.VideoCapture(tmp_path)

            frame_id = 0
            placeholder = st.empty()
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                bboxes = [r for r in results if r["frame_id"] == frame_id]

                fram_result = draw_frame(frame, bboxes)
                placeholder.image(fram_result, channels="RGB")
                frame_id += 1
            cap.release()
        else:
            st.error("Recognition error!")

elif option == "Webcam":
    webcam_image = st.camera_input("Take a picture")
    if webcam_image is not None:
        file_bytes = webcam_image.getvalue()
        response = requests.post(
            API_URL,
            files={"file": ("webcam.jpg", file_bytes)},
            data={"input_type": "webcam"}
        )
        if response.status_code == 200:
            recognized_plate = response.json().get("license_plate")
            z = response.json().get("z")
            image = Image.open(io.BytesIO(file_bytes))
            draw = ImageDraw.Draw(image)
            if z:
                x1, y1, x2, y2 = z
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                text = recognized_plate if recognized_plate else "N/A"
                draw.text((x1, y1 - 20), str(text), fill="green")
            st.image(image, caption='Detected License Plate', use_container_width=True)
            st.success(f"Recognized License Plate: {recognized_plate}")
        else:
            st.error("Recognition error!")