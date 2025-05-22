import numpy as np
import cv2
import time
from configs.config import YOLOv8, OCR


def process_image(image, output_path, yolo_session, ocr_session, characters):
    # YOLO inference
    detections = YOLOv8(image, yolo_session ).main()
    # Perform object detection and obtain the output image
    all_crop_images = []
    for box in detections:
        x, y, w, h = box
        x, y, w, h = int(x), int(y), int(w), int(h)

        # Tính tọa độ góc dưới bên phải
        x2 = x + w
        y2 = y + h
        all_crop_images.append(image[y:y2, x:x2])
    plates = OCR(ocr_session, characters).main(all_crop_images)
    print(plates)


# -------------------- VIDEO PROCESSING --------------------

def process_video(video_path, save_path, yolo_session, ocr_session, characters):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Không thể mở video!")
        return

    # Lấy thông tin khung hình gốc để khởi tạo VideoWriter
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Hoặc 'XVID', 'avc1' tùy vào codec bạn cài

    # Kích thước sau resize
    out_w, out_h = 1280, 720
    out = cv2.VideoWriter(save_path, fourcc, fps, (out_w, out_h))

    frame_nmr = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        detections = YOLOv8(frame, yolo_session).main()

        all_crop_images = []
        all_bboxs = []
        for box in detections:
            x, y, w, h = map(int, box)
            x2, y2 = x + w, y + h
            all_bboxs.append([x, y, x2, y2])
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 5)
            all_crop_images.append(frame[y:y2, x:x2])

        plates = OCR(ocr_session, characters).main(all_crop_images)
        plate_results = []
        for text, box in zip(plates, all_bboxs):
            x, y, _, _ = box
            if text:
                cv2.putText(frame, str(text[0]), (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 0), 5)
                plate_results.append({'text': text, 'bbox': box})

        # Ghi FPS lên góc phải trên
        frame_time = time.time() - start_time
        current_fps = 1.0 / frame_time if frame_time > 0 else 0
        fps_text = f"FPS: {current_fps:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        thickness = 4
        text_size, _ = cv2.getTextSize(fps_text, font, font_scale, thickness)
        x_pos = frame.shape[1] - text_size[0] - 50
        y_pos = 90
        cv2.putText(frame, fps_text, (x_pos, y_pos), font, font_scale, (255, 180, 100), thickness)

        # Resize và ghi frame vào video
        frame_resized = cv2.resize(frame, (out_w, out_h))
        out.write(frame_resized)

        print(f"[FRAME {frame_nmr}] FPS: {current_fps:.2f}")
        if plate_results:
            for plate in plate_results:
                print(f"Biển số: {plate['text']}, BBox: {plate['bbox']}")
        else:
            print("Không phát hiện biển số nào sau khi OCR.")

        frame_nmr += 1

    cap.release()
    out.release()
    print(f"✅ Video đã xử lý xong và lưu tại: {save_path}")


