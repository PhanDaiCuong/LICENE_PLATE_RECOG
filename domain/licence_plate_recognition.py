import numpy as np
import cv2
import time
from configs.config import YOLOv8, OCR

class Process:
    def __init__(self,yolo_session,ocr_session, characters):
        self.characters = characters
        self.yolo_session = yolo_session
        self.ocr_session = ocr_session

    # -------------------- IMAGE PROCESSING --------------------
    def process_image(self,image):
        self.image = image
        # YOLO inference
        detections = YOLOv8(self.image, self.yolo_session ).main()
        all_crop_images = []
        all_bboxes = []
        plate_results = []
        for box in detections:
            x, y, w, h = box
            x2 = int(x + w)
            y2 = int(y + h)
            x1 = int(x)
            y1 = int(y) 
            all_bboxes.append((x1, y1, x2, y2))
            all_crop_images.append(self.image[y1:y2, x1:x2])
            
        plates = OCR(self.ocr_session, self.characters).main(all_crop_images)
        for (x1,y1,x2,y2), plate in zip(all_bboxes, plates):
            plate_results.append({
                                "plate_text": plate,
                                "bbox": [x1, y1, x2, y2]
                            })
        return plate_results


    # -------------------- VIDEO PROCESSING --------------------

    def process_video(self, video: str, frame_stride: int = 1):
        cap = cv2.VideoCapture(int(video)) if str(video).isdigit() else cv2.VideoCapture(video)
        if not cap.isOpened():
            raise RuntimeError("[process_video] Không thể mở video: {}".format(video))

        plate_results = []
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_stride == 0:
                    start = time.time()
                    detections = YOLOv8(frame, self.yolo_session).main()  # placeholder

                    crops = []
                    bboxs = []
                    for box in detections:
                        x, y, w, h = map(int, box)
                        x1, y1 = max(0, x), max(0, y)
                        x2, y2 = x1 + max(0, w), y1 + max(0, h)
                        x2 = min(frame.shape[1]-1, x2)
                        y2 = min(frame.shape[0]-1, y2)
                        bboxs.append([x1, y1, x2, y2])
                        # crop safe
                        crops.append(frame[y1:y2, x1:x2])

                    # --- OCR trên các crops ---
                    plates = []
                    if crops:
                        plates = OCR(self.ocr_session, self.characters).main(crops)  # placeholder

                    # build results
                    for maybe_text, box in zip(plates, bboxs):
                        text = None
                        if maybe_text is None:
                            text = ""
                        elif isinstance(maybe_text, (list, tuple)):
                            txt0 = maybe_text[0] if len(maybe_text) > 0 else ""
                            text = txt0 if isinstance(txt0, str) else str(txt0)
                        else:
                            text = str(maybe_text)

                        plate_results.append({
                            "frame_id": frame_idx,
                            "plate_text": text,
                            "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])]
                        })

                    elapsed = time.time() - start
                frame_idx += 1

        finally:
            cap.release()

        return plate_results


