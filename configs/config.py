import onnxruntime as ort
import numpy as np
import cv2
import re
from typing import List, Tuple

# Path to the YOLO and OCR model (must be downloaded beforehand)
PATH_MODEL_DETECT= "./models/yolov8_f32.onnx"
PATH_MODEL_RECOGNITION= './models/rec_model_f32.onnx'



# ========================== CONFIGURATION PARAMETERS ==========================
FRAME_SKIP = 2  # Number of frames to skip before processing (reduces computational load)
CHARACTER_DICT_PATH = "configs/en_dict.txt"
MAX_TEXT_LENGTH = 25
IMAGE_SHAPE = (3, 48, 320)
CONFIDENCE_THRESHOLD = 0.3  # Confidence threshold to retain detected faces
STREAM = True  # Enable streaming mode for processing video frames
IMAGE_SIZE_YOLO = (480, 640)  # Target image size (height, width) for model input


#define providers
if "CUDAExecutionProvider" in ort.get_available_providers():
    providers=["CUDAExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]



# ========  Load từ điển ký tự =========
def load_character_dict(use_space_char=True):
    with open(CHARACTER_DICT_PATH, 'r', encoding='utf-8') as f:
        characters = [line.strip('\n') for line in f]
    if use_space_char:
        characters.append(" ")  # Thêm ký tự space
    return characters


def create_model_detect():
    """
    Load the YOLO ONNX model and move it to the selected device (GPU or CPU).

    :return: YOLO model instance ready for inference.
    """
    # Load model YOLO ONNX
    yolo_session = ort.InferenceSession(PATH_MODEL_DETECT, providers=providers)

    return yolo_session



def create_model_recog():
    # Load model OCR ONNX
    ocr_session = ort.InferenceSession(PATH_MODEL_RECOGNITION, providers=providers)
    return ocr_session

class YOLOv8:
    """
    YOLOv8 object detection model class for handling inference and visualization.

    This class provides functionality to load a YOLOv8 ONNX model, perform inference on images,
    and visualize the detection results.

    Attributes:
        onnx_model (str): Path to the ONNX model file.
        input_image (str): Path to the input image file.
        confidence_thres (float): Confidence threshold for filtering detections.
        iou_thres (float): IoU threshold for non-maximum suppression.
        classes (List[str]): List of class names from the COCO dataset.
        color_palette (np.ndarray): Random color palette for visualizing different classes.
        input_width (int): Width dimension of the model input.
        input_height (int): Height dimension of the model input.
        img (np.ndarray): The loaded input image.
        img_height (int): Height of the input image.
        img_width (int): Width of the input image.
    """

    def __init__(self, input_image: np.ndarray, yolo_session, confidence_thres:float = 0.5, iou_thres: float = 0.5):
        """
        Initialize an instance of the YOLOv8 class.

        Args:
            onnx_model (str): Path to the ONNX model.
            input_image (str): Path to the input image.
            confidence_thres (float): Confidence threshold for filtering detections.
            iou_thres (float): IoU threshold for non-maximum suppression.
        """
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.session = yolo_session
        # Load the class names from the COCO dataset
        self.classes = ['licence_plate']

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.img = input_image
    def letterbox(self, img: np.ndarray, new_shape: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Resize and reshape images while maintaining aspect ratio by adding padding.

        Args:
            img (np.ndarray): Input image to be resized.
            new_shape (Tuple[int, int]): Target shape (height, width) for the image.

        Returns:
            (np.ndarray): Resized and padded image.
            (Tuple[int, int]): Padding values (top, left) applied to the image.
        """
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top, left)

    def draw_detections(self, img: np.ndarray, box: List[float], score: float, class_id: int) -> None:
        """
        Draw bounding boxes and labels on the input image based on the detected objects.

        Args:
            img (np.ndarray): The input image to draw detections on.
            box (List[float]): Detected bounding box coordinates [x, y, width, height].
            score (float): Confidence score of the detection.
            class_id (int): Class ID for the detected object.
        """
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess the input image before performing inference.

        This method reads the input image, converts its color space, applies letterboxing to maintain aspect ratio,
        normalizes pixel values, and prepares the image data for model input.

        Returns:
            (np.ndarray): Preprocessed image data ready for inference with shape (1, 3, height, width).
            (Tuple[int, int]): Padding values (top, left) applied during letterboxing.
        """

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        img, pad = self.letterbox(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data, pad

    def postprocess(self, input_image: np.ndarray, output: List[np.ndarray], pad: Tuple[int, int]) -> np.ndarray:
        """
        Perform post-processing on the model's output to extract and visualize detections.

        This method processes the raw model output to extract bounding boxes, scores, and class IDs.
        It applies non-maximum suppression to filter overlapping detections and draws the results on the input image.

        Args:
            input_image (np.ndarray): The input image.
            output (List[np.ndarray]): The output arrays from the model.
            pad (Tuple[int, int]): Padding values (top, left) used during letterboxing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        gain = min(self.input_height / self.img_height, self.input_width / self.img_width)
        outputs[:, 0] -= pad[1]
        outputs[:, 1] -= pad[0]

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) / gain)
                top = int((y - h / 2) / gain)
                width = int(w / gain)
                height = int(h / gain)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)
        all_bboxs = []
        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            all_bboxs.append(box)

        # Return the modified input image
        return all_bboxs

    def main(self) -> np.ndarray:
        """
        Perform inference using an ONNX model and return the output image with drawn detections.

        Returns:
            (np.ndarray): The output image with drawn detections.
        """
        # Get the model inputs
        model_inputs = self.session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # Preprocess the image data
        img_data, pad = self.preprocess()

        # Run inference using the preprocessed image data
        outputs = self.session.run(None, {model_inputs[0].name: img_data})

        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(self.img, outputs, pad)  # output image



class OCR:
    def __init__(self, ocr_session, characters, img_shape=(48, 320), batch_size=8):
        self.session = ocr_session
        self.input_name = self.session.get_inputs()[0].name
        self.img_shape = img_shape  # (H, W)
        self.batch_size = batch_size
        self.characters = characters

    def preprocess(self, image):
        target_h, target_w = self.img_shape
        h, w, _ = image.shape

        # Tính tỉ lệ resize theo chiều cao
        scale = target_h / h
        new_w = int(w * scale)

        if new_w > target_w:
            # Resize theo chiều rộng nếu quá to
            scale = target_w / w
            new_h = int(h * scale)
            img_resized = cv2.resize(image, (target_w, new_h))
            pad_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            pad_img[y_offset:y_offset + new_h, :, :] = img_resized
        else:
            # Resize giữ nguyên chiều cao, pad chiều rộng
            img_resized = cv2.resize(image, (new_w, target_h))
            pad_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            x_offset = (target_w - new_w) // 2
            pad_img[:, x_offset:x_offset + new_w, :] = img_resized

        # Normalize và chuyển sang (C, H, W)
        img_norm = pad_img.astype(np.float32) / 255.0
        img_norm = np.transpose(img_norm, (2, 0, 1))  # (H, W, C) → (C, H, W)
        return img_norm

    import re

    def ctc_decode(self, output):
        results = []
        batch_size = output.shape[0]

        for i in range(batch_size):
            pred = np.argmax(output[i], axis=1)
            max_probs = np.max(output[i], axis=1)

            last_char = -1
            text = ""
            score_sum = 0.0
            count = 0

            for j, char_id in enumerate(pred):
                if char_id != last_char and char_id != 0:
                    if (char_id - 1) < len(self.characters):
                        char = self.characters[char_id - 1]
                        if re.match(r'^[A-Za-z0-9]$', char):  # chỉ giữ chữ và số
                            text += char
                            score_sum += max_probs[j]
                            count += 1
                last_char = char_id

            avg_score = float(score_sum) / count if count > 0 else 0.0
            if avg_score > 0.6:
                results.append((text, avg_score))

        return results

    def main(self, imgs):
        all_results = []
        for i in range(0, len(imgs), self.batch_size):
            batch_imgs = imgs[i : i + self.batch_size]
            batch_input = np.stack([self.preprocess(im) for im in batch_imgs], axis=0)  # (B, C, H, W)
            outputs = self.session.run(None, {self.input_name: batch_input})
            results = self.ctc_decode(outputs[0])
            all_results.extend(results)
        return all_results





