from domain.licence_plate_recognition import process_image, process_video
import os
from configs.config import load_character_dict, create_model_detect, create_model_recog
import cv2

def is_image_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in image_exts

def is_video_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in video_exts

def get_output_path(input_path, input_root='test', output_root='output'):
    rel_path = os.path.relpath(input_path, start=input_root)
    output_path = os.path.join(output_root, rel_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    return output_path

if __name__ == "__main__":
    dir_path = './test'
    input_path = os.path.join(dir_path, 'videos/video1.mp4')

    output_dir = './output'

    image_exts = ['.jpg', '.jpeg', '.png']
    video_exts = ['.mp4']

    yolo_session= create_model_detect()
    ocr_session= create_model_recog()
    output_path = get_output_path(input_path)
    characters = load_character_dict()


    if is_image_file(input_path):
        input_image = cv2.imread(input_path)

        process_image(input_image, output_path, yolo_session, ocr_session, characters)
    elif is_video_file(input_path):
        process_video(input_path, output_path,yolo_session, ocr_session, characters)
    else:
        print("[ERROR] Unsupported file format!")