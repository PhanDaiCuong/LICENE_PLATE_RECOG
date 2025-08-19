from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, StreamingResponse
import cv2
import numpy as np
from domain.licence_plate_recognition import Process
from configs.config import load_character_dict, create_model_detect, create_model_recog
import tempfile, os, json


yolo_session= create_model_detect()
ocr_session= create_model_recog()
characters = load_character_dict()
app = FastAPI()
processor = Process(yolo_session, ocr_session, characters)

@app.post("/recognize/")
async def recognize(
    file: UploadFile = File(...),
    input_type: str = Form("image")  # "image", "video", "webcam"
):
    ext = file.filename.split('.')[-1].lower()
    contents = await file.read()

    if input_type == "image":
        np_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        plate_results = processor.process_image(image)
        return JSONResponse(content={"results": plate_results})

    elif input_type == "video":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
            tmp_in.write(contents)
            tmp_in_path = tmp_in.name
        try:
            plate_results = processor.process_video(tmp_in_path)

            return JSONResponse(content={"results": plate_results})
        except Exception as e:
            return JSONResponse(content={"error": f"processing error: {str(e)}"}, status_code=500)
        finally:
            try:
                if os.path.exists(tmp_in_path):
                    os.remove(tmp_in_path)
            except Exception:
                pass

    elif input_type == "webcam":
        np_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        recognized_plate = processor.process_video(image)
        return JSONResponse(content={"license_plate": recognized_plate})

    else:
        return JSONResponse(content={"error": "Invalid input type"}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

