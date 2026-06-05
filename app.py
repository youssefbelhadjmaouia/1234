from fastapi import FastAPI, Request
from ultralytics import YOLO
from PIL import Image
import io
import traceback

app = FastAPI()

model = YOLO("best.pt")

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/detect")
async def detect(request: Request):

    try:
        image_bytes = await request.body()

        print(f"Received bytes: {len(image_bytes)}")

        image = Image.open(io.BytesIO(image_bytes))

        print(f"Image size: {image.size}")

        results = model(image, conf=0.1)

        detections = []

        for r in results:
            for box in r.boxes:
                detections.append({
                    "class": model.names[int(box.cls[0])],
                    "confidence": float(box.conf[0])
                })

        return {
            "success": True,
            "detections": detections
        }

    except Exception as e:
        print(traceback.format_exc())

        return {
            "success": False,
            "error": str(e)
        }
