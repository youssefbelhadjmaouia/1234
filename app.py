from fastapi import FastAPI, Request
from ultralytics import YOLO
from PIL import Image
import io

print("Loading model...")

model = YOLO("best.pt")

print("Model loaded!")

app = FastAPI()

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/detect")
async def detect(request: Request):

    image_bytes = await request.body()

    image = Image.open(io.BytesIO(image_bytes))

    results = model(image, conf=0.10)

    detections = []

    for r in results:
        for box in r.boxes:

            detections.append({
                "class": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0])
            })

    return {"detections": detections}
