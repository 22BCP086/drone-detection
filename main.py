from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import torch
import io
from PIL import Image
import numpy as np
import base64
import uvicorn

app = FastAPI()

# Load your YOLOv5 model
print("🔹 Loading model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=False)
print("✅ Model loaded successfully.")

@app.post("/detect")
async def detect(request: Request):
    # Get raw bytes
    raw_bytes = await request.body()

    # Convert raw bytes to image
    try:
        img = Image.open(io.BytesIO(raw_bytes))
    except Exception:
        # If it's not a clean image, try to decode it manually (base64 or binary)
        try:
            decoded = base64.b64decode(raw_bytes)
            img = Image.open(io.BytesIO(decoded))
        except Exception as e:
            return JSONResponse(content={"error": f"Image decode failed: {str(e)}"}, status_code=400)

    # Run YOLO inference
    results = model(img)

    # Convert results to dictionary
    detections = results.pandas().xyxy[0].to_dict(orient="records")

    # Send back only drone detections (optional filter)
    drones = [d for d in detections if d['name'].lower() == 'drone']

    return JSONResponse(content={"detections": drones or detections})
    

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
