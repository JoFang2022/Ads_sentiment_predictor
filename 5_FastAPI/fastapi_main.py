import io
import cv2
import uvicorn
from fastapi import FastAPI, File
from tempfile import NamedTemporaryFile
from prediction import VideosPredictor

# To run: uvicorn fastapi_main:app
# http://0.0.0.0:8000/docs

app = FastAPI(debug=True)
app.video_predictor = VideosPredictor()


@app.post("/")
def main(file: bytes = File(...)):
    binary_video = io.BytesIO(file)
    tfile = NamedTemporaryFile(delete=True)
    tfile.write(binary_video.read())
    cap = cv2.VideoCapture(tfile.name)
    while cap.isOpened() is True:
        result = app.video_predictor.get_prediction(tfile.name)
        return {"The predicted sentiment of this video is: " + result}


if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=8000)
