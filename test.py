from face_detection import detector
from fast_alignment import LandmarkPredictor
import cv2
from face_detection import RetinaFace
import torch

if __name__ == "__main__":
    from contexttimer import Timer
    from fast_alignment.utils import drawLandmark_multiple

    checkpoint = "weights/mobilenet_224_model_best_gdconv_external.pth"

    predictor = LandmarkPredictor(checkpoint, 0)
    detector = RetinaFace(0)

    imgname = "examples/obama.jpg"
    img = cv2.imread(imgname)

    faces = detector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    if len(faces) == 0:
        print("NO face is detected!")
        exit(-1)

    feeds = []

    for face in faces:
        box, landmarks, score = face
        feed = LandmarkPredictor.prepare_feed(img, box)
        feeds.append(feed)
    results = predictor(feeds)
    with Timer() as timer:
        results = predictor(feeds)

    print(timer.elapsed)

    for face, landmarks in zip(faces, results):
        drawLandmark_multiple(img, face[0], landmarks)

    cv2.imshow("", img)
    cv2.waitKey(0)
