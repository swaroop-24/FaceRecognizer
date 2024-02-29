import cv2
import numpy as np

faceDetect = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('trainer/trainer.yml')
id = 0
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
        id, conf = rec.predict(gray[y:y + h, x:x + w])
        if conf < 50:
            if id == 1:
                id = "Sheikh Zayed"
            elif id == 2:
                id = "Steve Jobs"
            elif id == 3:
                id = ""
            elif id == 4:
                id = ""
            elif id == 5:
                id = ""
            elif id == 6:
                id = ""
            elif id == 7:
                id = ""
            elif id == 8:
                id = ""
            elif id == 9:
                id = ""
        else:
            id = "Unknown"
        cv2.putText(img, str(id), (x, y + h), font, 3, (0, 0, 255), thickness=4)
    cv2.imshow('Face', img)
    if cv2.waitKey(10) == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
