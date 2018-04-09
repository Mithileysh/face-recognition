import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);


cam = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX
#cv2.putText(img,"Cat",(x,y-10),font,0.55,(0,255,0),1)
#font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)


while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if Id==1:
            Id="Mush"
        elif Id==2:
            Id="elon"
        elif Id==3:
            Id="vijay SIR"
        elif Id==4:
            Id="shekher"
        else:
            Id="Unknown"
        
        cv2.putText(im,str(Id),(x,y+h),font,0.55,(0,255,0),1)###
    
    cv2.imshow('windowname',im) 
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break


cam.release()
cv2.destroyAllWindows()

