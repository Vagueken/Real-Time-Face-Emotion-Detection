import cv2
from keras.models import load_model
import numpy as np
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  

    

def face_detection(img,size=0.5):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)              # converting image into grayscale image
    face_roi = face_detect.detectMultiScale(img_gray, 1.3,1)      # ROI (region of interest of detected face) 
    class_labels = ['Angry','Happy','Neutral','Fear']
                                                              
    
    if face_roi is ():                                            # checking if face_roi is empty that is if no face detected
        return img

    for(x,y,w,h) in face_roi:                                     # iterating through faces and draw rectangle over each face
        x = x - 5
        w = w + 10
        y = y + 7
        h = h + 2
        cv2.rectangle(img, (x,y),(x+w,y+h),(125,125,10), 1)       # (x,y)- top left point  ; (x+w,y+h)-bottom right point  ;  (125,125,10)-colour of rectangle ; 1- thickness 
        img_gray_crop = img_gray[y:y+h,x:x+w]                     # croping gray scale image 
        img_color_crop = img[y:y+h,x:x+w]                         # croping color image
        
        model=load_model('model.h5')
        final_image = cv2.resize(img_color_crop, (48,48))         # size of colured image is resized to 48,48
        final_image = np.expand_dims(final_image, axis = 0)       # array is expanded by inserting axis at position 0
        final_image = final_image/255.0                           # feature scaling of final image
    
        prediction = model.predict(final_image)                   # predicting emotion of captured image from the trained model
        label=class_labels[prediction.argmax()]                   # finding the label of class which has maximaum probalility 
        cv2.putText(frame,label, (50,60), cv2.FONT_HERSHEY_SCRIPT_COMPLEX,2, (120,10,200),3)  
                                                                  # putText is used to draw a detected emotion on image
                                                                  # (50,60)-top left coordinate   FONT_HERSHEY_SCRIPT_COMPLEX-font type
                                                                  # 2-fontscale   (120,10,200)-font colour   3-font thickness


    img_color_crop = cv2.flip(img_color_crop, 1)                  # fliping the image
    return img

cap = cv2.VideoCapture(0)                                         # capturing the video that is live webcam

while True:
    ret, frame = cap.read()
    cv2.imshow('LIVE', face_detection(frame))                     # captured frame will be sent to face_detection function for emotion detection
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()