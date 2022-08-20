import cv2

def detect_face_eyes(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert from BGR to grayscale  
    
    # detect faces
    faces = face_cascade.detectMultiScale(img_gray, 
                                               scaleFactor=1.05, 
                                               minNeighbors=5,
                                               minSize=(30, 30),
                                               flags=cv2.CASCADE_SCALE_IMAGE)
    count = 0 # count number of faces
    
    for (x,y,w,h) in faces:
        count +=1
        
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 10) # red rectangle on face
        
        faceROI = img_gray[y:y+h, x:x+w] # region of interest of face to use to detect eyes
        
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2) * 0.25))
            cv2.circle(img, eye_center, radius, (255,0,0), 4)
            
    return img, count


cap = cv2.VideoCapture(0)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Draw a rectangle on stream
    
    img, faces = detect_face_eyes(frame)

    # Display the resulting frame
    text = 'Faces: ' + str(faces)
    cv2.putText(img, text, (10,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
    cv2.imshow('face', img)

    # This command let's us quit with the "q" button on a keyboard.
    # Simply pressing X on the window won't work!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()





