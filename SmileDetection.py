import cv2



#face fector
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')

#Choose face to detect smile
webcam = cv2.VideoCapture(0)

while True:
 #read the current frame
 successful_frame_read,frame = webcam.read()

 #if there is an error abort
 if not successful_frame_read:
     break

 #converting to greyscale
 grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

 #detect faces first then smiles
 faces = face_detector.detectMultiScale(grayscaled_img)
 

 #Draw a rectangle around faces
 for(x,y,w,h) in faces:
     cv2.rectangle(frame, (x, y), (x+w, h+y), ( 0, 255, 0), 2 )
     #sub image
     #slice the image(using numpy N dementiona array slices) 
     the_faces = frame[y:y+h , x:x+w ]
     #the_faces = (x,y,w,h)
     #convert sub image to grayscale
     grayscaled_face = cv2.cvtColor(the_faces, cv2.COLOR_BGR2GRAY)
     
     #detect smile
     smiles = smile_detector.detectMultiScale(grayscaled_face, scaleFactor=1.7,minNeighbors=20)
     eyes   = eye_detector.detectMultiScale(grayscaled_face, scaleFactor=1.1,minNeighbors=10)
     #find all smiles in the face
     for(x_ , y_ , w_ , h_) in smiles:
              cv2.rectangle(the_faces, (x_ , y_), (x_+w_ , h_+y_), ( 50, 50, 200), 4 )

     #label smiling face
     if len(smiles) > 0 :
         cv2.putText(frame, 'Smiling', (x ,y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))
     for(x__ , y__ , w__ , h__) in eyes:
              cv2.rectangle(the_faces, (x__ , y__), (x__+w__ , h__+y__), ( 100, 50, 100), 2 )
 #show imge in seperate window
 cv2.imshow("window_name",frame)

 #waiting until key press
 key = cv2.waitKey(1)

#Quit
 if key==81 or key==113 :
    break
#cleanup
webcam.release() 
cv2.destroyAllwindows()

print("code complted")