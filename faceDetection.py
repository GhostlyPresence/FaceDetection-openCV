import cv2 as cv

def main():    

    
    cap = cv.VideoCapture(0,cv.CAP_DSHOW)
   
    if cap.isOpened():
        ret,frame = cap.read()
    else:
        ret = False
        
    while ret:

        ret,frame = cap.read()
        
        
        #path = 'C:\\Users\\Lenovo\\AppData\\Local\\Programs\\Python\\Python37-32\\lib\\site-packages\\cv2\\data\\'

        grayscale_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        face_cascade = cv.CascadeClassifier('Data\\haarcascade_frontalface_alt.xml')
        eye_cascade = cv.CascadeClassifier('Data\\haarcascade_eye.xml')
        detected_faces = face_cascade.detectMultiScale(grayscale_image)
        detected_eyes  = eye_cascade.detectMultiScale(grayscale_image)
       
        for(coloumn, row, width, height) in detected_faces:
            cv.rectangle(frame, (coloumn, row), (coloumn + width, row + height), (255, 100, 100), 2)
        for(coloumn, row, width, height) in detected_eyes:
            cv.rectangle(frame, (coloumn, row), (coloumn + width, row + height), (0, 255, 100), 2)
        cv.imshow('image', frame)
        k = cv.waitKey(20)
        if k == 27:
            break
            
    cv.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    main()
