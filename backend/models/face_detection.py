import cv2
import os

FOLDER_PREPROCESSED = "uploads/preprocessed/"
os.makedirs(FOLDER_PREPROCESSED, exist_ok=True)  

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        return None  

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    img = cv2.equalizeHist(img)  
    img = cv2.resize(img, (128, 128)) 

    file_name = os.path.basename(image_path)
    new_path = os.path.join(FOLDER_PREPROCESSED, file_name)
    cv2.imwrite(new_path, img)  

    return new_path

def detect_face(image_path):
    preprocessed_path = preprocess_image(image_path)
    
    if preprocessed_path is None:
        return False  

    img = cv2.imread(preprocessed_path)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))

    return len(faces) > 0