import cv2

def extract_face(img):
    # Reading cascade
    face_cascade = cv2.CascadeClassifier('ann/haarcascade_frontalface_default.xml')

    # Converting RGB to Gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detecting faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # If there is no detected face, return None
    if len(faces) == 0:
        
        return None

    biggest_face = faces[0]

    for f in faces:
        if (f[2] * f[3]) > (biggest_face[2] * biggest_face[3]):
            biggest_face = f

    # Return biggest face area
    roi = img[biggest_face[1]:(biggest_face[1] + biggest_face[3]), biggest_face[0]:(biggest_face[0] + biggest_face[2])]
    
    return roi
