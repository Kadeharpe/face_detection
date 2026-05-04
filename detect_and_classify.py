import cv2
from ultralytics import YOLO

#load trained YOLO classifier
model = YOLO("runs/classify/train4/weights/best.pt")  # use best model path

#load OpenCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

#load image
image_path = "test.jpg" #change path for different image
img = cv2.imread(image_path)

if img is None:
    print("Could not load image.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)

#detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=4, #strength of detecting face
    minSize=(30, 30) #minimum size of face detecting
)

if len(faces) == 0:
    print("No face detected. Using full image instead.")
    
    crop_path = "cropped_face.jpg"
    cv2.imwrite(crop_path, img)

else:
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    face_crop= img[y:y+h, x:x+w]

    crop_path = "cropped_face.jpg"
    cv2.imwrite(crop_path, face_crop)


#use largest detected face
faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
x, y, w, h = faces[0]

#crop face
face_crop = img[y:y+h, x:x+w]

#save cropped image
crop_path = "cropped_face.jpg"
cv2.imwrite(crop_path, face_crop)

#classify
results = model(crop_path)

predicted_class_id = results[0].probs.top1
confidence = results[0].probs.top1conf.item()
if confidence < 0.50: #if low confidence say identity is unknown
    predicted_identity = "Unknown"
else:
    predicted_identity = results[0].names[predicted_class_id]

print("Predicted identity:", predicted_identity)
print("Confidence:", round(confidence * 100, 2), "%")

#type identity on image
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

label = f"{predicted_identity} ({confidence*100:.1f}%)"
cv2.putText(img, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#resize image
scale_percent = 300
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
display_img = cv2.resize(img, (width, height))

cv2.imshow("Detection + Classification", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()