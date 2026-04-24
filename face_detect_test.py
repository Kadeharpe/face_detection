import cv2

# Load openCV
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

#image you want to test
image_path = "test.jpg"

img = cv2.imread(image_path)

if img is None:
    print("Could not load image.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=6, #weaker detections are overlooked
    minSize=(50, 50) #ignores smaller detections
)

print("Faces detected:", len(faces))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

scale_percent = 50
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
display_img = cv2.resize(img, (width, height))

cv2.imshow("Detected Faces", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()