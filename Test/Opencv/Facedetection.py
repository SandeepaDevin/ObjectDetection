import cv2
from matplotlib import pyplot as plt

img = cv2.imread("sample_data/sample_image.jpeg")
real = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(real)
plt.show()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

faces = face_cascade.detectMultiScale(img_gray)
count = len(faces)
  
if count != 0:
    for (x, y, width, height) in faces:
        cv2.rectangle(real, (x, y), 
                      (x + height, y + width), 
                      (0, 255, 0), 5)
plt.imshow(real)
plt.show()
