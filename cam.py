import cv2

car_cascade = cv2.CascadeClassifier('cars.xml')
video = cv2.VideoCapture('video.mp4')

while True:
    check, frame = video.read()
    cars = car_cascade.detectMultiScale(frame, 1.13, 3)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('Car Detection', frame)

    # This line is moved down to ensure 'k' is defined
    k = cv2.waitKey(10)

    if k == 27:
        break

video.release()
cv2.destroyAllWindows()
