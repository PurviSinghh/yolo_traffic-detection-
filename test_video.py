import cv2

print("TEST STARTED")

video_path = "yolo_traffic mp4 .mp4"

cap = cv2.VideoCapture(video_path)

print("Video opened:", cap.isOpened())

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    cv2.imshow("Video Test", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("TEST COMPLETED")