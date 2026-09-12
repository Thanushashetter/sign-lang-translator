import cv2
from extract_keypoints import extract_landmarks

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    landmarks = extract_landmarks(frame)

    print("Shape:", landmarks.shape)

    cv2.imshow("Landmark Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()