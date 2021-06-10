import cv2
print('set')
cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 1080)

print('streaming')
while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('test', frame)
        k = cv2.waitKey(1)
        if k == 27:
            cv2.destroyAllWindows()
            cap.release()
            break

    else:
        cv2.destroyAllWindows()
        cap.release()
        break


