import cv2
import zmq
import base64
import time

fps = 30

context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
footage_socket.bind('tcp://192.168.0.11:13462')
footage_socket.connect('tcp://192.168.0.9:13462')

camera = cv2.VideoCapture(0)  # init the camera

while True:
    try:
        grabbed, frame = camera.read()  # grab the current frame
        frame = cv2.resize(frame, (640, 480))  # resize the frame
        jpg_as_text = base64.b64encode(cv2.imencode('.jpg', frame)[1])
        footage_socket.send(jpg_as_text)
        #time.sleep(fps/60)

    except KeyboardInterrupt:
        camera.release()
        cv2.destroyAllWindows()
        break

