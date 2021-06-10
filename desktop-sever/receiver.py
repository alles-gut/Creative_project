import zmq
import base64
import numpy as np

import cv2

HOST = '0.0.0.0'
PORT = 13462

print('Socket created')
context = zmq.Context()
footage_socket = context.socket(zmq.SUB)
footage_socket.bind('tcp://*:'+str(PORT))
footage_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
print('Socket now listening')

while True:
    try:
        frame = footage_socket.recv_string()
        frame = base64.b64decode(frame)
        npimg = np.fromstring(frame, dtype=np.uint8)
        source = cv2.imdecode(npimg, 1)
        cv2.imshow("Stream", source)
        cv2.waitKey(1)

    except KeyboardInterrupt:
        cv2.destroyAllWindows()
        break
