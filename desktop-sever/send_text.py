import base64
import cv2
import random
import zmq
import time

context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
footage_socket.bind('tcp://192.168.0.9:32605')
footage_socket.connect('tcp://192.168.0.14:32605')

while True:
    try:
        colors = ['red', 'blue', 'green', 'orange', 'black']
        ind = random.randint(0,4)
        buffer = 'Stop! '+colors[ind]+' shirt'
        text_to_read = base64.b64encode(buffer.encode('utf-8'))
        footage_socket.send(text_to_read)
        time.sleep(2)

    except KeyboardInterrupt:
        break
