import zmq
import base64
import numpy as np
import os

HOST = '0.0.0.0'
PORT = 32605

print('Socket created')
context = zmq.Context()
pi_socket = context.socket(zmq.SUB)
pi_socket.bind('tcp://*:'+str(PORT))
pi_socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
print('Socket now listening')

while True:
    try:
        string_recv = pi_socket.recv_string()
        text_to_speak = base64.b64decode(string_recv).decode('utf-8')
        print(text_to_speak)
        os.system('echo %s | festival --tts' %text_to_speak)

    except KeyboardInterrupt:
        break
