import socket
import recognition

host = "127.0.0.1"
port = 8282

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((host, port))

import time
while True:
    #url = client.recv(4096).decode('utf-8')
    output = recognition.rec('http://172.20.10.2:8080/?action=snapshot')
    client.send(output.encode('utf-8'))
    time.sleep(1)