import socket
import recognition

host = "127.0.0.1"
port = 8181

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((host, port))

while True:
    url = client.recv(4096).decode('utf-8')
    output = recognition.rec(url)
    client.send(output.encode('utf-8'))
