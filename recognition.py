import sys
import alexnet
from PIL import Image
import numpy as np
import requests
import os

net = alexnet.Network()
net.restore()

import time
def download(url):
    filename = 'rec.jpg'
    res = requests.get(url)
    while res.status_code == 404:
        time.sleep(0.5)
        res = requests.get(url)
    with open(filename, 'wb') as f:
        res = requests.get(url)
        f.write(res.content)
    return filename

def Usage():
    print("Usage :")
    print("    python {} url".format(sys.argv[0]))

def rec(url):
    name = download(url)
    im = Image.open(name).resize([227, 227])
    image = []
    image.append(np.array(im) / 255)
    output = net.recognition(image)
    os.remove(name)
    return output

def main():
    argc = len(sys.argv)
    if argc == 2:
        url = sys.argv[1]
        fname = download(url)
        im = Image.open(fname).resize([227, 227])
        image = []
        image.append(np.array(im) / 255)
        net = alexnet.Network()
        net.restore()
        output = net.recognition(image)
        print(output)
        os.remove(fname)
    else:
        Usage()

if __name__ == '__main__':
    main()