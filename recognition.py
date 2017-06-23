import sys
import alexnet
from PIL import Image
import numpy as np

def Usage():
    print("Usage :")
    print("    python {} imagefilename".format(sys.argv[0]))

def main():
    argc = len(sys.argv)
    if argc == 2:
        fname = sys.argv[1]
        im = Image.open(fname).resize([227, 227])
        image = []
        image.append(np.array(im) / 255)
        net = alexnet.Network()
        net.restore()
        net.recognition(image)
    else:
        Usage()

if __name__ == '__main__':
    main()