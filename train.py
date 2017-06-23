import os
from PIL import Image
import numpy as np

class Package:
    def __init__(self):
        self.image = []
        self.label = []
    def add(self, image, label):
        self.image.append(image)
        self.label.append(label)
    
class Data:
    def __init__(self):
        self.train = Package()
        self.test = Package()
        image_path = "./image/"
        filenames = os.listdir(image_path)
        for fn in filenames:
            fn_wo_ext = os.path.splitext(fn)[0]
            spec, no = fn_wo_ext.split('-')
            spec = int(spec)
            no = int(no)
            im = Image.open(image_path + fn)
            im_fat = im.resize([227, 227])
            image = np.array(im_fat)/255
            label = [0.0]*10
            label[spec] = 1.0
            if no < 2:
                self.test.add(image, label)
            else:
                self.train.add(image, label)

from random import randint
import alexnet

net = alexnet.Network()
data = Data()
train_size = len(data.train.image)
for i in range(10000):
    n = randint(0, train_size-1)
    image = [data.train.image[n]]
    label = [data.train.label[n]]
    net.train(image, label)
    if i % 100 == 0:
        entropy = net.GetLoss(data.train.image, data.train.label)
        accuracy = net.GetAccuracy(data.train.image, data.train.label)
        print("No {} iter.  entropy: {}\taccuracy: {}".format(i, entropy, accuracy))
test_accuracy = net.GetAccuracy(data.test.image, data.test.label)
print("Test accuracy: {}".format(test_accuracy))
net.save()
