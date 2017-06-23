import tensorflow as tf

class Network(object):
    def __init__(self):
        self.xs = tf.placeholder(tf.float32, [None, 227, 227, 3])
        self.ys = tf.placeholder(tf.int32, [None, 10])

        conv1 = tf.layers.conv2d(self.xs, 48, 11, strides=4,activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(conv1, 3, 2)
        
        conv2 = tf.layers.conv2d(pool1, 128, 5, padding='same', activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(conv2, 3, 2)
        
        conv3 = tf.layers.conv2d(pool2, 192, 3, padding='same', activation=tf.nn.relu)
        
        conv4 = tf.layers.conv2d(conv3, 192, 3, padding='same', activation=tf.nn.relu)
        
        conv5 = tf.layers.conv2d(conv4, 128, 3, padding='same', activation=tf.nn.relu)
        pool5 = tf.layers.max_pooling2d(conv5, 3, 2)
        
        pool5_fat = tf.reshape(pool5, [-1, 4608])

        fc1 = tf.layers.dense(pool5_fat, 1024, activation=tf.nn.relu)
        output = tf.layers.dense(fc1, 10)

        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.ys, logits=output)
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
        
        self.accuracy = tf.metrics.accuracy(tf.argmax(self.ys, axis=1), tf.argmax(output, axis=1))[1]
        self.prob = tf.nn.softmax(output)
        self.recognition = tf.argmax(self.prob, 1)
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver()

    def __del__(self):
        self.sess.close()

    def train(self, image, label):
        feed = {
            self.xs: image,
            self.ys: label
        }
        self.sess.run(self.train_step, feed)
    
    def GetLoss(self, image, label):
        feed = {
            self.xs: image,
            self.ys: label
        }
        return self.sess.run(self.loss, feed)

    def GetAccuracy(self, image, label):
        feed = {
            self.xs: image,
            self.ys: label
        }
        return self.sess.run(self.accuracy, feed)
    
    def save(self):
        save_path = self.saver.save(self.sess, 'my_net/model.ckpt')
        print("Save to path: ", save_path)
    
    def restore(self):
        self.save.restore(sefl.sess, 'my_net/model.ckpt')

    def recognition(self, image):
        result_list = [
            'Nestle black tea ',
            'Chocolate milk   ',
            'Apple            ',
            'Yakult           ',
            'Pure eat tea...XD',
            'AB Yogurt        ',
            'Rose milk tea    ',
            'Pudding          ',
            'Mr. Brown Cafe   ',
            'Mai Xiang tea    '
        ]
        feed = {
            self.xs: image
        }
        prob = self.sess.run(self.prob, feed)[0]
        for i in range(10):
            print(result_list[i], ":", prob[i])
        no = self.sess.run(self.recognition, feed)[0]
        print("Result:", result_list[no])

#net=Network()