import tensorflow as tf

class Network(object):
    def __init__(self):
        self.xs = tf.placeholder(tf.float32, [None, 227, 227, 3])
        self.ys = tf.placeholder(tf.float32, [None, 10])
        
        conv1 = tf.layers.conv2d(self.xs, 48, 11, activation=tf.nn.relu)
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

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess = tf.Session()
        self.sess.run(init)

    def __del__(self):
        self.sess.close()

net = Network()