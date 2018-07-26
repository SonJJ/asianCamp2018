import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class MyDB:
    xy = None

    def normalize(self, data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0)
        self.xy = numerator / (denominator + 1e-7)

    def config_data(self, seq_length):
        dataX = []
        dataY = []

        labels = self.xy[:, [-1]]
        number_of_label = len(labels)

        for i in range(0, number_of_label - seq_length):
            _x = self.xy[i:i + seq_length]
            _label = labels[i + seq_length]

            dataX.append(_x)
            dataY.append(_label)
        return dataX, dataY

    def load_normalized(self, file_name):
        xy = np.loadtxt(file_name, delimiter=',')
        xy = xy[::-1]
        xy = self.normalize(xy)
        return xy

    def get_traindata(self, seq_lenght):
        dataX, dataY = self.config_data(seq_lenght)

        train_size = int(len(dataY) * 0.7)
        trainX = np.array(dataX[0:train_size])
        trainY = np.array(dataY[0:train_size])
        return trainX, trainY

    def get_testdata(self, seq_length):
        dataX, dataY = self.config_data(seq_length)

        train_size = int(len(dataY) * 0.7)

        testX = np.array(dataX[train_size:len(dataX)])
        testY = np.array(dataY[train_size:len(dataY)])
        return testX, testY


class MyLSTM:
    db = MyDB()

    seq_length = 7
    input_dim = 5
    output_dim = 1  # 출력수

    X = tf.placeholder(tf.float32, [None, seq_length, input_dim])  # [None, 7, 5] , 무한 5개출력 7줄 1box
    Y = tf.placeholder(tf.float32, [None, 1])

    hypo = None
    loss = None
    sess = None

    def init_network(self):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.output_dim, state_is_tuple=True)
        outputs, _states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32)
        self.hypo = outputs[:, -1]
        #

        self.loss = tf.reduce_sum(tf.square(self.hypo - self.Y))

    def learn(self):
        with tf.device('/gpu:0'):
            tf.set_random_seed(777)
            self.init_network()
            train = tf.train.GradientDescentOptimizer(0.05).minimize(self.loss)

            self.db.load_normalized('visitorCount.csv')  # 파일명
            trainX, trainY = self.db.get_traindata(self.seq_length)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        for i in range(500000):
            self.sess.run(train, feed_dict={self.X: trainX, self.Y: trainY})
            step_loss = self.sess.run(self.loss, feed_dict={self.X: trainX, self.Y: trainY})

            print(i, step_loss)

    def predict(self):
        # RMSE
        Y = tf.placeholder(tf.float32, [None, 1])
        P = tf.placeholder(tf.float32, [None, 1])
        rmse = tf.sqrt(tf.reduce_mean(tf.square(Y - P)))

        testX, testY = self.db.get_testdata(self.seq_length)
        predicted = self.sess.run(self.hypo, feed_dict={self.X: testX})
        err = self.sess.run(rmse, feed_dict={Y: testY, P: predicted})
        print("RMSE: ", err)

        plt.plot(testY)
        plt.plot(predicted)
        plt.xlabel("Time period")
        plt.ylabel("people")
        plt.show()


guy = MyLSTM()
guy.learn()
guy.predict()
