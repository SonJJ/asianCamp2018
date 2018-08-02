import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mat
mat.use('Agg')

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
    forget_bias = 1.0
    keep_prob = 1.0
    leaningRate = 0.001
    count = 1000   #leaning_count

    X = tf.placeholder(tf.float32, [None, seq_length, input_dim])  # [None, 7, 5] , 무한 5개출력 7줄 1box
    Y = tf.placeholder(tf.float32, [None, 1])

    hypo = None
    loss = None
    sess = None

    def cell(self):
        cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.output_dim, forget_bias=self.forget_bias,
                                            state_is_tuple=True, activation=tf.nn.softsign)
        if self.keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob = self.keep_prob)

        return cell

    def init_network(self):
        outputs, _states = tf.nn.dynamic_rnn(self.cell(), self.X, dtype=tf.float32)
        self.hypo = outputs[:, -1]


        self.loss = tf.reduce_sum(tf.square(self.hypo - self.Y))

    def learn(self):

        tf.set_random_seed(777)
        self.init_network()
        #Optimizer
        train = tf.train.AdamOptimizer(self.leaningRate).minimize(self.loss)

        self.db.load_normalized('visitorCount.csv')  # 파일명
        trainX, trainY = self.db.get_traindata(self.seq_length)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        for i in range(self.count):
            self.sess.run(train, feed_dict={self.X: trainX, self.Y: trainY})
            step_loss = self.sess.run(self.loss, feed_dict={self.X: trainX, self.Y: trainY})
            if(i%100 == 0):
                print(i, step_loss)
            elif(i == self.count):
                print(i, step_loss)

        text1 = "Rate:"+str(self.leaningRate) + ", Length:" + str(self.seq_length)\
                + ", Count:" + str(self.count)
        print("====================")
        print(text1)
        print("====================")

    def predict(self):
        # RMSE
        Y = tf.placeholder(tf.float32, [None, 1])
        P = tf.placeholder(tf.float32, [None, 1])
        rmse = tf.sqrt(tf.reduce_mean(tf.square(Y - P)))

        testX, testY = self.db.get_testdata(self.seq_length)
        predicted = self.sess.run(self.hypo, feed_dict={self.X: testX})
        err = self.sess.run(rmse, feed_dict={Y: testY, P: predicted})

        print("====================")
        print("RMSE: ", err)
        print("====================")

        plt.plot(testY)
        plt.plot(predicted)
        plt.xlabel("Time period")
        plt.ylabel("people")
        plt.title(str(self.leaningRate)+", "+str(self.count)+", RMSE:"+str(err))

        #model_save
        savePoint = "./RMSE_"+str(err)+"/test_"+str(err)
        saver = tf.train.Saver()
        saver.save(self.sess, savePoint)

        #plt_save
        fig = plt.gcf()
        fig.savefig('./RMSE_'+str(err)+'/plt.pdf')
        # plt.show()

start = MyLSTM()
start.learn()
start.predict()