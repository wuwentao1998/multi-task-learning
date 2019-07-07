import math
import tensorflow as tf
import  numpy as np
from sklearn.metrics import log_loss, accuracy_score
import logging
import time

# configure the logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class multiTaskModel(object):
    def __init__(
            self,
            sess,
            max_steps,
            input_dimenion,
            learning_rate,
            num_classes,
            max_epochs,
            display_step,
            batch_size,
            dropout_ratio,
            dense_units,
            lstm_units,
            lstm_num_layers,
            pos_weight
            ):

        self.sess = sess
        self.max_steps = max_steps
        self.input_dimenion = input_dimenion # 数据的维度
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.max_epochs = max_epochs
        self.display_step = display_step
        self.batch_size = batch_size
        self.dropout_ratio = dropout_ratio
        self.dense_units = dense_units
        self.lstm_units = lstm_units
        self.lstm_num_layers = lstm_num_layers
        self.pos_weight = pos_weight

        self.training_flag = True
        self.inputs = None
        self.labels = None
        self.seq_lens = None

        self.build_graph()


    def build_graph(self):
        self.inputs = tf.placeholder(tf.float32, [None, self.max_steps, self.input_dimenion], name="inputs")
        self.labels = tf.placeholder(tf.float32, [None, self.num_classes], name="labels")
        self.seq_lens = tf.placeholder(tf.int32, [None], name="seq_lens")
        self.training_flag = tf.placeholder(tf.bool)

        inputs = tf.unstack(self.inputs, self.max_steps, 1)

        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.lstm_units)

        if self.dropout_ratio > 0:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=1.0 - self.dropout_ratio)

        # get lstm cell output
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, inputs, dtype=tf.float32, sequence_length=self.seq_lens)
        outputs = tf.stack(outputs) # [n_steps,batch,n_hidden]
        outputs = tf.transpose(outputs, [1, 0, 2]) # [batch,n_steps,n_hidden]

        index = tf.range(0, tf.shape(outputs)[0]) * self.max_steps + (self.seq_lens - 1)

        # get last step hidden state
        last = tf.gather(tf.reshape(outputs, [-1, self.lstm_units]), index)

        self.pred = tf.layers.dense(last, self.num_classes)

        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
            logits=self.pred,
            labels=self.labels
            #pos_weight=tf.constant([1.0, self.pos_weight])
            )
        )

        self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


    def train(self, data):
        best_val_acc = 0.0
        best_val_loss = 1000 * 1000

        init = tf.global_variables_initializer()
        self.sess.run(init)

        val_x, val_y, val_seq_lens = data.validation()
        test_x, test_y, test_seq_lens = data.testing()

        epoch = 0
        time_init = time.time()
        time_start = time.time()
        train_size = int(data.num_samples * data.train_ratio)

        while epoch < self.max_epochs:
            num_batches = int(math.ceil(train_size / self.batch_size))
            step = 0
            train_loss_average = 0.0
            time_step_start = time.time()

            while step < num_batches:
                batch_x, batch_y, batch_seq_lens = data.next_batch()
                train_feed_dict = {
                    self.inputs : batch_x,
                    self.labels : batch_y,
                    self.seq_lens : batch_seq_lens,
                    self.training_flag : True
                }

                train_loss, train_optim, train_pred, train_acc = self.sess.run(
                    [self.loss, self.optim, self.pred, self.accuracy], train_feed_dict
                )

                step += 1
                train_loss_average += train_loss

                if step % self.display_step == 0:
                    val_loss, val_pred, val_acc = self._calculate_loss(val_x, val_y, val_seq_lens)

                    logger.info("validation loss: {:.8f}, best validation loss: {:.8f}".format(val_loss, best_val_loss))
                    logger.info(
                        "validation accuracy: {:.4f}, best validation accuracy: {:.4f}".format(val_acc, best_val_acc))
                    logger.info("It takes {:.4} seconds to run this step\n".format(time.time() - time_step_start))

                    if best_val_acc < val_acc:
                        best_val_acc = val_acc
                        best_val_loss = val_loss

            val_loss, val_pred, val_acc = self._calculate_loss(val_x, val_y, val_seq_lens)
            logger.info("epoch: {}, minibatch training loss: {:.8f}".format(epoch, train_loss_average / step))
            logger.info("validation loss: {:.8f}, best validation loss: {:.8f}".format(val_loss, best_val_loss))
            logger.info("validation accuracy: {:.4f}, best validation accuracy: {:.4f}".format(val_acc, best_val_acc))
            logger.info("It takes {:.4f} seconds to run this step".format(time.time() - time_start))
            time_start = time.time()

            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss

            epoch += 1

        logger.info("Optimization finished, best validation loss: {:.8f}, best validation accuracy: {:.4f}".format(
            best_val_loss, best_val_acc))

        test_loss, test_pred, test_acc = self._calculate_loss(test_x, test_y, test_seq_lens)
        logger.info("test loss: {:8f}, test accuracy: {:.4f}".format(test_loss, test_acc))
        logger.info("It takes {:.4f} seconds to run this step".format(time.time() - time_init))

        return np.argmax(test_pred, 1), np.argmax(test_y, 1), test_loss, test_acc


    def _calculate_loss(self, data_x, data_y, seq_lens):
        num_iters = int(math.ceil(len(data_x) / self.batch_size))
        predict_y = np.zeros([len(data_y), self.num_classes])

        step = 0
        while step < num_iters:
            feed_dict = {
                self.inputs: data_x[step * self.batch_size:min((step + 1) * self.batch_size, len(data_y))],
                self.seq_lens: seq_lens[step * self.batch_size:min((step + 1) * self.batch_size, len(data_y))],
                self.training_flag: False
            }

            predict_y_i = self.sess.run([self.pred], feed_dict)

            count = 0
            while count < self.batch_size and step * self.batch_size + count < len(data_y):
                predict_y[step * self.batch_size + count] = self._calculate_softmax(predict_y_i[0][count])
                count += 1

            step += 1

        loss = log_loss(data_y, predict_y)

        auc_score = accuracy_score(np.argmax(data_y, 1), np.argmax(predict_y, 1))

        return loss, predict_y, auc_score

    @staticmethod
    def _calculate_softmax(probs):
        e_x = np.exp(probs - np.max(probs))
        return e_x / e_x.sum()



