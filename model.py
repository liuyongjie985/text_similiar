import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class Model():
    def __init__(self, args, training=True):
        self.args = args
        if not training:
            args.batch_size = 1
            args.seq_length = 1

        # choose different rnn cell 
        if args.model == 'rnn':
            cell_fn = rnn.RNNCell
        elif args.model == 'gru':
            cell_fn = rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn.LSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        # warp multi layered rnn cell into one cell with dropout
        cells1 = []
        for _ in range(args.num_layers):
            cell1 = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell1 = rnn.DropoutWrapper(cell1,
                                           input_keep_prob=args.input_keep_prob,
                                           output_keep_prob=args.output_keep_prob)
            cells1.append(cell1)
        self.cell1 = cell1 = rnn.MultiRNNCell(cells1, state_is_tuple=True)

        cells2 = []
        for _ in range(args.num_layers):
            cell2 = cell_fn(args.rnn_size)
            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):
                cell2 = rnn.DropoutWrapper(cell2,
                                           input_keep_prob=args.input_keep_prob,
                                           output_keep_prob=args.output_keep_prob)
            cells2.append(cell2)
        self.cell2 = cell2 = rnn.MultiRNNCell(cells2, state_is_tuple=True)

        # input/target data (int32 since input is char-level)
        self.input_data_1 = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])
        self.input_data_2 = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])

        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, 1])

        self.initial_state1 = cell1.zero_state(args.batch_size, tf.float32)
        self.initial_state2 = cell2.zero_state(args.batch_size, tf.float32)

        # softmax output layer, use softmax to classify
        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w",
                                        [args.seq_length * args.rnn_size * 2, 2])
            softmax_b = tf.get_variable("softmax_b", [2])

        # transform input to embedding
        zero_embedding = tf.zeros([1, args.rnn_size])
        # zero_embedding = tf.get_variable("zero_embedding", [args.rnn_size])
        embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])

        embedding = tf.concat([zero_embedding, embedding], 0)

        print(embedding)

        inputs1 = tf.nn.embedding_lookup(embedding, self.input_data_1)
        inputs2 = tf.nn.embedding_lookup(embedding, self.input_data_2)

        # dropout beta testing: double check which one should affect next line
        if training and args.output_keep_prob:
            inputs1 = tf.nn.dropout(inputs1, args.output_keep_prob)
            inputs2 = tf.nn.dropout(inputs2, args.output_keep_prob)
        # unstack the input to fits in rnn model
        inputs1 = tf.split(inputs1, args.seq_length, 1)
        inputs2 = tf.split(inputs2, args.seq_length, 1)

        inputs1 = [tf.squeeze(input_, [1]) for input_ in inputs1]
        inputs2 = [tf.squeeze(input_, [1]) for input_ in inputs2]

        # loop function for rnn_decoder, which take the previous i-th cell's output and generate the (i+1)-th cell's input
        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        # rnn_decoder to generate the ouputs and final state. When we are not training the model, we use the loop function.
        outputs1, last_state1 = legacy_seq2seq.rnn_decoder(inputs1, self.initial_state1, cell1,
                                                           loop_function=loop if not training else None, scope='rnnlm')

        outputs2, last_state2 = legacy_seq2seq.rnn_decoder(inputs2, self.initial_state2, cell1,

                                                           loop_function=loop if not training else None, scope='rnnlm')

        a = tf.concat(outputs1, 1)
        b = tf.concat(outputs2, 1)

        output = tf.concat([a, b], 1)

        # batch_size * seq_length
        output = tf.reshape(output, [-1, args.seq_length * args.rnn_size * 2])

        # output layer
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        # self.probs = tf.nn.softmax(self.logits)

        # loss is calculate by the log loss and taking the average.

        a = [self.logits]
        b = [tf.reshape(self.targets, [-1])]
        c = [tf.ones([args.batch_size])]
        loss = legacy_seq2seq.sequence_loss_by_example(a, b, c)
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = tf.concat([last_state1, last_state2], 2)
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        print("所有可更新参数为", tvars)
        # calculate gradients
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                                          args.grad_clip)
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)

        # apply gradient change to the all the trainable variable.
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

    def sample(self, sess, chars, vocab, num=200, prime='', sampling_type=1):
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return (int(np.searchsorted(t, np.random.rand(1) * s)))

        ret = prime
        char = prime[-1]
        for _ in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = chars[sample]
            ret += pred
            char = pred
        return ret
