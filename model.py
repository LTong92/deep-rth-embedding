import tensorflow as tf
import numpy as np

from tensorflow.contrib import rnn

def conv_layer(x, scope, kernel_shape, stride):
    """ Create CNN layer. """

    with tf.variable_scope("cnn_layer_{}".format(scope), reuse=tf.AUTO_REUSE) as scp:
        kernel = tf.get_variable(
            name=scope + '-kernel',
            initializer=tf.truncated_normal(kernel_shape, dtype=tf.float32, stddev=1e-1))

        conv = tf.nn.conv2d(x, kernel, stride, padding='SAME')
        biases = tf.get_variable(
            name=scope+'-biases',
            trainable=True,
            initializer=tf.constant(0.0, shape=[kernel_shape[-1]], dtype=tf.float32))

        bias = tf.nn.bias_add(conv, biases)
        conv = tf.nn.relu(bias, name=scope)
        lrn = tf.nn.local_response_normalization(conv,
                                                alpha=1e-4,
                                                beta=0.75,
                                                depth_radius=2,
                                                bias=2.0)
    return lrn

class DeepRthModel(object):
    """ Deep r-th root model. """

    def __init__(self, r, ts_dim, timesteps, encode_size, cnn_filter_shapes, cnn_strides, cnn_dense_layers, rnn_hidden_states, batch_size):
        """
        """
        self._r = r
        self._ts_dim = ts_dim
        self._timesteps = timesteps
        self._encode_size = encode_size
        self._cnn_filter_shapes = cnn_filter_shapes
        self._cnn_strides = cnn_strides
        self._cnn_dense_layers = cnn_dense_layers
        self._rnn_hidden_states = rnn_hidden_states
        self._batch_size = batch_size


    def construct_cnn(self, X):
        """ Construct CNN part. """

        conv = tf.cast(X, tf.float32)
        for i, (flt, stride) in enumerate(zip(self._cnn_filter_shapes, self._cnn_strides)):
            conv = conv_layer(conv, "cnn_layer_{}".format(i), flt, stride)
        _, h, w, n_channel = conv.shape
        flat = tf.reshape(conv, [-1, h*w*n_channel])

        dense = flat
        for i, lsize in enumerate(self._cnn_dense_layers):
            with tf.variable_scope("cnn_dense_{}".format(i), reuse=tf.AUTO_REUSE) as scp:
                dense = tf.layers.dense(inputs=dense, units=lsize, activation=tf.nn.relu, name="cnn_dense_{}".format(i))
        return dense

    def construct_rnn(self, X):
        """ Construct RNN part. """

        X = tf.cast(X, tf.float32)
        X = tf.unstack(X, self._timesteps, 1)
        with tf.variable_scope("rnn-part", reuse=tf.AUTO_REUSE) as scp:
            lstm_cell = rnn.BasicLSTMCell(self._rnn_hidden_states, forget_bias=1.0, name="lstm_cell")
            outputs, _ = rnn.static_rnn(lstm_cell, X, dtype=tf.float32)
        return outputs[-1]

    def binary_encode(self, X, corr):
        """ Binary encode. """

        cnn_dense = self.construct_cnn(corr)
        rnn_output = self.construct_rnn(X)
        encode = tf.concat([cnn_dense, rnn_output], axis=1)
        bias = tf.reduce_mean(encode, axis=1, keepdims=True)
        with tf.variable_scope("encode_layer", reuse=tf.AUTO_REUSE) as scp:
            shape = (encode.shape[1].value, self._encode_size)
            encode_w = tf.get_variable(
                name="encode_w",
                initializer=tf.truncated_normal(shape, dtype=tf.float32, stddev=1e-1))
            bencode = tf.matmul(encode-bias, encode_w)
        return bencode

    def construct_loss(self):
        self.x0 = tf.placeholder(tf.float32, shape=(self._batch_size, self._timesteps, self._ts_dim))
        self.corr0 = tf.placeholder(tf.float32, shape=(self._batch_size, self._ts_dim, self._ts_dim, 1))
        encode0 = self.binary_encode(self.x0, self.corr0)

        self.x1 = tf.placeholder(tf.float32, shape=(self._batch_size, self._timesteps, self._ts_dim))
        self.corr1 = tf.placeholder(tf.float32, shape=(self._batch_size, self._ts_dim, self._ts_dim, 1))
        encode1 = self.binary_encode(self.x1, self.corr1)

        self.x2 = tf.placeholder(tf.float32, shape=(self._batch_size, self._timesteps, self._ts_dim))
        self.corr2 = tf.placeholder(tf.float32, shape=(self._batch_size, self._ts_dim, self._ts_dim, 1))
        encode2 = self.binary_encode(self.x2, self.corr2)
        encode_weight = tf.get_variable(name="encode_w", shape=(self._cnn_dense_layers[-1]*2, self._encode_size))
        _lambda = 0.01
        self.loss = (tf.pow(tf.sigmoid(tf.norm(encode0-encode1) - tf.norm(encode0-encode2)), 1/self._r) +
                     _lambda*tf.nn.l2_loss(encode_weight))


    def test(self):
        corr = np.random.rand(self._batch_size, self._ts_dim, self._ts_dim, 1)
        x = np.random.rand(self._batch_size, 128, self._ts_dim)
        encode = self.binary_encode(x, corr)

        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)
            encode = sess.run(encode)
        import pdb;pdb.set_trace()

if __name__ == "__main__":
    model = DeepRthModel(ts_dim=20,
                         encode_size=32,
                         cnn_filter_shapes=[[3, 3, 1, 16], [3, 3, 16, 32], [3, 3, 32, 64], [3, 3, 64, 64]],
                         cnn_strides=[[1, 1, 1, 1], [1, 2, 2, 1], [1, 2, 2, 1], [1, 1, 1, 1]],
                         cnn_dense_layers=[256, 128],
                         rnn_hidden_states=128,
                         batch_size=128)
    model.test()
