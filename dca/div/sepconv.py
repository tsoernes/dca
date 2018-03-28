import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from nets.utils import InPlaneLocallyConnected2D


def rescale(arr):
    # Quick method to fit output into plotting range
    arr = arr - np.min(arr, (0, 1))
    return arr / np.max(arr, (0, 1))


def main():
    channels = 3
    feed = np.asarray(np.round(np.random.random((1, 10, 10, channels)), 1), np.float32)
    # Feed of shape [batch,height,width,channels]

    individual_channels = tf.split(feed, channels, -1)
    print(individual_channels)

    # With variable scope and tf.layers.conv2d
    # results = []
    # with tf.variable_scope('channel_conv', reuse=tf.AUTO_REUSE):
    #     for channel in individual_channels:
    #         #Change conv parameters, add bias, add activation, etc. as desired
    #         #NAME IS REQUIRED, OTHERWISE A DEFAULT WILL BE ASSIGNED WHICH DIFFERS BY CHANNEL
    #         #THIS NAME *MUST NOT VARY* THROUGHOUT THE LOOP OR NEW FILTERS WILL BE CREATED
    #         conv = tf.layers.conv2d(
    #             channel,
    #             filters=1,
    #             kernel_size=[3, 3],
    #             padding='VALID',
    #             use_bias=False,
    #             name='conv')
    #         results.append(conv)
    # output = tf.concat(results, -1)
    # [<tf.Variable 'channel_conv/conv/kernel:0' shape=(3, 3, 1, 1) dtype=float32_ref>]
    # kernel = tf.get_default_graph().get_tensor_by_name('channel_conv/conv/kernel:0')
    with tf.variable_scope('conv'):
        output = tf.contrib.layers.conv2d_in_plane(
            tf.constant(feed), [3, 3],
            stride=1,
            padding='VALID',
            activation_fn=None,
            biases_initializer=None)

    # With tf.nn.conv2d
    # kernel=tf.get_variable('var', (3,3,1,1),tf.float32)
    # Grabbing previous kernel to show equivalent results,
    # commented out line is how you would create kernel otherwise
    # print(tf.global_variables())
    # [<tf.Variable 'conv/ConvInPlane/weights:0' shape=(3, 3, 1, 1) dtype=float32_ref>]
    kernel = tf.get_default_graph().get_tensor_by_name('conv/ConvInPlane/weights:0')

    results = []
    for channel in individual_channels:
        # Change conv parameters, add bias, add activation, etc. as desired
        conv = tf.nn.conv2d(channel, kernel, [1, 1, 1, 1], 'VALID')
        results.append(conv)
    output2 = tf.concat(results, -1)

    lconv = InPlaneLocallyConnected2D(kernel_size=[3, 3], use_bias=False)
    output3 = lconv(tf.constant(feed))

    feed2 = np.repeat(np.random.random((1, 10, 10, 1)), channels, 0).astype(np.float32)
    f = feed2.T
    print(f.shape)
    lconv2 = InPlaneLocallyConnected2D(kernel_size=[3, 3], use_bias=False)
    output4 = lconv2(tf.constant(f))
    # Demonstrate the same output from both methods
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        out, out2, out3, out4 = sess.run((output[0], output2[0], output3[0], output4[0]))
        assert (out.shape == out2.shape == out3.shape), (out.shape, out2.shape,
                                                         out3.shape)
        assert (out4[:, :, 0] == out4[:, :, 1]).all()
        assert (out4[:, :, 1] == out4[:, :, 2]).all()
        # assert (out == out2).all(), (out[0], out2[0])
        fig, axs = plt.subplots(3, 1, True, True)
        axs[0].imshow(rescale(out))
        axs[1].imshow(rescale(out2))
        axs[2].imshow(rescale(out3))
        plt.show(fig)


main()
