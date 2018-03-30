import numpy as np  # noqa
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import (activations, initializers,
                                                 regularizers)
from tensorflow.python.keras._impl.keras.engine import Layer
from tensorflow.python.keras._impl.keras.utils import conv_utils
from tensorflow.python.ops import nn


def split_axis(input_shape):
    d = input_shape[-1]
    split_axis = [70] * (d // 70)
    if d % 70 > 0:
        split_axis.append(int(d % 70))
    return split_axis


class SplitConv:
    def __init__(self,
                 kernel_size=3,
                 stride=1,
                 use_bias=True,
                 padding="SAME",
                 kernel_initializer=tf.constant_initializer(0.1)):
        self.kernel_size, self.stride = kernel_size, stride
        self.padding = padding.upper()
        self.biases_initializer = tf.zeros_initializer if use_bias else None
        self.kernel_initializer = kernel_initializer

    def apply(self, inp, concat=True):
        if type(inp) is list:
            fps = inp
        else:
            splitaxis = split_axis(inp.shape)
            fps = tf.split(inp, splitaxis, -1)
        convs = [self.part_fn(feature_part) for feature_part in fps]
        out = tf.concat(convs, -1) if concat else convs
        return out

    def part_fn(self, feature_part):
        return feature_part


class InPlaneSplit(SplitConv):
    """Use same kernel for each feature part.
    Defaults to ReLU"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def part_fn(self, feature_part):
        return tf.contrib.layers.conv2d_in_plane(
            inputs=feature_part,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            biases_initializer=self.biases_initializer,
            weights_initializer=self.kernel_initializer)


class SeparableSplit(SplitConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if type(self.stride) is int:
            self.stride = (1, self.stride, self.stride, 1)

    def part_fn(self, feature_part):
        """        
        """
        # shape = list(map(int, (*feature_part.shape[1:], 1)))
        in_chs = int(feature_part.shape[-1])
        # depthwise_filter: [filter_height, filter_width, in_channels, channel_multiplier].
        # Contains in_channels convolutional filters of depth 1.
        depthwise_shape = [self.kernel_size, self.kernel_size, in_chs, 1]
        depthwise_filter = tf.Variable(self.kernel_initializer(depthwise_shape))
        # pointwise_filter: [1, 1, channel_multiplier * in_channels, out_channels].
        # Pointwise filter to mix channels after depthwise_filter has convolved spatially.
        pointwise_shape = [1, 1, in_chs, in_chs]
        pointwise_filter = tf.Variable(self.kernel_initializer(pointwise_shape))

        conv = tf.nn.separable_conv2d(
            feature_part,
            depthwise_filter=depthwise_filter,
            pointwise_filter=pointwise_filter,
            strides=self.stride,
            padding=self.padding,
        )
        outputs = tf.nn.relu(conv)
        if self.biases_initializer is not None:
            biases = variables.model_variable(
                'biases',
                shape=[
                    in_chs,
                ],
                dtype=feature_part.dtype,
                initializer=self.biases_initializer,
            )
            outputs = nn.bias_add(outputs, biases)
        return outputs


class InPlaneSplitLocallyConnected2D(Layer):
    """In-plane Split Locally-connected layer for 2D inputs.
  The `InPlaneSplitLocallyConnected2D` layer works similarly
  to the `InPlaneSplit` layer, except that weights are unshared spatially,
  that is, a different set of filters is applied at each
  different patch of the input.
  Input shape:
      4D tensor with shape:
      `(samples, rows, cols, channels)`
  Output shape:
      4D tensor with shape:
      `(samples, new_rows, new_cols, channels)`
      `rows` and `cols` values might have changed due to padding.
  """

    def __init__(self,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 **kwargs):
        super(InPlaneSplitLocallyConnected2D, self).__init__(**kwargs)
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, 2, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        if self.padding != 'valid':
            raise ValueError('Invalid border mode for LocallyConnected2D '
                             '(only "valid" is supported): ' + padding)
        self.data_format = conv_utils.normalize_data_format(None)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        input_row, input_col, input_depth = input_shape[1:]
        self.output_row = conv_utils.conv_output_length(input_row, self.kernel_size[0],
                                                        self.padding, self.strides[0])
        self.output_col = conv_utils.conv_output_length(input_col, self.kernel_size[1],
                                                        self.padding, self.strides[1])
        self.kernel_shape = (self.output_row * self.output_col,
                             self.kernel_size[0] * self.kernel_size[1], 1)
        # print("Kernel shape", self.kernel_shape)
        self.splitaxis = split_axis(input_shape)
        self.kernels = [
            self.add_weight(
                shape=self.kernel_shape,
                initializer=self.kernel_initializer,
                name='kernel' + str(i),
                regularizer=self.kernel_regularizer,
                constraint=None) for i in range(len(self.splitaxis))
        ]
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.output_row, self.output_col, input_depth),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=None,
                constraint=None)
        else:
            self.bias = None

    def call(self, inputs):
        frep_parts = tf.split(inputs, self.splitaxis, -1)
        convs = []
        for i, frep_part in enumerate(frep_parts):
            individual_channels = tf.split(frep_part, frep_part.shape[-1], -1)
            for ind_ch in individual_channels:
                conv = K.local_conv2d(ind_ch, self.kernels[i], self.kernel_size,
                                      self.strides, (self.output_row, self.output_col),
                                      self.data_format)
                convs.append(conv)
        outputs = tf.concat(convs, -1)
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)
        outputs = self.activation(outputs)
        return outputs
