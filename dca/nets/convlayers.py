import numpy as np
import tensorflow as tf
from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras import (activations, constraints,
                                                 initializers, regularizers)
from tensorflow.python.keras._impl.keras.engine import InputSpec, Layer
from tensorflow.python.keras._impl.keras.utils import conv_utils


class SplitConv:
    def __init__(self,
                 inp,
                 kernel_size=3,
                 stride=1,
                 use_bias=True,
                 padding="SAME",
                 kernel_initializer=tf.constant_initializer(0.1)):
        self.padding = padding.upper()
        if inp.shape[-1] == 70 * 70 * 70 + 1:
            self.split_axis = [70, 70, 70, 1]
        elif inp.shape[-1] == 70 + 1:
            self.split_axis = [70, 1]
        else:
            raise NotImplementedError
        self.biases_initializer = tf.zeros_initializer() if use_bias else None
        self.kernel_initializer = kernel_initializer
        self.fps = tf.split(inp, self.split_axis, -1)


class InPlaneSplit(SplitConv):
    """Defaults to ReLU"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        convs = []
        for feature_part in self.fps:
            conv = tf.contrib.layers.conv2d_in_plane(
                inputs=feature_part,
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                biases_initializer=self.biases_initializer,
                weights_initializer=self.kernel_initializer)
            convs.append(conv)
        out = tf.concat(convs, -1)
        return out


class InPlaneLocallyConnected2D(Layer):
    """In-plane Locally-connected layer for 2D inputs.
  The `InPlaneLocallyConnected2D` layer works similarly
  to the `Conv2D_in_plane` layer, except that weights are unshared,
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
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(InPlaneLocallyConnected2D, self).__init__(**kwargs)
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
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=4)

    # @shape_type_conversion
    def build(self, input_shape):
        input_row, input_col = input_shape[1:-1]
        input_filter = input_shape[3]
        # print("Build", input_shape)
        self.filters = input_filter
        if input_row is None or input_col is None:
            raise ValueError('The spatial dimensions of the inputs to '
                             ' a LocallyConnected2D layer '
                             'should be fully-defined, but layer received '
                             'the inputs shape ' + str(input_shape))
        output_row = conv_utils.conv_output_length(input_row, self.kernel_size[0],
                                                   self.padding, self.strides[0])
        output_col = conv_utils.conv_output_length(input_col, self.kernel_size[1],
                                                   self.padding, self.strides[1])
        self.output_row = output_row
        self.output_col = output_col
        self.kernel_shape = (output_row * output_col,
                             self.kernel_size[0] * self.kernel_size[1], 1)
        # self.kernel_size[0] * self.kernel_size[1] * int(input_filter), 1)
        # print("Kernel shape", self.kernel_shape)
        self.kernel = self.add_weight(
            shape=self.kernel_shape,
            initializer=self.kernel_initializer,
            name='kernel',
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(output_row, output_col, input_filter),
                initializer=self.bias_initializer,
                name='bias',
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(ndim=4, axes={-1: input_filter})
        self.built = True

    # @shape_type_conversion
    def compute_output_shape(self, input_shape):
        print("compute_output_shape CALLED")
        rows = input_shape[1]
        cols = input_shape[2]

        rows = conv_utils.conv_output_length(rows, self.kernel_size[0], self.padding,
                                             self.strides[0])
        cols = conv_utils.conv_output_length(cols, self.kernel_size[1], self.padding,
                                             self.strides[1])

        return (input_shape[0], rows, cols, input_shape[3])

    def call(self, inputs):
        individual_channels = tf.split(inputs, inputs.shape[3], -1)
        convs = []
        for channel in individual_channels:
            conv = K.local_conv2d(channel, self.kernel, self.kernel_size, self.strides,
                                  (self.output_row, self.output_col), self.data_format)
            convs.append(conv)
        outputs = tf.concat(convs, -1)
        if self.use_bias:
            outputs = K.bias_add(outputs, self.bias, data_format=self.data_format)
        outputs = self.activation(outputs)
        return outputs

    def get_config(self):
        print("get_config CALLED")
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(InPlaneLocallyConnected2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
