import tensorflow as tf
import h5py
import numpy as np

import utils
import imp
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
rel = imp.reload
