import tensorflow as tf

inp = tf.ones((1, 2, 2, 5))
shape = (3, 3, 5, 1)
filters = tf.Variable(tf.zeros_initializer()(shape))
conv = tf.nn.depthwise_conv2d(inp, filters, strides=[1, 1, 1, 1], padding="SAME")
value_layer = tf.layers.Dense(
    units=1, kernel_initializer=tf.zeros_initializer(), use_bias=False, activation=None)
val = value_layer.apply(conv)

trainer = tf.train.GradientDescentOptimizer(1e-4)
v_grads, _ = zip(*trainer.compute_gradients(val))
v_gradsf = tf.concat([tf.reshape(e, [-1, 1]) for e in v_grads], axis=0)
vweights = tf.random_uniform((50, 1))
dot = tf.matmul(tf.transpose(v_gradsf), vweights)
v_hess, _ = zip(*trainer.compute_gradients(dot))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run((v_grads, dot, v_hess)))
