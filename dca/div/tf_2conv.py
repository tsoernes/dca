import tensorflow as tf

sess = tf.Session()
inp1 = tf.ones((1, 2, 2, 5))
inp2 = tf.zeros((1, 2, 2, 5))

conv_layer = tf.layers.Conv2D(
    filters=2,
    kernel_size=2,
    padding='SAME',
    kernel_initializer=tf.glorot_uniform_initializer(),
    name="vconv")
dense_layer = tf.layers.Dense(units=1, use_bias=False, activation=None)

tdout1 = dense_layer.apply(inp1)
tdout2 = dense_layer.apply(inp2)
sess.run(tf.global_variables_initializer())
# This works:
dout1, dout2 = sess.run((tdout1, tdout2))

tcout1 = conv_layer.apply(inp1)
tcout2 = conv_layer.apply(inp2)
sess.run(tf.global_variables_initializer())
# This does not work:
print(sess.run((tcout1, tcout2)))
# ValueError: cannot add op with name vconv/convolution as that name is already used
