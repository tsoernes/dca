import numpy as np
import tensorflow as tf

sess = tf.Session()
n_actions = 20
batch_sz = 10
value = tf.constant(np.random.uniform(size=batch_sz))
policy = tf.constant(np.random.uniform(size=(batch_sz, n_actions)))
# Discounted rewards
target_v = tf.constant(np.random.uniform(size=batch_sz))
advantages = tf.constant(np.random.uniform(size=batch_sz))
actions = tf.constant(np.random.randint(n_actions, size=batch_sz))

actions_oh = tf.one_hot(actions, n_actions, dtype=tf.float64)

# Source: A3C medium
responsible_outputs = tf.reduce_sum(policy * actions_oh, [1])
value_loss_1 = tf.reduce_mean(tf.square(target_v - tf.reshape(value, [-1])))
policy_loss_1 = -tf.reduce_sum(tf.log(responsible_outputs) * advantages)
entropy_1 = -tf.reduce_sum(policy * tf.log(policy))

# loss1 = 0.25 * value_loss1 + policy_loss1 - entropy * 0.01


# Source: A2C OpenAI. Code below assumes 'policy' is logits. Code above assume it's
# already softmax'ed
def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)


value_loss_2 = tf.losses.mean_squared_error(tf.squeeze(value), target_v)
value_loss_3 = tf.losses.mean_squared_error(value, target_v)
neglogpac = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=policy, labels=actions)
policy_loss_from_logits_1 = tf.reduce_mean(advantages * neglogpac)
# Or, equivalently
policy_loss_from_logits_2 = -tf.reduce_mean(advantages * tf.reduce_sum(
    actions_oh * tf.log(tf.nn.softmax(policy)), reduction_indices=[1]))
# Not equivalent to entropy_1 above
entropy_2 = tf.reduce_mean(cat_entropy(policy))
# nearly equivalent to (off by factor of batch_sz)
entropy_3 = -tf.reduce_sum(tf.nn.softmax(policy) * tf.log(tf.nn.softmax(policy)))
# loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

tf.global_variables_initializer()
value_losses = sess.run([value_loss_1, value_loss_2, value_loss_3])
print(value_losses)

policy_losses = sess.run(
    [policy_loss_1, policy_loss_from_logits_1, policy_loss_from_logits_2])
print(policy_losses)

entropies = sess.run([entropy_1, entropy_2, entropy_3])
print(entropies)
