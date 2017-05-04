import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


def xavier_init(size):
  in_dim = size[0]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
  return tf.random_normal(shape=size, stddev=xavier_stddev)


X = tf.placeholder(tf.float32, shape=[None, 784], name='X-data')

with tf.name_scope('D_params') as theta_D:
  D_W1 = tf.Variable(xavier_init([784, 128]), name='D_W1')
  D_b1 = tf.Variable(tf.zeros(shape=[128]), name='D_b1')

  D_W2 = tf.Variable(xavier_init([128, 1]), name='D_W2')
  D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')

  theta_D = [D_W1, D_W2, D_b1, D_b2]

Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z-noise')

with tf.name_scope('G_params') as theta_G:
  G_W1 = tf.Variable(xavier_init([100, 128]), name='G_W1')
  G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

  G_W2 = tf.Variable(xavier_init([128, 784]), name='G_W2')
  G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')

  theta_G = [G_W1, G_W2, G_b1, G_b2]


def sample_Z(m, n):
  return np.random.uniform(-1., 1., size=[m, n])


def generator(z):
  with tf.name_scope('generator') as generator:
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1, name='G_h1')
    G_log_prob = tf.matmul(G_h1, G_W2, name='G_matmul') + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob, name='G_prob')

  return G_prob


def discriminator(x, name):
  with tf.name_scope(name) as discriminator:
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1, name='D_relu')
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit, name='D_prob')

  return D_prob, D_logit


def plot(samples):
  fig = plt.figure(figsize=(4, 4))
  gs = gridspec.GridSpec(4, 4)
  gs.update(wspace=0.05, hspace=0.05)

  for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

  return fig


G_sample = generator(Z)
D_real, D_logit_real = discriminator(X, 'discriminator_real')
D_fake, D_logit_fake = discriminator(G_sample, 'discriminator_fake')

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
with tf.name_scope('D_loss_fn') as d_loss:
  D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)), name='D_loss_real')
  D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)), name='D_loss_fake')
  D_loss = D_loss_real + D_loss_fake

with tf.name_scope('G_loss_fn') as g_loss:
  G_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)), name='G_loss')

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D, name='D_solver')
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G, name='G_solver')

mb_size = 128
Z_dim = 100

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
  os.makedirs('out/')

tf.summary.scalar('hi', 1)

# merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('logdir', sess.graph)

i = 0
for it in range(10):
  if it % 1000 == 0:
    samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

    fig = plot(samples)
    plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
    i += 1
    plt.close(fig)

  X_mb, _ = mnist.train.next_batch(mb_size)

  _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
  _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

  if it % 1000 == 0:
    print('Iter: {}'.format(it))
    print('D loss: {:.4}'.format(D_loss_curr))
    print('G_loss: {:.4}'.format(G_loss_curr))
    print()

# sess.run([merged])
