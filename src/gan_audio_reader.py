import argparse

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from src.wavenet.audio_reader import AudioReader
from src.wavenet.model import WaveNetModel

def xavier_init(size):
  in_dim = size[0]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
  return tf.random_normal(shape=size, stddev=xavier_stddev)

def sample_Z(m, n):
  return np.random.uniform(-1., 1., size=[m, n])


def generator(z, G_W1, G_b1, G_W2, G_b2):
  G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
  G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
  G_prob = tf.nn.sigmoid(G_log_prob)

  return G_prob


def discriminator(x, D_W1, D_b1, D_W2, D_b2):
  D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
  D_logit = tf.matmul(D_h1, D_W2) + D_b2
  D_prob = tf.nn.sigmoid(D_logit)

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


def create_audio_reader(args, coord):
  # TODO Calculate receptive_field:
  # receptive_field = WaveNetModel.calculate_receptive_field(...)
  receptive_field = 1

  return AudioReader(
    args.audio_dir,
    coord,
    args.sample_rate,
    args.gc_enabled,
    receptive_field,
    # sample_size=None,
    # silence_threshold=None,
    # queue_size=32
  )


def main(args):
  X = tf.placeholder(tf.float32, shape=[None, 784])

  D_W1 = tf.Variable(xavier_init([784, 128]))
  D_b1 = tf.Variable(tf.zeros(shape=[128]))

  D_W2 = tf.Variable(xavier_init([128, 1]))
  D_b2 = tf.Variable(tf.zeros(shape=[1]))

  theta_D = [D_W1, D_W2, D_b1, D_b2]

  Z = tf.placeholder(tf.float32, shape=[None, 100])

  G_W1 = tf.Variable(xavier_init([100, 128]))
  G_b1 = tf.Variable(tf.zeros(shape=[128]))

  G_W2 = tf.Variable(xavier_init([128, 784]))
  G_b2 = tf.Variable(tf.zeros(shape=[784]))

  theta_G = [G_W1, G_W2, G_b1, G_b2]

  G_sample = generator(Z, G_W1, G_b1, G_W2, G_b2)
  D_real, D_logit_real = discriminator(X, D_W1, D_b1, D_W2, D_b2)
  D_fake, D_logit_fake = discriminator(G_sample, D_W1, D_b1, D_W2, D_b2)

  # D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
  # G_loss = -tf.reduce_mean(tf.log(D_fake))

  # Alternative losses:
  # -------------------
  D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
  D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
  D_loss = D_loss_real + D_loss_fake
  G_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

  mb_size = 128
  Z_dim = 100

  # mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  if not os.path.exists('out/'):
    os.makedirs('out/')

  coord = tf.train.Coordinator()
  audio_reader = create_audio_reader(args, coord)

  i = 0
  for it in range(args.iters):
    # if it % 1000 == 0:
    #   samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
    #
    #   fig = plot(samples)
    #   plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
    #   i += 1
    #   plt.close(fig)

    # X_mb, _ = mnist.train.next_batch(mb_size)
    X_mb = audio_reader.dequeue(mb_size)
    
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

    if it % 1000 == 0:
      print('Iter: {}'.format(it))
      print('D loss: {:.4}'.format(D_loss_curr))
      print('G_loss: {:.4}'.format(G_loss_curr))
      print()

  coord.request_stop()


def parse_args():
  parser = argparse.ArgumentParser(description='Quick hack together of audio_reader + gan_tensorflow')
  parser.add_argument('audio_dir',
    help='Directory containing input WAV files')
  parser.add_argument('--iters', type=int, default=1000000)
  parser.add_argument('--sample_rate', type=int, default=16000)
  parser.add_argument('--gc_enabled', action='store_true')

  return parser.parse_args()


if __name__ == '__main__':
  args = parse_args()
  main(args)
