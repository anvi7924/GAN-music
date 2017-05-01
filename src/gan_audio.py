import argparse
import json

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

from gan_audio_reader import GanAudioReader, calculate_receptive_field
from wavenet.model import WaveNetModel
from wavenet.ops import *


def xavier_init(size):
  in_dim = size[0]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
  return tf.random_normal(shape=size, stddev=xavier_stddev)


def sample_Z(m, n):
  return np.random.normal(size=(m, n))


def generator(z, G_W1, G_b1, G_W2, G_b2, G_W3, G_b3):
  G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
  G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
  G_log_prob = tf.matmul(G_h2, G_W3) + G_b3
  return tf.nn.sigmoid(G_log_prob)


def discriminator(x, D_W1, D_b1, D_W2, D_b2, D_W3, D_b3):
  D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
  D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
  return tf.matmul(D_h2, D_W3) + D_b3


def make_layer(in_dim, out_dim):
  W = tf.Variable(xavier_init([in_dim, out_dim]))
  b = tf.Variable(tf.zeros(shape[out_dim]))
  return W, b


def main(args):
  with open(args.wavenet_params, 'r') as f:
    receptive_field = calculate_receptive_field(json.load(f))
  total_sample_size = receptive_field + args.sample_size
  print('total_sample_size: {}'.format(total_sample_size))

  HIDDEN_DIM = 1024
  Z_DIM = 100

  X = tf.placeholder(tf.float32, shape=[None, total_sample_size])

  D_W1, D_b1 = make_layer(total_sample_size, HIDDEN_DIM)
  D_W2, D_b2 = make_layer(HIDDEN_DIM, HIDDEN_DIM)
  D_W3, D_b3 = make_layer(HIDDEN_DIM, 1)

  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

  Z = tf.placeholder(tf.float32, shape=[None, Z_DIM])

  G_W1, G_b1 = make_layer(Z_DIM, HIDDEN_DIM)
  G_W2, G_b2 = make_layer(HIDDEN_DIM, HIDDEN_DIM)
  G_W3, G_b3 = make_layer(HIDDEN_DIM, total_sample_size)

  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

  G_sample = generator(Z, G_W1, G_b1, G_W2, G_b2, G_W3, G_b3)
  D_real = discriminator(X, D_W1, D_b1, D_W2, D_b2, D_W3, D_b3)
  D_fake = discriminator(G_sample, D_W1, D_b1, D_W2, D_b2, D_W3, D_b3)

  D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
  G_loss = -tf.reduce_mean(D_fake)

  print('Creating D_solver')
  D_solver = tf.train.RMSPropOptimizer(
      learning_rate=args.learning_rate).minimize(-D_loss, var_list=theta_D)

  print('Creating G_solver')
  G_solver = tf.train.RMSPropOptimizer(
      learning_rate=args.learning_rate).minimize(G_loss, var_list=theta_G)

  D_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in theta_D]

  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  audio_reader = GanAudioReader(args, sess, receptive_field)

  G_decode = mu_law_decode(tf.to_int32(G_sample * args.quantization_channels),
      args.quantization_channels)

  for it in range(args.iters):
    for _ in range(5):
      X_mb = audio_reader.next_audio_batch()
      if len(X_mb) < total_sample_size:
        X_mb = np.pad(X_mb, ((0, total_sample_size - len(X_mb))), 'constant',
            constant_values=0.)
      X_mb = X_mb.reshape([1, total_sample_size])

      _, D_loss_curr, _ = sess.run([D_solver, D_loss, D_clip],
          feed_dict={X: X_mb, Z: sample_Z(args.batch_size, Z_DIM)})

    _, G_loss_curr = sess.run([G_solver, G_loss],
        feed_dict={Z: sample_Z(args.batch_size, Z_DIM)})

    if it % 10 == 0:
      print('Iter: {}\nD_loss: {:.4}\nG_loss: {:.4}\n'.format(
        it, D_loss_curr, G_loss_curr))
      if it % 200 == 0:
        samples, decoded = sess.run([G_sample, G_decode],
            feed_dict={Z: sample_Z(args.batch_size, Z_DIM)})
        with open('output_{}'.format(it), 'w') as f:
          f.write(','.join(str(sample) for sample in samples[0]))
          f.write('\n')
          f.write(','.join(str(sample) for sample in decoded[0]))
      #import pdb; pdb.set_trace()

  audio_reader.done()


LEARNING_RATE = 1e-4
WAVENET_PARAMS = '../config/wavenet_params.json'
# STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = 0.01
# EPSILON = 0.001
MOMENTUM = 0.9
# MAX_TO_KEEP = 5
# METADATA = False


def parse_args():
  parser = argparse.ArgumentParser(description='Quick hack together of audio_reader + gan_tensorflow')
  parser.add_argument('audio_dir',
    help='Directory containing input WAV files')
  parser.add_argument('--iters', type=int, default=1000000)
  parser.add_argument('--sample_rate', type=int, default=16000)
  parser.add_argument('--file_seconds', help='Number of audio seconds to use from each input file',
    type=int, default=29)
  parser.add_argument('--quantization_channels', type=int, default=256)
  parser.add_argument('--gc_enabled', action='store_true')
  parser.add_argument('--batch_size', type=int, default=1)
  parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
    help='Learning rate for training. Default: ' + str(LEARNING_RATE) + '.')
  parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
    help='JSON file with the network parameters. Default: ' + WAVENET_PARAMS + '.')
  parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
    help='Concatenate and cut audio samples to this many '
         'samples. Default: ' + str(SAMPLE_SIZE) + '.')
  parser.add_argument('--l2_regularization_strength', type=float,
    default=L2_REGULARIZATION_STRENGTH,
    help='Coefficient in the L2 regularization. '
         'Default: False')
  parser.add_argument('--silence_threshold', type=float,
    default=SILENCE_THRESHOLD,
    help='Volume threshold below which to trim the start '
         'and the end from the training set samples. Default: ' + str(SILENCE_THRESHOLD) + '.')

  args = parser.parse_args()

  args.samples = args.file_seconds * args.sample_rate
  print('Extracting %s samples per file' % args.samples)

  return args


if __name__ == '__main__':
  args = parse_args()
  main(args)
