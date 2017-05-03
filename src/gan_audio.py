import argparse
import json

from datetime import datetime

import librosa
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

from gan_audio_reader import GanAudioReader
from wavenet.model import WaveNetModel
from wavenet.ops import *


def xavier_init(size):
  in_dim = size[0]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
  return tf.random_normal(shape=size, stddev=xavier_stddev)

def sample_Z(m, n):
  return np.random.uniform(-1., 1., size=[m, n])


def generator(z, net):
  return net.predict_proba_incremental(z)


def generate(z, args, net, wavenet_params, sess):
  # def main():
  #   args = get_arguments()
  # started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
  # logdir = os.path.join(args.logdir, 'generate', started_datestring)
  # with open(args.wavenet_params, 'r') as config_file:
  #   wavenet_params = json.load(config_file)

  # sess = tf.Session()

  # net = WaveNetModel(
  #   batch_size=1,
  #   dilations=wavenet_params['dilations'],
  #   filter_width=wavenet_params['filter_width'],
  #   residual_channels=wavenet_params['residual_channels'],
  #   dilation_channels=wavenet_params['dilation_channels'],
  #   quantization_channels=wavenet_params['quantization_channels'],
  #   skip_channels=wavenet_params['skip_channels'],
  #   use_biases=wavenet_params['use_biases'],
  #   scalar_input=wavenet_params['scalar_input'],
  #   initial_filter_width=wavenet_params['initial_filter_width'],
  #   global_condition_channels=args.gc_channels,
  #   global_condition_cardinality=args.gc_cardinality)

  samples = tf.placeholder(tf.int32)

  # if args.fast_generation:
  #   next_sample = net.predict_proba_incremental(samples, args.gc_id)
  # else:
  # next_sample = net.predict_proba(samples, args.gc_id)
  next_sample = net.predict_proba(samples, None)

  # if args.fast_generation:
  #   sess.run(tf.global_variables_initializer())
  #   sess.run(net.init_ops)

  # variables_to_restore = {
  #   var.name[:-2]: var for var in tf.global_variables()
  #   if not ('state_buffer' in var.name or 'pointer' in var.name)}
  # saver = tf.train.Saver(variables_to_restore)
  #
  # print('Restoring model from {}'.format(args.checkpoint))
  # saver.restore(sess, args.checkpoint)

  # decode = mu_law_decode(samples, wavenet_params['quantization_channels'])

  quantization_channels = wavenet_params['quantization_channels']
  # if args.wav_seed:
  #   seed = create_seed(args.wav_seed,
  #     wavenet_params['sample_rate'],
  #     quantization_channels,
  #     net.receptive_field)
  #   waveform = sess.run(seed).tolist()
  # else:

  # TODO Need to make this noisier?
  # Silence with a single random sample at the end.
  waveform = [quantization_channels / 2] * (net.receptive_field - 1)
  waveform.append(np.random.randint(quantization_channels))

  # if args.fast_generation and args.wav_seed:
  #   # When using the incremental generation, we need to
  #   # feed in all priming samples one by one before starting the
  #   # actual generation.
  #   # TODO This could be done much more efficiently by passing the waveform
  #   # to the incremental generator as an optional argument, which would be
  #   # used to fill the queues initially.
  #   outputs = [next_sample]
  #   outputs.extend(net.push_ops)
  #
  #   print('Priming generation...')
  #   for i, x in enumerate(waveform[-net.receptive_field: -1]):
  #     if i % 100 == 0:
  #       print('Priming sample {}'.format(i))
  #     sess.run(outputs, feed_dict={samples: x})
  #   print('Done.')

  last_sample_timestamp = datetime.now()
  for step in range(args.samples):
    # if args.fast_generation:
    #   outputs = [next_sample]
    #   outputs.extend(net.push_ops)
    #   window = waveform[-1]
    # else:
    if len(waveform) > net.receptive_field:
      window = waveform[-net.receptive_field:]
    else:
      window = waveform
    outputs = [next_sample]

    # Run the WaveNet to predict the next sample.
    prediction = sess.run(outputs, feed_dict={samples: window})[0]

    # Scale prediction distribution using temperature.
    temperature = 1.0
    np.seterr(divide='ignore')
    scaled_prediction = np.log(prediction) / temperature
    scaled_prediction = (scaled_prediction -
                         np.logaddexp.reduce(scaled_prediction))
    scaled_prediction = np.exp(scaled_prediction)
    np.seterr(divide='warn')

    # Prediction distribution at temperature=1.0 should be unchanged after
    # scaling.
    # if args.temperature == 1.0:
    np.testing.assert_allclose(
      prediction, scaled_prediction, atol=1e-5,
      err_msg='Prediction scaling at temperature=1.0 '
              'is not working as intended.')

    sample = np.random.choice(
      np.arange(quantization_channels), p=scaled_prediction)
    waveform.append(sample)

    # Show progress only once per second.
    current_sample_timestamp = datetime.now()
    time_since_print = current_sample_timestamp - last_sample_timestamp
    if time_since_print.total_seconds() > 1.:
      print('Sample {:3<d}/{:3<d}'.format(step + 1, args.samples))
      last_sample_timestamp = current_sample_timestamp

    # If we have partial writing, save the result so far.
    # if (args.wav_out_path and args.save_every and
    #       (step + 1) % args.save_every == 0):
    #   out = sess.run(decode, feed_dict={samples: waveform})
    #   write_wav(out, args.sample_rate, args.wav_out_path)

  # Introduce a newline to clear the carriage return from the progress.
  print()

  # Save the result as an audio summary.
  # datestring = str(datetime.now()).replace(' ', 'T')
  # writer = tf.summary.FileWriter(logdir)
  # tf.summary.audio('generated', decode, wavenet_params['sample_rate'])
  # summaries = tf.summary.merge_all()
  # summary_out = sess.run(summaries,
  #   feed_dict={samples: np.reshape(waveform, [-1, 1])})
  # writer.add_summary(summary_out)

  # Save the result as a wav file.
  # if args.wav_out_patddh:
  decode = mu_law_decode(samples, wavenet_params['quantization_channels'])

  out = sess.run(decode, feed_dict={samples: waveform})
  #   write_wav(out, args.sample_rate, args.wav_out_path)

  # print('Finished generating.')
  return out


def write_wav(waveform, sample_rate, filename):
  y = np.array(waveform)
  librosa.output.write_wav(filename, y, sample_rate)
  print('Updated wav file at {}'.format(filename))


# def generator(z, G_W1, G_b1, G_W2, G_b2):
#   G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
#   G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
#   G_prob = tf.nn.sigmoid(G_log_prob)
#
#   return G_prob
#
#
# def discriminator(x, D_W1, D_b1, D_W2, D_b2):
#   D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
#   D_logit = tf.matmul(D_h1, D_W2) + D_b2
#   D_prob = tf.nn.sigmoid(D_logit)
#
#   return D_prob, D_logit


def discriminator(x):
  # TODO Hard-coded at p = 1/2
  D_logit = tf.zeros_like(x)
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


def create_wavenet(args, wavenet_params):
  # Create network.
  net = WaveNetModel(
    batch_size=args.batch_size,
    dilations=wavenet_params["dilations"],
    filter_width=wavenet_params["filter_width"],
    residual_channels=wavenet_params["residual_channels"],
    dilation_channels=wavenet_params["dilation_channels"],
    skip_channels=wavenet_params["skip_channels"],
    quantization_channels=wavenet_params["quantization_channels"],
    use_biases=wavenet_params["use_biases"],
    scalar_input=wavenet_params["scalar_input"],
    initial_filter_width=wavenet_params["initial_filter_width"],
    # histograms=args.histograms,
    # global_condition_channels=args.gc_channels,
    # global_condition_cardinality=reader.gc_category_cardinality)
  )

  if args.l2_regularization_strength == 0:
    args.l2_regularization_strength = None

  return net


def main(args):
  X = tf.placeholder(tf.float32, shape=[None, args.samples])

  # D_W1 = tf.Variable(xavier_init([args.samples, 128]))
  # D_b1 = tf.Variable(tf.zeros(shape=[128]))
  #
  # D_W2 = tf.Variable(xavier_init([128, 1]))
  # D_b2 = tf.Variable(tf.zeros(shape=[1]))
  #
  # theta_D = [D_W1, D_W2, D_b1, D_b2]

  # Z = tf.placeholder(tf.float32, shape=[None, 100])
  Z = tf.placeholder(tf.int32, shape=[None, 100])

  # G_W1 = tf.Variable(xavier_init([100, 128]))
  # G_b1 = tf.Variable(tf.zeros(shape=[128]))
  #
  # G_W2 = tf.Variable(xavier_init([128, args.samples]))
  # G_b2 = tf.Variable(tf.zeros(shape=[args.samples]))
  #
  # theta_G = [G_W1, G_W2, G_b1, G_b2]

  # G_sample = generator(Z, G_W1, G_b1, G_W2, G_b2)
  # D_real, D_logit_real = discriminator(X, D_W1, D_b1, D_W2, D_b2)
  # D_fake, D_logit_fake = discriminator(G_sample, D_W1, D_b1, D_W2, D_b2)

  # D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
  # G_loss = -tf.reduce_mean(tf.log(D_fake))

  # Alternative losses:
  # -------------------
  # D_loss_real = tf.reduce_mean(
  #   tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
  # D_loss_fake = tf.reduce_mean(
  #   tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
  # D_loss = D_loss_real + D_loss_fake

  ###
  # Start WaveNet stuff
  ###
  sess = tf.Session()

  with open(args.wavenet_params, 'r') as f:
    wavenet_params = json.load(f)

  print('Creating reader...')
  audio_reader = GanAudioReader(args, sess, wavenet_params)

  print('Creating WaveNet...')
  net = create_wavenet(args, wavenet_params)

  print('Initializing audio batch...')
  audio_batch = audio_reader.dequeue()

  print('Initializing Discriminator loss function...')
  D_loss = net.loss(input_batch=audio_batch,
    # global_condition_batch=gc_id_batch,
    l2_regularization_strength=args.l2_regularization_strength)
  ###
  # End WaveNet stuff
  ###

  sess.run(tf.global_variables_initializer())

  print('Initializing Generator...')
  # G_sample = generator(Z, args, net, wavenet_params, sess)
  G_sample = generator(Z, net)
  D_fake, D_logit_fake = discriminator(G_sample)

  print('Initializing Generator loss function...')
  G_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

  print('Creating optimizer...')
  optimizer = tf.train.AdamOptimizer(
    learning_rate=args.learning_rate,
    # epsilon=1e-4
    epsilon=0.01
  )
  trainable = tf.trainable_variables()

  print("Creating D_solver...")
  D_solver = optimizer.minimize(D_loss, var_list=trainable)

  print("Creating G_solver...")
  # G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=trainable)

  Z_dim = 100

  # if not os.path.exists('out/'):
  #   os.makedirs('out/')

  i = 0
  for it in range(args.iters):
    # X_mb, _ = mnist.train.next_batch(mb_size)
    # X_mb = audio_reader.next_audio_batch()

    # _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(args.batch_size, Z_dim)})
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={Z: sample_Z(args.batch_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(args.batch_size, Z_dim)})

    # if it % 1000 == 0:
    print('Iter: {}'.format(it))
    print('D loss: {:.4}'.format(D_loss_curr))
    print('G_loss: {:.4}'.format(G_loss_curr))
    print('\n')

  audio_reader.done()


LEARNING_RATE = 1e-3
WAVENET_PARAMS = '../config/wavenet_params.json'
# STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = None
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
