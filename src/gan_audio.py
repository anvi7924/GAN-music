import argparse
import json

from datetime import datetime

import librosa
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
  samples = tf.placeholder(tf.int32)
  next_sample = net.predict_proba(samples, None)

  quantization_channels = wavenet_params['quantization_channels']

  # Silence with a single random sample at the end.
  waveform = [quantization_channels / 2] * (net.receptive_field - 1)
  waveform.append(np.random.randint(quantization_channels))

  last_sample_timestamp = datetime.now()
  for step in range(args.samples):
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

  print()

  # Save the result as a wav file.
  # if args.wav_out_patddh:
  decode = mu_law_decode(samples, wavenet_params['quantization_channels'])

  out = sess.run(decode, feed_dict={samples: waveform})
  return out


def write_wav(waveform, sample_rate, filename):
  y = np.array(waveform)
  librosa.output.write_wav(filename, y, sample_rate)
  print('Updated wav file at {}'.format(filename))

def discriminator(x):
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
  )

  if args.l2_regularization_strength == 0:
    args.l2_regularization_strength = None

  return net


def main(args):
  X = tf.placeholder(tf.float32, shape=[None, args.samples])
  Z = tf.placeholder(tf.int32, shape=[None, 100])

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
    l2_regularization_strength=args.l2_regularization_strength)

  sess.run(tf.global_variables_initializer())

  print('Initializing Generator...')
  G_sample = generator(Z, net)
  D_fake, D_logit_fake = discriminator(G_sample)

  print('Initializing Generator loss function...')
  G_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

  print('Creating optimizer...')
  optimizer = tf.train.AdamOptimizer(
    learning_rate=args.learning_rate,
    epsilon=0.01
  )
  trainable = tf.trainable_variables()

  print("Creating D_solver...")
  D_solver = optimizer.minimize(D_loss, var_list=trainable)

  print("Creating G_solver...")
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=trainable)

  Z_dim = 100

  for it in range(args.iters):
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={Z: sample_Z(args.batch_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(args.batch_size, Z_dim)})

    print('Iter: {}'.format(it))
    print('D loss: {:.4}'.format(D_loss_curr))
    print('G_loss: {:.4}'.format(G_loss_curr))
    print('\n')

  audio_reader.done()


LEARNING_RATE = 1e-3
WAVENET_PARAMS = '../config/wavenet_params.json'
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = None
MOMENTUM = 0.9


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
