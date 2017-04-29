import tensorflow as tf
from wavenet.audio_reader import AudioReader
from wavenet.model import WaveNetModel
from wavenet.ops import mu_law_encode

class GanAudioReader(object):

  def __init__(self, args, sess, wavenet_params):
    self.args = args
    self.sess = sess
    self.wavenet_params = wavenet_params
    self.coord = tf.train.Coordinator()

    self.audio_reader = self._create_audio_reader()

    self.threads = tf.train.start_queue_runners(sess=sess, coord=self.coord)
    self.audio_reader.start_threads(sess, 1)


  def _create_audio_reader(self):
    # TODO Calculate receptive_field:
    receptive_field = WaveNetModel.calculate_receptive_field(self.wavenet_params["filter_width"],
      self.wavenet_params["dilations"],
      self.wavenet_params["scalar_input"],
      self.wavenet_params["initial_filter_width"]
    )
    # receptive_field = 1

    return AudioReader(
      self.args.audio_dir,
      self.coord,
      self.args.sample_rate,
      self.args.gc_enabled,
      receptive_field,
      sample_size=self.args.sample_size,
      silence_threshold=self.args.silence_threshold,
      queue_size=32
    )


  def next_audio_batch(self):
    audio_tensor = self.dequeue()
    # print(audio_tensor)

    audio_tensor = self._encode(audio_tensor)
    # print(audio_tensor)

    audio_tensor = tf.reshape(audio_tensor, [self.args.batch_size, -1])
    # print(audio_tensor)

    audio_tensor = tf.slice(audio_tensor, [0, 0], [-1, self.args.samples])
    # print(audio_tensor)

    samples = self.sess.run(audio_tensor)
    # print(samples)

    return samples


  def dequeue(self):
    return self.audio_reader.dequeue(self.args.batch_size)


  def _one_hot(self, input_batch):
    '''One-hot encodes the waveform amplitudes.
  
    This allows the definition of the network as a categorical distribution
    over a finite set of possible amplitudes.
    '''
    with tf.name_scope('one_hot_encode'):
      encoded = tf.one_hot(
        input_batch,
        depth=self.args.quantization_channels,
        dtype=tf.float32)

      shape = [self.args.batch_size, -1, self.args.quantization_channels]
      encoded = tf.reshape(encoded, shape)

    return encoded


  def _encode(self, input_batch):
    with tf.name_scope('wavenet'):
      # We mu-law encode and quantize the input audioform.
      encoded_input = mu_law_encode(input_batch,
        self.args.quantization_channels)

      # gc_embedding = self._embed_gc(global_condition_batch)
      encoded = self._one_hot(encoded_input)
      # if self.scalar_input:
      #   network_input = tf.reshape(
      #     tf.cast(input_batch, tf.float32),
      #     [self.batch_size, -1, 1])
      # else:
      #   network_input = encoded
      return encoded


  def done(self):
    self.coord.request_stop()
