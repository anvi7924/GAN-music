import tensorflow as tf
from src.wavenet.audio_reader import AudioReader

input_dir = '/Users/pstover/workspace/school/csci5622/project/data/magnatagatune/wav/first_5'
coord = tf.train.Coordinator()
reader = AudioReader(audio_dir=input_dir, coord=coord, sample_rate=100, gc_enabled=False, receptive_field=1)

sample = reader.dequeue(1)
print(sample)

coord.request_stop()