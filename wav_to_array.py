import numpy
from scipy.io.wavfile import read
from scipy.io.wavfile import write

a = read("/Users/annavillani/Desktop/piano_notes_64/39152__jobro__piano-ff-005.wav")
numpy.array(a[1], dtype=float)
print(numpy.amax(a[1]))
#write("/Users/annavillani/Desktop/sample.wav", 1, a[1])
#array of 30,924 arrays, each with depth 2

# for item in a[1]:
# 	print(item)

  #outer: 7424000 -> 30924
  #inner: 464000 -> 2