import tensorflow as tf

print(tf.__version__)
print(tf.test.is_gpu_available(cuda_only=True))
print(tf.config.list_physical_devices('GPU'))