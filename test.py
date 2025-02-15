import tensorflow as tf

# Check for GPU availability (TensorFlow 1.15 method)
with tf.compat.v1.Session() as sess:  # Use tf.compat.v1.Session for TF 1.x compatibility
    devices = sess.list_devices()
    gpu_available = any("/gpu:" in device.name.lower() for device in devices)

if gpu_available:
    print("GPU is available")
    for device in devices:
        if "/gpu:" in device.name.lower():
            print(device)

    # Simple test to ensure GPU is used
    with tf.device('/GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
        b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
        c = a + b
        result = sess.run(c)  # Run the computation within the session
        print(result)
else:
    print("GPU is NOT available")

print("TensorFlow version:", tf.__version__)