import tensorflow as tf

# Check if TensorFlow is built with CUDA support
print("Is TensorFlow built with CUDA?:", tf.test.is_built_with_cuda())

# List all physical devices, including GPUs
physical_devices = tf.config.list_physical_devices("GPU")
print("Physical devices available:", physical_devices)

# If GPUs are detected, print details
if physical_devices:
    print(f"Number of GPUs available: {len(physical_devices)}")
    for i, gpu in enumerate(physical_devices):
        print(f"GPU {i}: {gpu.name}")
else:
    print("No GPUs detected by TensorFlow.")
