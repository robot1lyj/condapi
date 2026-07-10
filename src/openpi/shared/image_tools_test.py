import jax.numpy as jnp
import torch

from openpi.shared import image_tools


def test_resize_with_pad_shapes():
    # Test case 1: Resize image with larger dimensions
    images = jnp.zeros((2, 10, 10, 3), dtype=jnp.uint8)  # Input images of shape (batch_size, height, width, channels)
    height = 20
    width = 20
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (2, height, width, 3)
    assert jnp.all(resized_images == 0)

    # Test case 2: Resize image with smaller dimensions
    images = jnp.zeros((3, 30, 30, 3), dtype=jnp.uint8)
    height = 15
    width = 15
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (3, height, width, 3)
    assert jnp.all(resized_images == 0)

    # Test case 3: Resize image with the same dimensions
    images = jnp.zeros((1, 50, 50, 3), dtype=jnp.uint8)
    height = 50
    width = 50
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert jnp.all(resized_images == 0)


def test_resize_with_pad_torch_preserves_batched_and_unbatched_rank():
    target = (8, 12)
    channels_last_batch = torch.zeros((1, 5, 7, 3), dtype=torch.float32)
    channels_first_batch = torch.zeros((1, 3, 5, 7), dtype=torch.float32)

    assert image_tools.resize_with_pad_torch(channels_last_batch, *target).shape == (1, *target, 3)
    assert image_tools.resize_with_pad_torch(channels_last_batch[0], *target).shape == (*target, 3)
    assert image_tools.resize_with_pad_torch(channels_first_batch, *target).shape == (1, 3, *target)
    assert image_tools.resize_with_pad_torch(channels_first_batch[0], *target).shape == (3, *target)

    # Test case 3: Resize image with odd-numbered padding
    images = jnp.zeros((1, 256, 320, 3), dtype=jnp.uint8)
    height = 60
    width = 80
    resized_images = image_tools.resize_with_pad(images, height, width)
    assert resized_images.shape == (1, height, width, 3)
    assert jnp.all(resized_images == 0)
