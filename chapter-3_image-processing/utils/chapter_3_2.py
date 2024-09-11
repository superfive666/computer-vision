import numpy as np


def constant_padding(image: np.ndarray, kernel_size: int, constant: int) -> np.ndarray:
    assert kernel_size % 2 == 1, "Kernel size has to be an odd number"
    assert constant >= 0 and constant <= 255, "invalid pixel intensity value for padding"
    pad = kernel_size // 2

    new_image = np.full((image.shape[0] + 2*pad, image.shape[1] + 2*pad, image.shape[2]), constant, dtype="uint8")
    new_image[pad:-pad,pad:-pad,:] = image

    return new_image.astype("uint8")


def zero_padding(image: np.ndarray, kernel_size: int) -> np.ndarray:
    
    return constant_padding(image, kernel_size, 0)


def clamp_padding(image: np.ndarray, kernel_size: int) -> np.ndarray:
    pad = kernel_size // 2
    new_image = zero_padding(image, kernel_size)
    # left
    new_image[pad:-pad,:pad,:] = np.stack([image[:,0,:].reshape(image.shape[0], image.shape[2])] * pad, axis=1)
    # right
    new_image[pad:-pad,-pad:,:] = np.stack([image[:,-1,:].reshape(image.shape[0], image.shape[2])] * pad, axis=1)
    # top
    new_image[:pad,:,:] = np.stack([new_image[pad,:,:].reshape(new_image.shape[1], new_image.shape[2])] * pad, axis=0)
    # bottom
    new_image[-pad:,:,:] = np.stack([new_image[-pad-1,:,:].reshape(new_image.shape[1], new_image.shape[2])] * pad, axis=0)

    return new_image


def wrap_padding(image: np.ndarray, kernel_size: int) -> np.ndarray:
    assert kernel_size % 2 == 1, "Kernel size has to be an odd number"
    assert kernel_size < min(image.shape[0], image.shape[1]), "padding should not exceed the original size of the image"
    pad = kernel_size // 2
    # now wrap with repeat configuration of the original image to form a 9-tile collage, 
    # where the original image should be in the center of the resulting tiles
    new_image = image
    height, width, _ = new_image.shape
    new_image = np.hstack([new_image] * 3)
    new_image = np.vstack([new_image] * 3)

    return new_image[height-pad:pad-height, width-pad:pad-width, :]


def mirror_padding(image: np.ndarray, kernel_size: int) -> np.ndarray:
    assert kernel_size < min(image.shape[0], image.shape[1]), "padding should not exceed the original size of the image"
    pad = kernel_size // 2
    # similar to the wrap padding, the mirror padding mirror the original image at the border
    new_image = wrap_padding(image, kernel_size)
    # swap left, right
    left, right = new_image[pad:-pad, :pad, :].copy(), new_image[pad:-pad, -pad:, :].copy()
    new_image[pad:-pad, :pad, :], new_image[pad:-pad, -pad:, :] = right[:,::-1,:], left[:,::-1,:]

    # swap top, bottom
    top, bottom = new_image[:pad, pad:-pad, :].copy(), new_image[-pad:, pad:-pad, :].copy()
    new_image[:pad, pad:-pad, :], new_image[-pad:, pad:-pad, :] = bottom[::-1,:,:], top[::-1,:,:]

    # swap corners
    c1, c2 = new_image[:pad, :pad, :].copy(), new_image[-pad:, -pad:, :].copy()
    new_image[:pad, :pad, :], new_image[-pad:, -pad:, :] = c2[::-1, ::-1, :], c1[::-1, ::-1, :]

    c1, c2 = new_image[:pad, -pad:, :].copy(), new_image[-pad:, :pad, :].copy()
    new_image[:pad, -pad:, :], new_image[-pad:, :pad, :] = c2[::-1, ::-1, :], c1[::-1, ::-1, :]

    return new_image


def extend_padding(image: np.ndarray, kernel_size: int) -> np.ndarray:
    assert kernel_size < min(image.shape[0], image.shape[1]), "padding should not exceed the original size of the image"

    mask = np.zeros(image.shape)
    mask = constant_padding(mask, kernel_size, 1)
    mask = np.ma.make_mask(mask == 0)
    clamp = np.ma.array(clamp_padding(image, kernel_size), mask=mask).astype("uint8")
    mirror = np.ma.array(mirror_padding(image, kernel_size), mask=mask).astype("uint8")

    return np.array(mirror - clamp).clip(min=0).astype("uint8")


def constant_padding_grayscale(image: np.ndarray, kernel_size: int, constant: int) -> np.ndarray:
    assert kernel_size % 2 == 1, "Kernel size has to be an odd number"
    assert constant >= 0 and constant <= 255, "invalid pixel intensity value for padding"
    pad = kernel_size // 2

    new_image = np.full((image.shape[0] + 2*pad, image.shape[1] + 2*pad), constant, dtype="uint8")
    new_image[pad:-pad,pad:-pad] = image

    return new_image.astype("uint8")


def zero_padding_grayscale(image: np.ndarray, kernel_size: int) -> np.ndarray:
    
    return constant_padding_grayscale(image, kernel_size, 0)


def clamp_padding_grayscale(image: np.ndarray, kernel_size: int) -> np.ndarray:
    pad = kernel_size // 2
    new_image = zero_padding_grayscale(image, kernel_size)
    # left
    new_image[pad:-pad,:pad] = np.stack([image[:,0].reshape(image.shape[0])] * pad, axis=1)
    # right
    new_image[pad:-pad,-pad:] = np.stack([image[:,-1].reshape(image.shape[0])] * pad, axis=1)
    # top
    new_image[:pad,:] = np.stack([new_image[pad,:].reshape(new_image.shape[1])] * pad, axis=0)
    # bottom
    new_image[-pad:,:] = np.stack([new_image[-pad-1,:].reshape(new_image.shape[1])] * pad, axis=0)

    return new_image


def wrap_padding_grayscale(image: np.ndarray, kernel_size: int) -> np.ndarray:
    assert kernel_size % 2 == 1, "Kernel size has to be an odd number"
    assert kernel_size < min(image.shape[0], image.shape[1]), "padding should not exceed the original size of the image"
    pad = kernel_size // 2
    # now wrap with repeat configuration of the original image to form a 9-tile collage, 
    # where the original image should be in the center of the resulting tiles
    new_image = image
    height, width, _ = new_image.shape
    new_image = np.hstack([new_image] * 3)
    new_image = np.vstack([new_image] * 3)

    return new_image[height-pad:pad-height, width-pad:pad-width, :]


def mirror_padding_grayscale(image: np.ndarray, kernel_size: int) -> np.ndarray:
    assert kernel_size < min(image.shape[0], image.shape[1]), "padding should not exceed the original size of the image"
    pad = kernel_size // 2
    # similar to the wrap padding, the mirror padding mirror the original image at the border
    new_image = wrap_padding(image, kernel_size)
    # swap left, right
    left, right = new_image[pad:-pad, :pad].copy(), new_image[pad:-pad, -pad:].copy()
    new_image[pad:-pad, :pad], new_image[pad:-pad, -pad:, :] = right[:,::-1], left[:,::-1]

    # swap top, bottom
    top, bottom = new_image[:pad, pad:-pad].copy(), new_image[-pad:, pad:-pad].copy()
    new_image[:pad, pad:-pad], new_image[-pad:, pad:-pad] = bottom[::-1,:], top[::-1,:]

    # swap corners
    c1, c2 = new_image[:pad, :pad].copy(), new_image[-pad:, -pad:].copy()
    new_image[:pad, :pad], new_image[-pad:, -pad:] = c2[::-1, ::-1], c1[::-1, ::-1]

    c1, c2 = new_image[:pad, -pad:].copy(), new_image[-pad:, :pad].copy()
    new_image[:pad, -pad:], new_image[-pad:, :pad] = c2[::-1, ::-1], c1[::-1, ::-1]

    return new_image


def extend_padding_grayscale(image: np.ndarray, kernel_size: int) -> np.ndarray:
    assert kernel_size < min(image.shape[0], image.shape[1]), "padding should not exceed the original size of the image"

    mask = np.zeros(image.shape)
    mask = constant_padding(mask, kernel_size, 1)
    mask = np.ma.make_mask(mask == 0)
    clamp = np.ma.array(clamp_padding_grayscale(image, kernel_size), mask=mask).astype("uint8")
    mirror = np.ma.array(mirror_padding_grayscale(image, kernel_size), mask=mask).astype("uint8")

    return np.array(mirror - clamp).clip(min=0).astype("uint8")