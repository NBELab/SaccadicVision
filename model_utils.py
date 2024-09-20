import random
import numpy as np
import tensorflow as tf
from scipy.fftpack import idst, dst

from configs.config import saccade_resolution
from configs.config import saccade_h, saccade_w
from skimage.color import rgb2gray

# Creates a circular mask covering the color channels.


def create_circular_mask_opp(h=saccade_h, w=saccade_w, center=(saccade_h/2, saccade_w/2), radius=None):
    """
    Creates a circular mask for an image, with the third channel left untouched.

    Parameters:
    - h (int): The height of the image.
    - w (int): The width of the image.
    - center (tuple): The center of the circle.
    - radius (int): The radius of the circle.

    Returns:
    - mask (np.ndarray): The resulting circular mask, with the third channel set to 1.
    """
    if center is None:  # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
    mask[:, :, 2] = np.ones((h, w))
    return mask


def rgb2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb2opp(rgb):
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    O3 = rgb2gray(rgb)
    O1 = (R-G)/np.sqrt(2)
    O2 = (R + G - 2*B)/np.sqrt(6)
    return np.stack((O1, O2, O3), axis=-1)


def rgb2opp_batch(rgb):
    R, G, B = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
    O3 = rgb2gray(rgb)
    O1 = (R-G)/np.sqrt(2)
    O2 = (R + G - 2*B)/np.sqrt(6)
    return np.stack((O1, O2, O3), axis=-1)


@tf.function
def opp2rgb_tf(opp):
    a = 0.2989
    b = 0.587
    c = 0.114
    opp = tf.cast(opp, tf.float32)
    sqrt2 = tf.constant(np.sqrt(2), tf.float32)
    sqrt6 = tf.constant(np.sqrt(6), tf.float32)
    opp = tf.unstack(opp, axis=-1)
    O1, O2, O3 = opp[0], opp[1], opp[2]
    B = (O3-sqrt2/2*(a-b)*O1 - sqrt6/2*(a+b)*O2) / (a+b+c)
    G = (sqrt6*O2-sqrt2*O1+2*B)/2
    R = sqrt2*O1+G
    return tf.stack((R, G, B), axis=-1)


def opp2rgb(opp):
    a = 0.2989
    b = 0.587
    c = 0.114
    opp = opp.astype(np.float32)
    sqrt2 = np.sqrt(2).astype(np.float32)
    sqrt6 = np.sqrt(6).astype(np.float32)
    O1, O2, O3 = np.split(opp, 3, axis=-1)
    O1 = np.squeeze(O1, axis=-1)
    O2 = np.squeeze(O2, axis=-1)
    O3 = np.squeeze(O3, axis=-1)
    B = (O3 - sqrt2/2*(a-b)*O1 - sqrt6/2*(a+b)*O2) / (a+b+c)
    G = (sqrt6*O2 - sqrt2*O1 + 2*B) / 2
    R = sqrt2*O1 + G
    return np.stack((R, G, B), axis=-1)


def norm_image_np(img):
    img_min = np.min(img)
    img_max = np.max(img)
    if (img_max == img_min):
        return img
    return (img - img_min) / (img_max - img_min)


def denum_matrix(n_row, n_col):
    """
    Generate a matrix used for normalization in the reconstruction process.

    Args:
        n_row (int): Number of rows in the matrix.
        n_col (int): Number of columns in the matrix.

    Returns:
        np.ndarray: Flattened matrix after applying cosine transformations.
    """
    x = np.linspace(1, n_col, n_col)
    y = np.linspace(1, n_row, n_row)
    xv, yv = np.meshgrid(x, y)
    return np.transpose(((2*np.cos(np.pi*xv/(n_col+1))-2)+(2*np.cos(np.pi*yv/(n_row+1))-2)).flatten())


# https://stackoverflow.com/questions/53875821/scipy-generate-nxn-discrete-cosine-matrix
@tf.function
def solve_poiss_tf(grad):
    """
    Solve the Poisson equation for a given Laplacian image using TensorFlow operations.

    Args:
        grad (tf.Tensor): The Laplacian image tensor.

    Returns:
        tf.Tensor: The reconstructed intensity image tensor.
    """
    denum_mat = denum_matrix(saccade_h, saccade_h)
    denum_mat_ei = np.reshape(1/denum_mat, (saccade_h, saccade_h))
    dst_mat_100 = dst(np.eye(saccade_h), type=2, axis=0)
    dst_mat_70 = dst(np.eye(saccade_h), type=2, axis=0)
    idst_mat_100 = idst(np.eye(saccade_h), type=2, axis=0)
    idst_mat_70 = idst(np.eye(saccade_h), type=2, axis=0)
    c_denum_mat_ei = tf.constant(denum_mat_ei, dtype=tf.float32)
    c_dst_mat_100 = tf.constant(dst_mat_100, dtype=tf.float32)
    c_dst_mat_70 = tf.constant(dst_mat_70, dtype=tf.float32)
    c_idst_mat_100 = tf.constant(idst_mat_100, dtype=tf.float32)
    c_idst_mat_70 = tf.constant(idst_mat_70, dtype=tf.float32)

    z = tf.transpose(tf.matmul(c_dst_mat_100, tf.transpose(
        tf.matmul(c_dst_mat_70, grad), [0, 2, 1])), [0, 2, 1])
    d = c_denum_mat_ei*z
    res = tf.transpose(tf.matmul(c_idst_mat_100, tf.transpose(
        tf.matmul(c_idst_mat_70, d), [0, 2, 1])), [0, 2, 1])
    return res


@tf.function
def norm_image(img):
    """
    Normalize the pixel values of the image.

    Args:
        img (tf.Tensor): The image tensor to be normalized.

    Returns:
        tf.Tensor: The normalized image tensor.
    """
    min, max = tf.reduce_min(img), tf.reduce_max(img)
    return (img-min)/(max-min)


def solve_poisson_mat(img_grad):
    """
    Solve the Poisson equation for a given Laplacian image using NumPy operations.

    Args:
        img_grad (np.ndarray): The Laplacian image array (3D with channels or 2D).

    Returns:
        np.ndarray: The reconstructed intensity image array.
    """

    denum_mat = denum_matrix(saccade_h, saccade_h)
    denum_mat_ei = np.reshape(1/denum_mat, (saccade_h, saccade_h))
    dst_mat_100 = dst(np.eye(saccade_h), type=2, axis=0)
    dst_mat_70 = dst(np.eye(saccade_h), type=2, axis=0)
    idst_mat_100 = idst(np.eye(saccade_h), type=2, axis=0)
    idst_mat_70 = idst(np.eye(saccade_h), type=2, axis=0)

    if img_grad.ndim == 3:
        x = img_grad[:, :, 0]
    else:
        x = img_grad
    z = (dst_mat_100 @ (dst_mat_70 @ x).T).T
    d = denum_mat_ei*z
    res = (idst_mat_100 @ (idst_mat_70 @ d).T).T
    return res
