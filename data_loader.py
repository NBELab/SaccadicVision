import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import tables
import math
import scipy.ndimage
from configs.config import saccade_resolution, scene_resolution, radius, mask_rad, batch_size, num_samples, num_of_saccades, set_epochs
from model_utils import create_circular_mask_opp, rgb2opp, rgb2opp_batch, norm_image_np
import os


def genGaussiankernel(width, sigma):
    """
    Generates a 2D Gaussian kernel.

    Parameters:
    - width (int): The size of the kernel.
    - sigma (float): The standard deviation of the Gaussian distribution.

    Returns:
    - kernel_2d (np.ndarray): A 2D Gaussian kernel.
    """
    x = np.arange(-int(width/2), int(width/2)+1, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, x)
    kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    kernel_2d = kernel_2d / np.sum(kernel_2d)
    return kernel_2d


def pyramid(im, sigma=1, prNum=6):
    """
    Generates a Gaussian pyramid for a given image.

    Parameters:
    - im (np.ndarray): The input image.
    - sigma (float): Standard deviation for Gaussian kernel.
    - prNum (int): Number of pyramid levels.

    Returns:
    - pyramids (list of np.ndarray): A list of images representing the pyramid.
    """
    height_ori, width_ori, ch = im.shape
    G = im.copy()
    pyramids = [G]
    Gaus_kernel2D = genGaussiankernel(5, sigma)
    for i in range(1, prNum):
        G = cv2.filter2D(G, -1, Gaus_kernel2D)
        height, width, _ = G.shape
        G = cv2.resize(G, (int(width/2), int(height/2)))
        pyramids.append(G)
    # upsample
    for i in range(1, 6):
        curr_im = pyramids[i]
        for j in range(i):
            if j < i-1:
                im_size = (curr_im.shape[1]*2, curr_im.shape[0]*2)
            else:
                im_size = (width_ori, height_ori)
            curr_im = cv2.resize(curr_im, im_size)
            curr_im = cv2.filter2D(curr_im, -1, Gaus_kernel2D)
        pyramids[i] = curr_im
    return pyramids


def foveat_img(im, fixs, p=7.5, k=3, alpha=2.5):
    """
    Applies a foveation effect to an image based on given fixation points.

    Parameters:
    - im (np.ndarray): The input image.
    - fixs (list of tuples): A list of fixation points as (x, y) coordinates.
    - p (float): A parameter controlling the spread of the foveation effect.
    - k (float): A parameter controlling the steepness of the foveation transition.
    - alpha (float): A parameter controlling the strength of the foveation effect.

    Returns:
    - im_fov (np.ndarray): The foveated image.
    """
    sigma = 0.248
    prNum = 6
    As = pyramid(im, sigma, prNum)
    height, width, _ = im.shape
    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, y)
    theta = np.sqrt((x2d - fixs[0][0]) ** 2 + (y2d - fixs[0][1]) ** 2) / p
    for fix in fixs[1:]:
        theta = np.minimum(theta, np.sqrt(
            (x2d - fix[0]) ** 2 + (y2d - fix[1]) ** 2) / p)
    R = alpha / (theta + alpha)

    Ts = []
    for i in range(1, prNum):
        Ts.append(np.exp(-((2 ** (i-3)) * R / sigma) ** 2 * k))
    Ts.append(np.zeros_like(theta))
    omega = np.zeros(prNum)
    for i in range(1, prNum):
        omega[i-1] = np.sqrt(np.log(2)/k) / (2**(i-3)) * sigma
    omega[omega > 1] = 1
    layer_ind = np.zeros_like(R)
    for i in range(1, prNum):
        ind = np.logical_and(R >= omega[i], R <= omega[i - 1])
        layer_ind[ind] = i
    Bs = []
    for i in range(1, prNum):
        Bs.append((0.5 - Ts[i]) / (Ts[i-1] - Ts[i] + 1e-5))
    Ms = np.zeros((prNum, R.shape[0], R.shape[1]))
    for i in range(prNum):
        ind = layer_ind == i
        if np.sum(ind) > 0:
            if i == 0:
                Ms[i][ind] = 1
            else:
                Ms[i][ind] = 1 - Bs[i-1][ind]
        ind = layer_ind - 1 == i
        if np.sum(ind) > 0:
            Ms[i][ind] = Bs[i][ind]
    im_fov = np.zeros_like(As[0], dtype=np.float32)
    for M, A in zip(Ms, As):
        for i in range(3):
            im_fov[:, :, i] += np.multiply(M, A[:, :, i])
    im_fov = im_fov
    return im_fov


class DataGenerator(keras.utils.Sequence):
    """
    DataGenerator for loading and processing saccades and events.
    """

    def __init__(self, filename, all_inds, h=scene_resolution[0], w=scene_resolution[1], batch_size=8, shuffle=True, preload_data=False):
        self.batch_size = batch_size
        self.h = h
        self.w = w
        self.mask = create_circular_mask_opp(
            saccade_resolution[0], saccade_resolution[1], radius=mask_rad)
        self.indexes = all_inds[:num_samples].astype(
            np.int32) if num_samples > 0 else all_inds.copy().astype(np.int32)
        self.shuffle = shuffle
        self.on_epoch_end()
        self.h5_file = tables.open_file(filename, 'r')
        self.data_set = self.h5_file.root

        # Define crop boundaries for the center region
        self.crop_row_start = int(scene_resolution[0] / 2) - radius
        self.crop_row_end = int(scene_resolution[0] / 2) + radius
        self.crop_col_start = int(scene_resolution[1] / 2) - radius
        self.crop_col_end = int(scene_resolution[1] / 2) + radius

    def __len__(self):
        # Number of batches per epoch
        return self.indexes.shape[0] // self.batch_size

    def __getitem__(self, index):
        # Generate one batch of data
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]
        events, saccades = self.data_set.simple_event_frames[indexes, ...], norm_image_np(
            self.data_set.saccades[indexes, ...])
        num_of_events = self.data_set.num_of_event_frames[indexes, ...]
        center_points = self.data_set.points[indexes, ...].astype(
            np.int32)
        gaze_points = self.data_set.gaze_points[indexes, ...].astype(np.int32)
        scenes = norm_image_np(
            self.data_set.scenes[indexes, ...].astype(np.int32))
        new_events = []
        max_events = np.max(num_of_events)
        i = 0

        # # Save scenes and events for debugging
        # os.makedirs("debug", exist_ok=True)
        # for i in range(scenes.shape[0]):
        #     cv2.imwrite(f"debug/scene_{index}_{i}.png",
        #                 (scenes[i] * 255).astype(np.uint8))
        # for i in range(events.shape[0]):
        #     for j in range(events.shape[1]):
        #         for k in range(5, 9):
        #             debug_event = (norm_image_np(
        #                 events[i][j][k]) * 255).astype(np.uint8)
        #             cv2.imwrite(
        #                 f"debug/event_{index}_{i}_{j}_{k}.png", debug_event)

        for event in events:
            event_lsts = []
            real_world = np.zeros((scene_resolution))
            for m in range(num_of_saccades):
                event_lst = []
                for j in range(num_of_events[i][m]):
                    if center_points[i][m][j][0] == 0 and center_points[i][m][j][1] == 0:
                        num_of_events[i][m] = j
                        continue
                    real_world[center_points[i][m][j][0] - radius:center_points[i][m][j][0] + radius,
                               center_points[i][m][j][1] - radius:center_points[i][m][j][1] + radius] += event[m][j]

                    if (j % 25 == 0 and j != 0) or j == num_of_events[i][m] - 1:
                        real_world = norm_image_np(real_world)
                        event_lst.append(
                            real_world[self.crop_row_start:self.crop_row_end, self.crop_col_start:self.crop_col_end])

                        # # debug
                        # cv2.imwrite(
                        #     f"debug/single_evento_added_{index}_{i}_{m}_{j}.png", (norm_image_np(real_world[self.crop_row_start:self.crop_row_end, self.crop_col_start:self.crop_col_end])*255).astype(np.uint8))

                        real_world = np.zeros((scene_resolution))
                num_of_events[i][m] = math.ceil(num_of_events[i][m] / 25)
                while len(event_lst) < 10:
                    event_lst.append(np.zeros(saccade_resolution))
                event_lst = np.array(event_lst)

                # # debug x
                # os.makedirs("debug", exist_ok=True)
                # for s in range(10):
                #     cv2.imwrite(f"debug/event{m}{j}{s}.png",
                #                 (event_lst[s] * 255).astype(np.uint8))

                event_lsts.append(event_lst)
            new_events.append(np.array(event_lsts))
            i += 1
        new_events = np.array(new_events)

        # print(
        #     f'max of events: {np.max(new_events)}, min of events: {np.min(new_events)}')

        # Process saccade images
        fov_arr_list = []
        for saccade_idx in range(num_of_saccades + 1):
            fov_arr = self.process_saccade_images(saccades[:, saccade_idx], gaze_points, xc=int(saccade_resolution[0] / 2),
                                                  yc=int(saccade_resolution[1] / 2), p=2, mask=self.mask, channel_index=saccade_idx)
            fov_arr_list.append(fov_arr)

        # Concatenate foveated images and events
        concatenated_list = []
        for i in range(num_of_saccades + 1):
            if i == 0:
                concatenated = np.concatenate(
                    (fov_arr_list[i], np.zeros_like(new_events[:, 0, :, :, :])), axis=1)
            else:
                concatenated = np.concatenate(
                    (fov_arr_list[i], new_events[:, i - 1, :, :, :]), axis=1)
            concatenated_list.append(concatenated)

        # Stack all saccade arrays
        x = np.stack(concatenated_list, axis=1)
        x = x[:, 0:num_of_saccades + 1]

        at = 0
        y_opp = rgb2opp_batch(scenes[:])
        for i in range(y_opp.shape[0]):  # go over all center imgs
            y_opp[at, :, :, 2] = scipy.ndimage.laplace(y_opp[at, :, :, 2])
            at += 1
        y_opp = tf.transpose(y_opp, perm=[0, 3, 1, 2])

        y_opp = y_opp[:, :, self.crop_row_start:self.crop_row_end,
                      self.crop_col_start:self.crop_col_end]  # Crop to center

        x = np.expand_dims(x, axis=-1)

        # debug x
        os.makedirs("debug", exist_ok=True)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for k in range(x.shape[2]):

                    cv2.imwrite(
                        f"debug/x_{index}_{i}_{j}_{k}.png", (norm_image_np(x[i][j][k]) * 255).astype(np.uint8))

        return x, y_opp

    def process_saccade_images(self, saccade_images, gaze_points, xc, yc, p, mask, channel_index):
        fov_lst = []
        for at, img in enumerate(saccade_images):
            foveat = rgb2opp(foveat_img(img, [(xc, yc)], p=p))
            foveat[:, :, 2] = scipy.ndimage.laplace(foveat[:, :, 2])
            foveat[~mask] = 0
            new_arr = np.zeros((200, 200, 3), dtype=foveat.dtype)
            start_row = gaze_points[at, channel_index,
                                    0] - foveat.shape[0] // 2
            start_col = gaze_points[at, channel_index,
                                    1] - foveat.shape[1] // 2
            new_arr[start_row:start_row + foveat.shape[0],
                    start_col:start_col + foveat.shape[1], :] = foveat
            foveat = new_arr[self.crop_row_start:self.crop_row_end,
                             self.crop_col_start:self.crop_col_end]
            fov_lst.append(foveat.transpose((2, 0, 1)))
        return np.array(fov_lst)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __del__(self):
        self.h5_file.close()
