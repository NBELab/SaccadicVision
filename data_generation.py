# # from google.colab import drive # access data in Google Drive
# import sklearn
# import tables
# import glob
# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
# from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import StandardScaler
# from skimage.io import imread_collection
# from sklearn.datasets import load_sample_image
# import pandas as pd
from PIL import Image
import numpy as np
import cv2
import os
import os.path
import configs.v2e_config as v2e_config
from configs.data_generation_config import *
from model_utils import rgb2bgr, bgr2rgb
import random
from sklearn.cluster import KMeans
from skimage.draw import line
import h5py
import shutil
import tables
import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import argparse


def crop_center(img, crop_x, crop_y):
    y, x, _ = img.shape
    start_x = x // 2 - (crop_x // 2)
    start_y = y // 2 - (crop_y // 2)
    return img[start_y:start_y+crop_y, start_x:start_x+crop_x]


def resize_and_crop_image(img, target_size, output_dataset_folder, file_name):

    h, w = target_size
    img_name, extension = file_name.split('.')
    # Crop the image to maintain aspect ratio
    img_np = np.array(img)
    img_cropped = crop_center(img_np, min(
        w, img_np.shape[1]), min(h, img_np.shape[0]))

    # Resize the image to the target size
    img_resized = cv2.resize(img_cropped, (w, h), interpolation=cv2.INTER_AREA)

    # Get the file name without extension and create the new folder name
    new_folder_name = img_name
    new_file_name = f"{img_name}.{extension}"

    # Create the corresponding output directory
    output_dir = os.path.join(output_dataset_folder, new_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, new_file_name)

    cv2.imwrite(output_path, img_resized)
    print(f"Processed {file_name} -> {output_path}")
    return img_resized


def process_single_video(read_folder, name):
    """
    Process a single video using the v2e tool.

    Args:
        read_folder (str): Path to the folder containing the video.
        name (str): Name of the video file (without extension).

    Returns:
        None
    """
    # Set up paths
    fold = os.path.join(read_folder, name)
    video_path = os.path.join(fold, f"{name}.avi")
    output_folder = os.path.join(fold, "v2e")

    # Check if events file already exists
    if not os.path.exists(os.path.join(output_folder, "events.h5")):
        # Build the v2e command
        v2e_command = ["v2e"]
        v2e_command += ["-i", video_path]
        v2e_command += ["-o", output_folder]
        v2e_command.append("--overwrite")
        v2e_command += ["--unique_output_folder",
                        str(v2e_config.unique_output_folder).lower()]

        if v2e_config.davis_output:
            v2e_command += ["--davis_output"]

        v2e_command += ["--dvs_h5", v2e_config.out_filename]
        v2e_command += ["--dvs_aedat2", "None"]
        v2e_command += ["--dvs_text", "None"]
        v2e_command += ["--no_preview"]

        if v2e_config.skip_video_output:
            v2e_command += ["--skip_video_output"]
        else:
            v2e_command += ["--dvs_exposure", v2e_config.dvs_exposure]

        v2e_command += ["--input_frame_rate", str(v2e_config.input_frame_rate)]
        v2e_command += ["--input_slowmotion_factor",
                        str(v2e_config.input_slowmotion_factor)]

        if v2e_config.disable_slomo:
            v2e_command += ["--disable_slomo"]
            v2e_command += ["--auto_timestamp_resolution", "false"]
        else:
            v2e_command += ["--slomo_model", v2e_config.slomo_model]
            if v2e_config.auto_timestamp_resolution:
                v2e_command += ["--auto_timestamp_resolution",
                                str(v2e_config.auto_timestamp_resolution).lower()]
            else:
                v2e_command += ["--timestamp_resolution",
                                str(v2e_config.timestamp_resolution)]

        v2e_command += ["--pos_thres", str(v2e_config.thres)]
        v2e_command += ["--neg_thres", str(v2e_config.thres)]
        v2e_command += ["--sigma_thres", str(v2e_config.sigma)]
        v2e_command += ["--cutoff_hz", str(v2e_config.cutoff_hz)]
        v2e_command += ["--leak_rate_hz", str(v2e_config.leak_rate_hz)]
        v2e_command += ["--shot_noise_rate_hz",
                        str(v2e_config.shot_noise_rate_hz)]
        v2e_command += [f"--{v2e_config.output_mode}"]

        # Join command into a single string
        final_v2e_command = " ".join(v2e_command)

        # Run the command
        os.system(final_v2e_command)

    print(f"Processed video: {name}")


def generate_random_points(img, num_of_saccades, radius, resolution):
    """
    Generate random gaze center points for saccades.

    Parameters:
    - img (numpy.ndarray): The input image frame (not used in this function).
    - num_of_saccades (int): The number of saccades to generate.
    - radius (int): The radius to consider around each point.
    - resolution (tuple): The (height, width) resolution of the image.

    Returns:
    - list: A list of random points for saccades, including the center of the image.
    """
    minX = radius
    minY = radius
    maxX = resolution[0] - radius
    maxY = resolution[1] - radius

    points = []

    # Generate random points
    for _ in range(num_of_saccades):
        x = random.randint(minX, maxX)
        y = random.randint(minY, maxY)
        points.append((x, y))

    # Include the center of the image as the first point
    mid = (int(resolution[0] / 2 - 1), int(resolution[1] / 2 - 1))
    points.insert(0, mid)

    return points


def generate_good_points(img, num_of_saccades, radius, resolution):
    """
    Generate good points for saccades using GoodFeaturesToTrack and KMeans clustering.

    Parameters:
    - img (numpy.ndarray): The input image frame.
    - num_of_saccades (int): The number of saccades to generate.
    - radius (int): The radius to consider around each point.
    - resolution (tuple): The (height, width) resolution of the image.

    Returns:
    - list: A list of good points for saccades, including the center of the image.
    """

    # Define the region of interest, excluding the borders
    limited_img = img[radius+1:resolution[0] -
                      radius-2, radius+1:resolution[1]-radius-2]

    # Convert the image to grayscale
    gray = cv2.cvtColor(limited_img, cv2.COLOR_RGB2GRAY)

    # Detect corners in the image
    corners = cv2.goodFeaturesToTrack(gray, 25, 0.01, 10)

    if corners is None:
        # If no corners are found, return random points
        return generate_random_points(img)

    # Convert corners to integer coordinates
    corners = np.int0(corners)
    corner_list = [tuple(i.ravel()) for i in corners]

    # If fewer corners than the required number, generate random points
    if len(corner_list) < num_of_saccades:
        return generate_random_points(img)

    # Apply KMeans clustering to the corners
    kmeans = KMeans(
        init="random",
        n_clusters=num_of_saccades,
        n_init=10,
        max_iter=300,
        random_state=42
    )
    kmeans.fit(corner_list)
    centers = kmeans.cluster_centers_

    # Adjust cluster centers to fit within image boundaries
    fixed_centers = []
    for center in centers:
        point = np.array([
            int(center[0] + radius),
            int(center[1] + radius)
        ])

        # Ensure points are within image boundaries
        point[0] = np.clip(point[0], radius, resolution[0] - radius - 2)
        point[1] = np.clip(point[1], radius, resolution[1] - radius - 2)

        fixed_centers.append(point)

    # Include the center of the image as the first point
    mid = np.array([
        int(resolution[0] / 2 - 1),
        int(resolution[1] / 2 - 1)
    ])
    fixed_centers.insert(0, mid)

    return fixed_centers


def generate_saccades(scene_img, gaze_points, radius, dataset_directory, file_name):
    """
    Generate intersaccadic and saccade frames from a scene image based on gaze points.

    Parameters:
    - scene_img (numpy.ndarray): The image of the whole scene.
    - gaze_points (list of tuples): List of gaze fixation points [(x1, y1), (x2, y2), ...].
    - radius (int): The radius of visual perception.

    Returns:
    - intersaccadic_frames (list of lists of numpy.ndarray): Collection of frames between each pair of saccade points.
    - saccade_frames (list of numpy.ndarray): Frames for each fixation point.
    - point_lists (list of tuples): All the points the visual field passes through.
    """
    file_base, ext = file_name.split('.')
    scene_resolution = (scene_img.shape[0], scene_img.shape[1])
    scene_img = np.array(scene_img)

    intersaccadic_frames = []  # Will contain frames between each pair of points
    saccade_frames = []  # Frames for each fixation point
    point_lists = []  # Points the center of the visual field passes through

    num_points = len(gaze_points)

    for i in range(num_points - 1):
        # Check if gaze points are within the image bounds
        if (gaze_points[i][0] + radius > scene_resolution[0]) or (gaze_points[i][1]+radius > scene_resolution[1]) or (gaze_points[i+1][0] + radius > scene_resolution[0]) or (gaze_points[i+1][1]+radius > scene_resolution[1]):
            raise Exception("Center of gaze not possible" +
                            str(gaze_points) + "res "+str(scene_resolution))
        if (gaze_points[i][0] - radius < 0) or (gaze_points[i][1]-radius < 0) or (gaze_points[i+1][0] - radius < 0) or (gaze_points[i+1][1]-radius < 0):
            raise Exception("Center of gaze not possible" +
                            str(gaze_points) + "res"+str(scene_resolution))

        # Get all points on the line between two gaze points
        rr, cc = line(gaze_points[i][0], gaze_points[i]
                      [1], gaze_points[i+1][0], gaze_points[i+1][1])
        point_list = list(zip(rr, cc))
        point_lists.extend(point_list)

        # Collect frames between two gaze points
        frames_between_points = []
        for (x, y) in point_list:
            temp_frame = scene_img[x - radius:x +
                                   radius, y - radius:y + radius, :]
            frames_between_points.append(temp_frame)

        intersaccadic_frames.append(frames_between_points)

    # Collect saccade frames for each gaze point
    for i, (x, y) in enumerate(gaze_points):
        saccade_frame = scene_img[x - radius:x +
                                  radius, y - radius:y + radius, :]

        saccade_frames.append(saccade_frame)
        output_path = os.path.join(
            dataset_directory, file_base, f"{file_base}_saccade_{str(i)}.{ext}")
        cv2.imwrite(output_path, saccade_frame)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    try:
        output_path = os.path.join(
            dataset_directory, file_base, f"{file_base}.avi")
        video = cv2.VideoWriter(output_path,
                                fourcc, 120, visual_field_resolution)
        for lst in intersaccadic_frames:
            for frame in lst:
                video.write(frame)
        cv2.destroyAllWindows()
        video.release()
    except Exception as inst:
        print(type(inst))
        print(inst.args)
        print(inst)
        return None

    return intersaccadic_frames, saccade_frames, point_lists


def list_prior_generated_images(dataset_directory, num_of_saccades):
    finished_imgs_file_path = os.path.join(
        dataset_directory, 'finished_imgs.txt')
    prior_generated_imgs = []
    if os.path.isfile(finished_imgs_file_path):
        with open(finished_imgs_file_path, 'r') as file:
            # Read all lines from the file
            lines = file.readlines()
            # Process each line to remove leading/trailing whitespace
            prior_generated_imgs = [line.strip() for line in lines]
    for img_name in prior_generated_imgs:
        img_base_name, ext = img_name.split('.')
        img_dir = os.path.join(dataset_directory, img_base_name)
        files_to_check = []
        files_to_check.append(img_name)
        files_to_check.append('points.h5')
        files_to_check.append(os.path.join('v2e', 'dvs-video.avi'))
        files_to_check.append(os.path.join('v2e', 'dvs-video-frame_times.txt'))
        files_to_check.append(os.path.join('v2e', 'events.h5'))
        files_to_check.append(os.path.join('v2e', 'v2e-args.txt'))
        for i in range(num_of_saccades+1):
            files_to_check.append(f"{img_base_name}'_'{str(i)}'.'{ext}")
        for file in files_to_check:
            if not os.path.exists(os.path.join(dataset_directory, img_dir, file)):
                prior_generated_imgs.remove(img_name)
                print(f"Image {img_name} is not fully generated")
                break
    return prior_generated_imgs


def update_finished_imgs(dataset_directory, img_name):
    finished_imgs_file_path = os.path.join(
        dataset_directory, 'finished_imgs.txt')
    with open(finished_imgs_file_path, 'a') as file:
        file.write(img_name + '\n')


def prepare_dataset(input_directory, output_directory, num_of_saccades, resume=True):

    if resume:
        list_finished_imgs = list_prior_generated_images(
            output_directory, num_of_saccades)
    else:
        list_finished_imgs = []
        finished_imgs_file_path = os.path.join(
            output_directory, 'finished_imgs.txt')
        if os.path.exists(finished_imgs_file_path):
            os.remove(finished_imgs_file_path)
            shutil.rmtree(output_directory)

    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif')) and file not in list_finished_imgs:
                try:
                    output_img_folder = os.path.join(
                        output_directory, file.split('.')[0])
                    img_path = os.path.join(root, file)
                    img = Image.open(img_path)
                    resized_image = resize_and_crop_image(
                        img, scene_resolution, output_directory, file)
                    gaze_points = generate_good_points(
                        resized_image, num_of_saccades, radius, scene_resolution)
                    intersaccadic_frames, saccade_frames, point_lists = generate_saccades(
                        resized_image, gaze_points, radius, output_directory, file)
                    h5f = h5py.File(output_img_folder+'/points.h5', 'w')
                    h5f.create_dataset('points', data=np.array(point_lists))
                    h5f.create_dataset('gaze_points', data=gaze_points)
                    process_single_video(output_directory, file.split('.')[0])
                    update_finished_imgs(output_directory, file)

                except Exception as e:
                    print(f"Error processing image {file}: {e}")
                    continue


def process_image(dataset_path, name, max_event_frames_in_saccade):
        try:
            # Check if both required files (events.h5 and points.h5) exist for the current name
            events_path = os.path.join(dataset_path, name, 'v2e/events.h5')
            points_path = os.path.join(dataset_path, name, 'points.h5')

            if os.path.exists(events_path) and os.path.exists(points_path):
                # Load the main image (RGB format)
                img_path = os.path.join(dataset_path, name, f'{name}.JPEG')
                this_img = cv2.cvtColor(
                    cv2.imread(img_path), cv2.COLOR_BGR2RGB)

                # Load saccade images
                this_saccades = []
                for i in range(num_of_saccades+1):
                    saccade_img_path = os.path.join(
                        dataset_path, name, f'{name}_saccade_{i}.JPEG')
                    saccade_img = cv2.cvtColor(cv2.imread(
                        saccade_img_path), cv2.COLOR_BGR2RGB)
                    # Debugging output
                    this_saccades.append(saccade_img)

                # Convert list of saccades to a numpy array and print its shape
                np_this_saccades = np.array(this_saccades)

                # Load points and gaze points from the points.h5 file
                with h5py.File(points_path, 'r') as h5f:
                    this_points = h5f['points'][:]
                    this_gaze_points = h5f['gaze_points'][:]

                # Open events.h5 and load events data
                tbl = tables.open_file(events_path, 'r')
                events = tbl.root.events

                # Initialize variables to store event frames and points
                this_simple_event_frame = np.zeros((h, w))
                this_simple_event_frames = []
                saccade_simple_event_frames = []
                saccade_points = []
                new_points = []
                pointCount = []
                step = step_size
                eventCount = 0
                saccadeCount = 0
                frameCount = 0

                # Iterate through events and group them into saccades
                while saccadeCount < num_of_saccades+1 and eventCount < events.shape[0]:
                    event = events[eventCount]

                    # Check if it's time to move to the next frame based on the step size
                    if event[0] >= step:
                        step += step_size
                        saccade_simple_event_frames.append(
                            this_simple_event_frame)
                        saccade_points.append(this_points[frameCount])

                        # If the current points match the next gaze points, save the saccade data
                        if (this_points[frameCount] == this_gaze_points[saccadeCount + 1]).all():
                            # Convert saccade frames to a numpy array
                            np_saccade_simple_event_frames = np.array(
                                saccade_simple_event_frames)

                            # Pad saccade frames if they are less than max_event_frames_in_saccade
                            if np_saccade_simple_event_frames.shape[0] < max_event_frames_in_saccade:
                                padded_array = np.zeros(
                                    (max_event_frames_in_saccade, np_saccade_simple_event_frames.shape[1],
                                     np_saccade_simple_event_frames.shape[2]))
                                padded_array[:np_saccade_simple_event_frames.shape[0]
                                             ] = np_saccade_simple_event_frames
                                np_saccade_simple_event_frames = padded_array

                            # Append the processed saccade frames and points
                            this_simple_event_frames.append(
                                np_saccade_simple_event_frames)
                            saccade_simple_event_frames = []
                            saccadeCount += 1

                            # Pad saccade points if they are less than max_event_frames_in_saccade
                            while len(saccade_points) < max_event_frames_in_saccade:
                                saccade_points.append((0, 0))

                            new_points.append(np.array(saccade_points))
                            pointCount.append(len(saccade_points))
                            saccade_points = []

                        # Reset frame and move to the next
                        frameCount += 1
                        this_simple_event_frame = np.zeros((h, w))

                    # Update the event frame based on the event polarity
                    if event[3] == 1:
                        this_simple_event_frame[event[2], event[1]] += 1
                    else:
                        this_simple_event_frame[event[2], event[1]] -= 1

                    eventCount += 1

                # Handle any remaining saccade frames that haven't been added yet
                if saccadeCount < num_of_saccades:
                    np_saccade_simple_event_frames = np.array(
                        saccade_simple_event_frames)

                    # Pad the remaining saccade frames if necessary
                    if np_saccade_simple_event_frames.shape[0] < max_event_frames_in_saccade:
                        padded_array = np.zeros(
                            (max_event_frames_in_saccade, np_saccade_simple_event_frames.shape[1],
                             np_saccade_simple_event_frames.shape[2]))
                        padded_array[:np_saccade_simple_event_frames.shape[0]
                                     ] = np_saccade_simple_event_frames
                        np_saccade_simple_event_frames = padded_array

                    # Add the padded frames and points
                    this_simple_event_frames.append(
                        np_saccade_simple_event_frames)
                    saccadeCount += 1

                    while len(saccade_points) < max_event_frames_in_saccade:
                        saccade_points.append((0, 0))

                    new_points.append(np.array(saccade_points))
                    pointCount.append(len(saccade_points))
                tbl.close()
                # Store the final results for this iteration
                this_img= np.array(this_img)
                this_simple_event_frames = np.array(this_simple_event_frames)
                pointCount = np.array(pointCount)
                this_gaze_points = np.array(this_gaze_points)
                new_points = np.array(new_points)
                
                print(f"finsihed processing image {name}")
                return this_img, this_simple_event_frames, pointCount, this_gaze_points, new_points, np_this_saccades
                # Close the events file

        except Exception as e:
            print(f"Error processing image {name}: {e}")
            return    


def process_images_in_parallel(dataset_path, image_names, max_event_frames_in_saccade, num_cores):
    with multiprocessing.Pool(num_cores) as pool:
        results = pool.starmap(process_image, [(dataset_path, name, max_event_frames_in_saccade) for name in image_names])
    return results

def create_hdf5_dataset(dataset_path, hdf5_path, num_of_saccades):
    image_names = [d for d in os.listdir(
        dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]



    max_event_frames_in_saccade = 0
    frames_in_saccade = 0
    frames_in_saccade_lst = []
    for name in image_names:
        try:
            points_path = os.path.join(dataset_path, name, 'points.h5')
            if os.path.exists(points_path):
                with h5py.File(points_path, 'r') as h5f:
                    this_points = h5f['points'][:]
                    this_gaze_points = h5f['gaze_points'][:]
                # check number of points in a single saccade and find max
                for saccade_count in range(num_of_saccades+1):
                    while (True):
                        if (this_points[frames_in_saccade] == this_gaze_points[saccade_count]).all():
                            frames_in_saccade_lst.append(frames_in_saccade)
                            if frames_in_saccade > max_event_frames_in_saccade:
                                max_event_frames_in_saccade = frames_in_saccade
                            frames_in_saccade = 0
                            break

                        else:
                            frames_in_saccade += 1
        except Exception as e:
            print(f"Error processing image {name}: {e}")
            continue


    # add date to hdf5_path and add next number if name exists.
    current_date = datetime.datetime.now().strftime("%d_%m_%y")
    hdf5_path = f"{hdf5_path.split('.')[0]}_{current_date}.hdf5"
    if os.path.exists(hdf5_path):
        i = 1
        while True:
            new_hdf5_path = f"{hdf5_path.split('.')[0]}_{i}.hdf5"
            if not os.path.exists(new_hdf5_path):
                hdf5_path = new_hdf5_path
                break
            i += 1


    with h5py.File(hdf5_path, 'w') as hf:
        # Create empty resizable datasets in HDF5 file
        hf.create_dataset('scenes', shape=(0, scene_h, scene_w, 3),
                          maxshape=(None, scene_h, scene_w, 3), dtype='uint8')
        hf.create_dataset('saccades', shape=(0, num_of_saccades+1, h, w, 3),
                          maxshape=(None,  num_of_saccades+1, h, w, 3), dtype='uint8')
        hf.create_dataset('simple_event_frames', shape=(0, num_of_saccades, max_event_frames_in_saccade, h, w), maxshape=(
            None, num_of_saccades, max_event_frames_in_saccade, h, w), dtype='int16')
        hf.create_dataset('points', shape=(0, num_of_saccades, max_event_frames_in_saccade, 2), maxshape=(
            None, num_of_saccades, max_event_frames_in_saccade, 2), dtype='uint16')
        hf.create_dataset('gaze_points', shape=(0, num_of_saccades+1, 2), maxshape=(
            None, num_of_saccades+1, 2), dtype='uint16')
        hf.create_dataset('num_of_event_frames', shape=(
            0, num_of_saccades), maxshape=(None, num_of_saccades), dtype='uint8')

    #                 if len(this_points[saccade_count])>max_frames_in_saccade:
    #                     max_frames_in_saccade=len(this_points[saccade_count])
    #                 if (this_points[frameCount] == this_gaze_points[saccadeCount + 1]).all():
    #                     if len(this_points[i])>max_frames_in_saccade:
    #                         max_frames_in_saccade=len(this_points[i])
    



    
    
    num_cores = multiprocessing.cpu_count()
    results = process_images_in_parallel(dataset_path, image_names, max_event_frames_in_saccade, num_cores)# Collect all results before writing
    scenes = []
    saccades = []
    simple_event_frames = []
    points = []
    gaze_points = []
    num_of_event_frames = []

    results = process_images_in_parallel(dataset_path, image_names, max_event_frames_in_saccade, num_cores)

    for result in results:
        if result is not None:
            this_img, this_simple_event_frames, pointCount, this_gaze_points, new_points, np_this_saccades = result

            # Collect data in lists
            scenes.append(this_img)
            saccades.append(np_this_saccades)
            simple_event_frames.append(this_simple_event_frames)
            points.append(new_points)
            gaze_points.append(this_gaze_points)
            num_of_event_frames.append(pointCount)

    # Write to HDF5 in one go
    with h5py.File(hdf5_path, 'a') as hf:

        # Convert lists to numpy arrays
        scenes = np.array(scenes)
        saccades = np.array(saccades)
        simple_event_frames = np.array(simple_event_frames)
        points = np.array(points)
        gaze_points = np.array(gaze_points)
        num_of_event_frames = np.array(num_of_event_frames)

        # Resize and append datasets in bulk
        hf['scenes'].resize((hf['scenes'].shape[0] + scenes.shape[0]), axis=0)
        hf['scenes'][-scenes.shape[0]:] = scenes

        hf['saccades'].resize((hf['saccades'].shape[0] + saccades.shape[0]), axis=0)
        hf['saccades'][-saccades.shape[0]:] = saccades

        hf['simple_event_frames'].resize((hf['simple_event_frames'].shape[0] + simple_event_frames.shape[0]), axis=0)
        hf['simple_event_frames'][-simple_event_frames.shape[0]:] = simple_event_frames

        hf['points'].resize((hf['points'].shape[0] + points.shape[0]), axis=0)
        hf['points'][-points.shape[0]:] = points

        hf['gaze_points'].resize((hf['gaze_points'].shape[0] + gaze_points.shape[0]), axis=0)
        hf['gaze_points'][-gaze_points.shape[0]:] = gaze_points

        hf['num_of_event_frames'].resize((hf['num_of_event_frames'].shape[0] + num_of_event_frames.shape[0]), axis=0)
        hf['num_of_event_frames'][-num_of_event_frames.shape[0]:] = num_of_event_frames


def main():
    """
    Main function for data generation.
    args.data_generation_config.py contains the configuration for data generation.
    
    Args:
        --prepate_dataset: If provided, prepares dataset from a folder of images.
        --resume: If provided, resumes from a previous run.
        --create_hdf5: If provided, creates a hdf5 file from a prepared dataset.

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepate_dataset', action='store_true', help="Prepare dataset from a folder of images")
    parser.add_argument('--resume', action='store_true', help="Resume from a previous run")
    parser.add_argument('--create_hdf5',  action='store_true', help="From a prepared dataset, create a hdf5 file")
    args = parser.parse_args()
    if args.prepate_dataset:
        prepare_dataset(input_directory=input_path,
                     output_directory=output_path, num_of_saccades=num_of_saccades, resume=args.resume)
    if args.create_hdf5:
        create_hdf5_dataset(output_path, hdf5_path, num_of_saccades)


if __name__ == "__main__":
    main()
