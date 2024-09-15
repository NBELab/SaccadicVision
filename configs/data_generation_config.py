

num_of_saccades = 5  # Number of points viewed will be num_of_saccades+1
# False indicates that points will be chosen based on GoodFeatruresToTrack
random_saccades = False
radius = 64  # Size of visual field from fixation point. Resolution would be 2*radius,2*radius
visual_field_resolution = (radius*2, radius*2)
h, w = visual_field_resolution
scene_resolution = (200, 200)  # of the whole scene
scene_h, scene_w = scene_resolution
input_path = '/RG/rg-tsur/shyahia/tiny_ImageNet'  # a folder of images of any size
# a folder for all generated output (for model input)
output_path = '/RG/rg-tsur/shyahia/created_dataset'
step_size = (1/120.0) * 1e6
hdf5_path = '/RG/rg-tsur/shyahia/dataset_tiny_2.hdf5'
