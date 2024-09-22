# SaccadicVision

# Image Reconstruction from Retinal Inputs

This README provides instructions on how to use `data_generation.py` and `main.py` to create simulated retinal data and train a network to reconstruct colorful images using these incomplete and color-lacking retinal inputs.

## Prerequisites
Before getting started, ensure that you have the following installed:
- Python 3.x
- Required Python packages (specified in `requirements.txt`)

## Data Generation
To generate simulated retinal data, follow these steps:

1. Specify input and output paths, and any other configurations suchs as resolutions in `configs/data_generation_config.py` 

2. Open a terminal and navigate to the project directory.

3. Run the following command to execute `data_generation.py`:

```bash
python data_generation.py --prepate_dataset --resume --create_hdf5
```

The `--prepate_dataset` argument is used to prepare the dataset from a folder of images. 

The `--resume` argument is optional. If provided, it allows the script to resume from a previous run. This is useful if the dataset creation process was stopped before it was finished, as it will continue from where it left off.

The `--create_hdf5` argument is used to create an HDF5 file from the prepared dataset. this is necessary for the model.

Make sure to provide the necessary arguments according to your requirements.
You can first prepare the dataset, and then seperately create the hdf5 from the dataset folder as well.

   
4. The script will generate the simulated retinal data and save it in the specified output directory, and then create the hdf5 using the dataset generated.

## Training the Network
To train the network for image reconstruction, follow these steps:
1. Set the hdf5 file path and output folder, along with any other configurations for training in `configs/config.py`.
2. Open a terminal and navigate to the project directory.
3. Run the following command to execute `main.py`:
    ```
    python main.py
    ```
4. The script will load the simulated retinal data, train the network, and save the reconstructed images in the specified output directory.

## Additional Notes
- Adjust the parameters in the scripts (`data_generation.py` and `main.py`) according to your requirements.
- Refer to the code comments for more detailed explanations of the implementation.


## License