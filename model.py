# import cv2
# import pandas as pd
# import tables
# import datetime
# import random
# import math
# import scipy.ndimage
# import matplotlib.pyplot as plt
# from tensorflow.python.keras import backend as K
# from skimage.color import rgb2gray, gray2rgb
# from sklearn.model_selection import train_test_split
# from config import data_len1, saccade_h, saccade_w, set_epochs, scene_resolution, radius,mask_rad,batch_size,num_samples, h5_path
# from model_utils import create_circular_mask_opp, rgb2opp, rgb2opp_batch,opp2rgb, norm_image_np,solve_poisson_mat


import os
import numpy as np 
import tensorflow as tf
import datetime
from tensorflow import keras
from keras import layers
from keras.layers import LeakyReLU
import lpips1.lpips.models.lpips_tensorflow as lpips_tf 

from scipy.fftpack import idst, dst
import math
from configs.config import saccade_resolution,num_of_saccades
from model_utils import opp2rgb_tf, denum_matrix,norm_image, solve_poiss_tf





mse_loss = tf.keras.losses.MeanAbsoluteError()
model_dir = '/RG/rg-tsur/shyahia/shyahia-shortcut/ImageReconstructionFromRetinalInputs/lpips1/lpips/models'
vgg_ckpt_fn = os.path.join(model_dir, 'vgg', 'exported')
lin_ckpt_fn = os.path.join(model_dir, 'lin', 'exported')
lpips2 = lpips_tf.learned_perceptual_metric_model(saccade_resolution, vgg_ckpt_fn, lin_ckpt_fn)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mae_loss = tf.keras.losses.MeanAbsoluteError()

def esmr_mse(gt, pred):
    return tf.metrics.mse(gt, pred)

def L_loss(real_image,fake_image):
  ssim_loss = 1-tf.reduce_mean(tf.clip_by_value(tf.image.ssim(real_image, fake_image, max_val=1.0, filter_size=3), 0, 1))
  return mse_loss(real_image,fake_image) +0.25*ssim_loss

def lpips_loss(gt, pred_img, lpips=lpips2):
    return tf.reduce_mean(lpips([255*gt, 255*pred_img]))

@tf.function
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = 0.5*(real_loss + fake_loss)
    return total_loss

@tf.function
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

@tf.function
def ssim_rgb(real_images,generated_images):
   return tf.reduce_mean(tf.clip_by_value(tf.image.ssim(generated_images, real_images, max_val=1.0, filter_size=5), 0, 1))

@tf.function
def ssim_opp2rgb(image1,image2):
  image1_rgb = opp2rgb_tf(image1)
  image2_rgb = opp2rgb_tf(image2)
  return tf.reduce_mean(tf.clip_by_value(tf.image.ssim(image1_rgb,image2_rgb , max_val=1.0, filter_size=5), 0, 1))

@tf.function
def swish(x):
    return x*tf.sigmoid(x)

@tf.function
def mish(x):
    return x*tf.tanh(tf.nn.softplus(x))


def make_generator_model(frames=None, width=saccade_resolution[0], height=saccade_resolution[1], channels=1,out_put_channels=3):


    trailer_input  = keras.Input(shape=(None,frames, width, height, channels)
                    , name='trailer_input')

    input_events = layers.Lambda(lambda x: x[:, :, 3:, :, :, :])(trailer_input)
    
    color_centers = layers.Lambda(lambda x: x[:, :, 0:2, :, :, :])(trailer_input)
    color_centers = tf.transpose(color_centers, [0,1, 5, 3, 4, 2])
    color_centers = layers.Reshape((-1, height, width, 2))(color_centers)
    
    intensity_laplacian_centers = layers.Lambda(lambda x: x[:, :, 2, :, :, :])(trailer_input)
    input_intensity_laplacian_centers = layers.Reshape((-1, height, width, 1))(intensity_laplacian_centers)
    
    #Reconstruct 
    
    input_events= layers.Masking(mask_value=0.0) (input_events)
    first_conv= layers.Conv2D(filters=32, kernel_size=(5, 5)
                       , data_format='channels_last'
                       , activation=LeakyReLU()
                       ,padding='same')(input_events)
    first_conv_normalized= layers.BatchNormalization()(first_conv)
    first_ConvLSTM = layers.TimeDistributed(layers.ConvLSTM2D(filters=32, kernel_size=(5, 5)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation=LeakyReLU()
                       , padding='same', return_sequences=True))(first_conv_normalized)
    first_ConvLSTM_normalized= layers.BatchNormalization()(first_ConvLSTM)

    first_block_conv1=  layers.Conv2D(filters=32, kernel_size=(5, 5)
                       , data_format='channels_last'
                       , activation=LeakyReLU()
                       ,padding='same')(first_ConvLSTM_normalized)
    first_block_conv1_normalized= layers.BatchNormalization()(first_block_conv1)
    first_block_conv2=   layers.Conv2D(filters=32, kernel_size=(5, 5)
                       , data_format='channels_last'
                       , activation= None
                       ,padding='same')(first_block_conv1_normalized) #no activation yet
    first_block_conv2_normalized= layers.BatchNormalization()(first_block_conv2)


    addition1 = layers.add([first_ConvLSTM_normalized, first_block_conv2_normalized])  # Element-wise addition
    addition1_relu = layers.Activation(LeakyReLU())(addition1)
    addition1_normalized= layers.BatchNormalization()(addition1_relu)

    
    
    second_ConvLSTM = layers.TimeDistributed(layers.ConvLSTM2D(filters=32, kernel_size=(5, 5)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation=LeakyReLU()
                       , padding='same', return_sequences=True))(addition1_normalized)
    second_ConvLSTM_normalized= layers.BatchNormalization()(second_ConvLSTM)

    second_block_conv1=  layers.Conv2D(filters=32, kernel_size=(5, 5)
                       , data_format='channels_last'
                       , activation=LeakyReLU()
                       ,padding='same')(second_ConvLSTM_normalized)

    second_block_conv1_normalized= layers.BatchNormalization()(second_block_conv1)
    second_block_conv2=   layers.Conv2D(filters=32, kernel_size=(5, 5)
                       , data_format='channels_last'
                       , activation=None
                       ,padding='same')(second_block_conv1_normalized) #no activation yet
    second_block_conv2_normalized= layers.BatchNormalization()(second_block_conv2)


    addition2 = layers.add([second_ConvLSTM_normalized, second_block_conv2_normalized])  # Element-wise addition
    addition2_relu = layers.Activation(LeakyReLU())(addition2)
    addition2_normalized= layers.BatchNormalization()(addition2_relu)


    

    last_ConvLSTM= layers.TimeDistributed(layers.ConvLSTM2D(filters=1, kernel_size=(1, 1)
                        , data_format='channels_last'
                        , activation= 'tanh'
                        , padding='same', return_sequences=False))(addition2_normalized)
    last_ConvLSTM_normalized= layers.BatchNormalization()(last_ConvLSTM)
    rec_intensity_from_events = layers.Reshape((-1, height, width,1))(last_ConvLSTM_normalized) #Recreated Laplacian of intensity
    
    
    '''
    intensity_rec= layers.Concatenate(axis=-1)([rec_laplacian_from_events, intensity_rec])
    
    

    intensity_rec = layers.Conv2D(filters=32, kernel_size=(5, 5)
                       , data_format='channels_last'
                       , activation=LeakyReLU()
                       ,padding='same')(intensity_rec)
                       
    intensity_rec= layers.BatchNormalization() (intensity_rec)
    intensity_rec = layers.ConvLSTM2D(filters=64, kernel_size=(5, 5)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation=LeakyReLU()
                       , padding='same', return_sequences=True)(intensity_rec)
    intensity_rec= layers.BatchNormalization()(intensity_rec)
    
    intensity_rec = layers.Conv2D(filters=64, kernel_size=(5, 5)
                       , data_format='channels_last'
                       , activation=LeakyReLU()
                       ,padding='same')(intensity_rec)
    intensity_rec= layers.BatchNormalization() (intensity_rec)
    intensity_rec = layers.ConvLSTM2D(filters=1, kernel_size=(1, 1)
                        , data_format='channels_last'
                        , activation= None
                       , padding='same', return_sequences=True)(intensity_rec)
    '''

    
    
    input_intensity_laplacian_centers=layers.Reshape((-1,height,width))(input_intensity_laplacian_centers)
    intensity_rec = layers.TimeDistributed(ReconstructionLayer(height,width))(input_intensity_laplacian_centers)
    intensity_rec=layers.Reshape((-1,height,width,1))(intensity_rec)
    
    
    
    intensity_rec= layers.Concatenate(axis=-1)([rec_intensity_from_events, intensity_rec])
    
    

    intensity_rec = layers.Conv2D(filters=32, kernel_size=(5, 5)
                       , data_format='channels_last'
                       , activation=LeakyReLU()
                       ,padding='same')(intensity_rec)
                       
    intensity_rec= layers.BatchNormalization() (intensity_rec)
    intensity_rec = layers.ConvLSTM2D(filters=64, kernel_size=(5, 5)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation=LeakyReLU()
                       , padding='same', return_sequences=True)(intensity_rec)
    intensity_rec= layers.BatchNormalization()(intensity_rec)
    
    intensity_rec = layers.Conv2D(filters=64, kernel_size=(5, 5)
                       , data_format='channels_last'
                       , activation=LeakyReLU()
                       ,padding='same')(intensity_rec)
    intensity_rec= layers.BatchNormalization() (intensity_rec)
    intensity_rec = layers.ConvLSTM2D(filters=1, kernel_size=(1, 1)
                        , data_format='channels_last'
                        , activation= None
                       , padding='same', return_sequences=True)(intensity_rec)
    
    
  
    
    intensity_rec_first=layers.Reshape((-1,height, width,1 ))(intensity_rec)


    coloring_input= layers.Concatenate(-1)([color_centers, intensity_rec_first]) 
    coloring_input= layers.Reshape((-1,height, width,3 ))(coloring_input)


    # Entry block Colorization
    x = layers.Conv2D(128, 3, padding="same")(coloring_input)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    conv1 = layers.Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(x)
    conv1 =layers.LeakyReLU()(conv1)
    conv1 = layers.Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 =layers.LeakyReLU()(conv1)

    pool1 = layers.Conv2D(256, 2,strides=2,activation = 'relu',  padding = 'same', kernel_initializer = 'he_normal')(conv1)
    conv2 = layers.Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool1) # 64,64
    conv2 =layers.LeakyReLU()(conv2)
    conv2 = layers.Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv2) # 64,64
    conv2 = layers.BatchNormalization()(conv2)
    conv2 =layers.LeakyReLU()(conv2)

    pool2 = layers.Conv2D(256, 2,strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    conv3 = layers.Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(pool2) #32,32
    conv3 =layers.LeakyReLU()(conv3)
    conv3 = layers.Conv2D(512, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv3) #32,32
    conv3 = layers.BatchNormalization()(conv3)
    conv3 =layers.LeakyReLU()(conv3)

    up4 = layers.TimeDistributed(layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', use_bias=False))(conv3)
    resize4 = layers.TimeDistributed(tf.keras.layers.experimental.preprocessing.Resizing(64,64, interpolation="bilinear"))(up4) #(64,64)
    merge5 = layers.add([conv2,resize4])
    conv5 = layers.Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge5)
    conv5 =layers.LeakyReLU()(conv5)
    conv5 = layers.Conv2D(256, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 =layers.LeakyReLU()(conv5)

    up6 = layers.TimeDistributed(layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', use_bias=False))(conv5)
    merge6 = layers.add([conv1,up6])
    conv6 = layers.Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 =layers.LeakyReLU()(conv6)
    conv6 = layers.Conv2D(128, 3, activation = None, padding = 'same', kernel_initializer = 'he_normal')(conv6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 =layers.LeakyReLU()(conv6)
    

    x = layers.Conv2D(out_put_channels, 3, activation=None, padding="same")(conv6)

    opp_rec = layers.TimeDistributed(tf.keras.layers.experimental.preprocessing.Resizing(height, width, interpolation="bilinear"))(x) # 3- channels  128,128
    unet_intensity_rec =  layers.Lambda(lambda x: x[:,:,:,:,2])(opp_rec) # intensity
    unet_intensity_rec = layers.Reshape((-1,height,width,1))(unet_intensity_rec)
    

    intensity_rec=layers.Concatenate(axis=-1)([intensity_rec, unet_intensity_rec])
    intensity_rec = layers.Conv2D(64, 3, strides=1, padding="same")(intensity_rec)
    intensity_rec = layers.LeakyReLU()(intensity_rec)
    intensity_rec= layers.BatchNormalization() (intensity_rec)
    intensity_rec = layers.Conv2D(128, 3, strides=1, padding="same")(intensity_rec)
    intensity_rec = layers.LeakyReLU()(intensity_rec)
    intensity_rec= layers.BatchNormalization() (intensity_rec)
    intensity_rec = layers.Conv2D(1, 1, strides=1, padding="same", activation=None)(intensity_rec)
    intensity_rec= layers.Reshape((-1,height, width, 1))(intensity_rec)
    
    
    intensity_rec = layers.Conv2D(32, 3, strides=1, padding="same",)(intensity_rec)
    intensity_rec = layers.LeakyReLU()(intensity_rec)
    intensity_rec= layers.BatchNormalization() (intensity_rec)
    intensity_rec = layers.ConvLSTM2D(filters=32, kernel_size=(5, 5)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation=LeakyReLU()
                       , padding='same', return_sequences=True)(intensity_rec)
    intensity_rec= layers.BatchNormalization()(intensity_rec)
    
    intensity_rec = layers.Conv2D(64, 3, strides=1, padding="same",)(intensity_rec)
    intensity_rec = layers.LeakyReLU()(intensity_rec)
    intensity_rec= layers.BatchNormalization() (intensity_rec)
    intensity_rec = layers.ConvLSTM2D(filters=1, kernel_size=(1, 1)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation=LeakyReLU()
                       , padding='same', return_sequences=True)(intensity_rec)
    intensity_rec= layers.Reshape((-1,height, width, 1))(intensity_rec)
  
    
     
    opp_color_channel1 =  layers.Lambda(lambda x: x[:,:,:,:,0])(opp_rec) # color channels
    opp_color_channel1= layers.Reshape((-1,height,width,1))(opp_color_channel1)
    
    color_rec1 = layers.Conv2D(32, 3, strides=1, padding="same",)(opp_color_channel1)
    color_rec1 = layers.LeakyReLU()(color_rec1)
    color_rec1= layers.BatchNormalization() (color_rec1)
    color_rec1 = layers.ConvLSTM2D(filters=32, kernel_size=(5, 5)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation=LeakyReLU()
                       , padding='same', return_sequences=True)(color_rec1)
    color_rec1= layers.BatchNormalization()(color_rec1)
    
    color_rec1 = layers.Conv2D(64, 3, strides=1, padding="same",)(color_rec1)
    color_rec1 = layers.LeakyReLU()(color_rec1)
    color_rec1= layers.BatchNormalization() (color_rec1)
    color_rec1 = layers.ConvLSTM2D(filters=1, kernel_size=(1, 1)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation=LeakyReLU()
                       , padding='same', return_sequences=True)(color_rec1)
    
    color_rec1= layers.Reshape((-1,height, width, 1))(color_rec1)
    
    
    
    opp_color_channel2 =  layers.Lambda(lambda x: x[:,:,:,:,1])(opp_rec) # color channels
    opp_color_channel2= layers.Reshape((-1,height,width,1))(opp_color_channel2)
    
    color_rec2 = layers.Conv2D(32, 3, strides=1, padding="same",)(opp_color_channel2)
    color_rec2 = layers.LeakyReLU()(color_rec2)
    color_rec2= layers.BatchNormalization() (color_rec2)
    color_rec2 = layers.ConvLSTM2D(filters=32, kernel_size=(5, 5)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation=LeakyReLU()
                       , padding='same', return_sequences=True)(color_rec2)
    color_rec2= layers.BatchNormalization()(color_rec2)
    
    color_rec2 = layers.Conv2D(64, 3, strides=1, padding="same",)(color_rec2)
    color_rec2 = layers.LeakyReLU()(color_rec2)
    color_rec2= layers.BatchNormalization() (color_rec2)
    color_rec2 = layers.ConvLSTM2D(filters=1, kernel_size=(1, 1)
                       , data_format='channels_last'
                       , recurrent_activation='hard_sigmoid'
                       , activation=LeakyReLU()
                       , padding='same', return_sequences=True)(color_rec2)
  
    
    color_rec= layers.Concatenate(-1)([color_rec1, color_rec2]) 
    color_rec= layers.Reshape((-1,height, width, 2))(color_rec)
    

    model = keras.Model(inputs=trailer_input, outputs=[rec_intensity_from_events ,intensity_rec_first, intensity_rec,color_rec ], name='Model')


    return model


def make_discriminator_model(img_size, out_put_channels=3):
  """
    Creates a discriminator model for a GAN.

    Args:
    img_size (tuple): Size of the input image (height, width).
    out_put_channels (int): Number of channels in the input image (e.g., 3 for RGB images).

    Returns:
    model (keras.Model): The discriminator model.
  """
  inputs = keras.Input(shape=img_size + (out_put_channels,))
  x = layers.Conv2D(64, 4,strides=2, padding="same")(inputs)
  x =layers.LeakyReLU(alpha=0.2)(x)
  x = layers.Conv2D(128, 4,strides=2, padding="same",use_bias = False)(x)
  x = layers.BatchNormalization()(x)
  x =layers.LeakyReLU(alpha=0.2)(x)
  x = layers.Conv2D(256, 4,strides=2, padding="same",use_bias = False)(x)
  x = layers.BatchNormalization()(x)
  x =layers.LeakyReLU(alpha=0.2)(x)
  outputs = layers.Conv2D(1, 4,strides=2, padding="same",use_bias = False)(x)
  model = keras.Model(inputs, outputs)
  return model
  
  
  
class ConditionalGAN(keras.Model):

    def __init__(self, discriminator, generator):
        """
        Initialize the Conditional GAN with a discriminator and a generator.

        Args:
        discriminator (tf.keras.Model): The discriminator model.
        generator (tf.keras.Model): The generator model.
        """  
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.ssim_tracker = keras.metrics.Mean(name="ssim")
        self.lpips_tracker = keras.metrics.Mean(name="lpips")
        self.L_loss_tracker = keras.metrics.Mean(name="L_loss")
        self.events_mae_tracker = keras.metrics.Mean(name="event_mae")
        self.first_int_mae_tracker = keras.metrics.Mean(name="first_int_mae")
    def call(self, inputs):
      """
      Forward pass through the generator.

      Args:
      inputs (tf.Tensor): Input tensor for the generator.

      Returns:
      tf.Tensor: Output tensor from the generator.
      """
      x = inputs
      x = self.generator(x)
      return x
    @property
    def metrics(self):
        """
        List of metrics to monitor during training.

        Returns:
        list: List of metric trackers.
        """
        return [self.gen_loss_tracker, self.disc_loss_tracker,self.ssim_tracker]

    def compile(self, d_optimizer, g_optimizer,gen_loss_fn,disc_loss_fn, fn_ssim,fn_ssim_rgb,lpips_loss):
        """
        Compile the GAN model with optimizers and loss functions.

        Args:
        d_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the discriminator.
        g_optimizer (tf.keras.optimizers.Optimizer): Optimizer for the generator.
        gen_loss_fn (callable): Function to calculate generator loss.
        disc_loss_fn (callable): Function to calculate discriminator loss.
        fn_ssim (callable): Function to calculate SSIM.
        fn_ssim_rgb (callable): Function to convert images to RGB for SSIM calculation.
        lpips_loss (callable): Function to calculate LPIPS loss.
        """
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.fn_ssim = fn_ssim
        self.mae =  tf.keras.losses.MeanAbsoluteError()
        self.ssim_rgb = fn_ssim_rgb
        self.lpips_loss = lpips_loss

    def train_step(self, data):
        """
        Perform one training step.

        Args:
        data (tuple): Tuple of input data and real images.

        Returns:
        dict: Dictionary of loss metrics and tracked values.
        """
        input_data,real_images = data
        real_images= tf.cast(real_images, dtype=tf.float32)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
          L_events_lst,L_lst,L_rec_lst,ab_lst = self.generator(input_data, training=True)
          L_real_rec = norm_image(solve_poiss_tf((real_images[:,2,:,:])))
          real_images_rec = tf.concat([real_images[:,0:2,:,:],tf.expand_dims(L_real_rec,1)],1)
          real_images_rec=  tf.transpose(real_images_rec, [0, 2, 3, 1])
          real_images_rec = tf.reshape(real_images_rec, (-1, saccade_resolution[0], saccade_resolution[1], 3))
          real_images=  tf.transpose(real_images, [0, 2, 3, 1])
          real_images_rec_rgb= opp2rgb_tf(real_images_rec)
            

          lpips_loss_lst=[]
          L_loss_lst=[]
          ab_loss_lst=[]
          ssim_loss_lst=[]
          gen_loss_lst=[]
          disc_loss_lst=[]
          events_mae_lst=[]
          first_int_mae_lst=[]
          
          for j in range(1, num_of_saccades+1):
            L_rec= L_rec_lst[:,j,:,:,:]
            L= L_lst[:,j,:,:]
            ab=ab_lst[:,j,:,:,:]
            L_events= L_events_lst[:,j,:,:,:]
            generated_images_rec = tf.concat([ab,(L_rec)],-1)
            generated_images_rec= tf.reshape(generated_images_rec, (-1, saccade_resolution[0], saccade_resolution[1], 3))
            real_output = self.discriminator(real_images_rec, training=True)
            fake_output = self.discriminator(generated_images_rec, training=True)
            generated_images_rec_rgb= opp2rgb_tf(generated_images_rec)   
            
            
            lpips_loss=self.lpips_loss(tf.clip_by_value(opp2rgb_tf(real_images_rec),0,1),tf.clip_by_value(opp2rgb_tf(generated_images_rec),0,1))
            lpips_loss_lst.append(lpips_loss)
            
            events_loss= self.mae(tf.expand_dims(L_real_rec,-1),L_events)
            events_mae_lst.append(100*events_loss)
            
            
            first_int_mae=100*self.mae(L, tf.expand_dims(L_real_rec,-1))
            first_int_mae_lst.append(first_int_mae) 
            
            
            loss_L=100*(first_int_mae+events_loss+self.mae(L_rec,tf.expand_dims(L_real_rec,-1))
                       +0.25*(1-tf.reduce_mean(tf.clip_by_value(tf.image.ssim(tf.expand_dims(L_real_rec,-1), L_rec, max_val=1.0, filter_size=5), 0, 1))))
                       
            L_loss_lst.append(loss_L)
            
            loss_ab=150*self.mae(real_images[:,:,:,0:2],ab)
            ab_loss_lst.append(loss_ab)
            
            
            ssim = self.fn_ssim(tf.clip_by_value(real_images_rec_rgb,0,1),tf.clip_by_value(generated_images_rec_rgb,0,1))
            ssim_loss=100*(1-ssim)
            ssim_loss_lst.append(ssim_loss)
            
            gen_loss_lst.append(1*self.gen_loss_fn(fake_output) +loss_L+loss_ab+ssim_loss +50*lpips_loss)
            disc_loss_lst.append(0.1*self.disc_loss_fn(real_output, fake_output))
            ssim_rgb = ssim
          
          
          first_int_mae= sum(first_int_mae_lst)/len(first_int_mae_lst)
          lpips_loss =sum(lpips_loss_lst) / len(lpips_loss_lst)
          ssim_loss =sum(ssim_loss_lst) / len(ssim_loss_lst)
          gen_loss= sum(gen_loss_lst) / len(gen_loss_lst)
          loss_L=sum(L_loss_lst) / len(L_loss_lst)
          loss_ab= sum(ab_loss_lst) / len(ab_loss_lst)
          disc_loss= sum(disc_loss_lst) / len(disc_loss_lst)
          
          
          events_mae= sum(events_mae_lst)/len(events_mae_lst)
          

          gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
          gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
          self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
          self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
          # Monitor loss.
          self.gen_loss_tracker.update_state(gen_loss)
          self.disc_loss_tracker.update_state(disc_loss)
          self.ssim_tracker.update_state(ssim_rgb)
          self.lpips_tracker.update_state(lpips_loss)
          
          
          
          self.first_int_mae_tracker.update_state(first_int_mae)
          self.L_loss_tracker.update_state(loss_L)
          self.events_mae_tracker.update_state(events_mae)
          
          
          
          return {
            "g_loss": self.gen_loss_tracker.result(),
            #"d_loss": self.disc_loss_tracker.result(),
            "L_loss": self.L_loss_tracker.result(),
            "events_mae": self.events_mae_tracker.result(),
            "first_int_mae": self.first_int_mae_tracker.result(),
            "ssim":   self.ssim_tracker.result(),
            "lpips":  self.lpips_tracker.result(),
          }
          
     

class ReconstructionLayer(tf.keras.layers.Layer):
  """
  A custom Keras layer to reconstruct an intensity image from a Laplacian image using DST and IDST.

  Attributes:
      h (int): Height of the input image.
      w (int): Width of the input image.
  """
  def __init__(self,h,w):
    """
    Initialize the ReconstructionLayer.

    Args:
        h (int): Height of the input image.
        w (int): Width of the input image.
    """
    super(ReconstructionLayer, self).__init__()
    self.h = h
    self.w = w

  def build(self, input_shape):
    """
    Build the layer by creating and initializing necessary matrices.

    Args:
        input_shape (tf.TensorShape): Shape of the input tensor.
    """
    denum_mat = denum_matrix(self.h,self.w)
    denum_mat_ei = np.reshape(1/denum_mat, (self.h, self.w))
    dst_mat_100 = dst(np.eye(self.h), type=2, axis=0)
    dst_mat_70 = dst(np.eye(self.h), type=2, axis=0)
    idst_mat_100 = idst(np.eye(self.h), type=2, axis=0)
    idst_mat_70 = idst(np.eye(self.h), type=2, axis=0)
    self.c_denum_mat_ei = tf.constant(denum_mat_ei, dtype=tf.float32)
    self.c_dst_mat_100 = tf.constant(dst_mat_100, dtype=tf.float32)
    self.c_dst_mat_70 = tf.constant(dst_mat_70, dtype=tf.float32)
    self.c_idst_mat_100 = tf.constant(idst_mat_100, dtype=tf.float32)
    self.c_idst_mat_70 = tf.constant(idst_mat_70, dtype=tf.float32)
    
    
  def compute_output_shape(self, input_shape):
  # The output shape is the same as the input shape
    return input_shape

  def call(self, inputs):
    """
    Perform the reconstruction of the image from the Laplacian image.

    Args:
      inputs (tf.Tensor): The Laplacian image (input).

    Returns:
      tf.Tensor: The reconstructed intensity image.
    """
    grad = inputs
    z = tf.transpose(tf.matmul(self.c_dst_mat_100, tf.transpose(tf.matmul(self.c_dst_mat_70, grad), [0, 2, 1])), [0, 2, 1])
    d = self.c_denum_mat_ei*z
    res = tf.transpose(tf.matmul(self.c_idst_mat_100, tf.transpose(tf.matmul(self.c_idst_mat_70, d), [0, 2, 1])), [0, 2, 1])
    return norm_image(res)