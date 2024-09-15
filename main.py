import datetime
import os
import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf
import datetime
from tensorflow import keras
import random
from configs.config import h5_path, saccade_resolution, scene_resolution,batch_size,num_of_saccades,set_epochs, save_models_path
from configs.config import data_len1, lr
from data_loader import DataGenerator
from model_utils import norm_image, solve_poiss_tf, opp2rgb_tf
from model_callbacks import EarlyStoppingWMinEpoch, RelativeReduceLROnPlateau
from model import ConditionalGAN, ssim_opp2rgb, ssim_rgb,lpips_loss,discriminator_loss,make_discriminator_model,make_generator_model,generator_loss, swish, mish



random.seed(0)
print(tf. __version__)
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", len(gpus))
print(tf.sysconfig.get_build_info()["cuda_version"])

os.makedirs(save_models_path, exist_ok=True)
current_date = datetime.datetime.now().strftime("%d%m%y")
model_name_prefix = "generator_net2_LSTM_Input"+current_date
number = 1
existing_files = os.listdir(save_models_path)
existing_numbers = [
int(filename[len(model_name_prefix)+1:])
for filename in existing_files
if filename.startswith(model_name_prefix)
]
if existing_numbers:
  number = max(existing_numbers) + 1


model_name = f"{model_name_prefix}_{number}"


log_dir = save_models_path+"logs/" + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
os.makedirs(log_dir, exist_ok=True)
best_model_file = 'best_model_gan.h5'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='epoch', profile_batch=0)
lr_callback = RelativeReduceLROnPlateau(monitor='g_loss', factor=0.8, patience=6, verbose=1, alpha=0.005,
                                        cooldown=0, min_lr=2e-6)
mc = tf.keras.callbacks.ModelCheckpoint("training_checkpoints/cp.ckpt", save_best_only=True, save_weights_only=False,monitor='ssim',mode='max')
early_stopping = EarlyStoppingWMinEpoch(monitor='ssim', min_delta=0.005, mode='max', patience=20,  earliest_epoch=150)
activation_type=4
activation_name = ['lerelu', 'relu', 'tanh', 'swish','mish'][activation_type]
activation = [tf.nn.leaky_relu, tf.nn.relu, tf.nn.tanh, swish, mish][activation_type]




inds1 = np.arange(data_len1)
np.random.seed(0)
np.random.shuffle(inds1)
num_val1 = max(1, round(data_len1*0.15))
num_test1 = num_val1
num_train1 = int(data_len1)-num_val1-num_test1


train = DataGenerator(h5_path,inds1[:int(num_train1)],h=scene_resolution[0],w=scene_resolution[1], batch_size= batch_size)
val = DataGenerator(h5_path,inds1[int(num_train1):int(num_train1)+int(num_val1)],h=scene_resolution[0],w=scene_resolution[1],batch_size= batch_size)
test = DataGenerator(h5_path,inds1[-int(num_test1):],h=scene_resolution[0],w=scene_resolution[1],batch_size= batch_size)
     

generator = make_generator_model()
discriminator = make_discriminator_model((saccade_resolution[0],saccade_resolution[1]), 3)

model_path=save_models_path+'/'+model_name
os.makedirs(model_path, exist_ok=True)
keras.utils.plot_model(generator, to_file=model_path+"/generator.png", show_shapes=True)
keras.utils.plot_model(discriminator, to_file=model_path+"/discriminator.png", show_shapes=True)

generator_optimizer = tf.keras.optimizers.Adam(lr)
discriminator_optimizer = tf.keras.optimizers.Adam(lr)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


directory_path = model_path + '/results/'

# Check if the directory exists
if not os.path.exists(directory_path):
    # If it doesn't exist, create it
    os.makedirs(directory_path)

generator.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
discriminator.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
cond_gan = ConditionalGAN(
    discriminator=discriminator, generator=generator)

cond_gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=lr),
    g_optimizer=keras.optimizers.Adam(learning_rate=lr),
    gen_loss_fn=generator_loss,
    disc_loss_fn=discriminator_loss,
    fn_ssim=ssim_opp2rgb,
    fn_ssim_rgb=ssim_rgb,
    lpips_loss=lpips_loss
)
cond_gan.build((None,None,None, saccade_resolution[0], saccade_resolution[1]))
cond_gan.fit(train, epochs=(set_epochs),
             callbacks = [ tensorboard_callback, lr_callback,  early_stopping])
  
cond_gan.generator.save(model_name)

    

ssim_intensity=[]
ssim_pred=[]
input_ssim=[]

n= num_test1//batch_size
for i in range(n):
  batch= test[i]
 
  real=tf.cast(batch[1], tf.float32)
  real_intensity_lst= norm_image(solve_poiss_tf((real[:,2,:,:])))
  
  lap_events_lst, laplacian_rec_lst, int_rec_lst, color_rec_lst, = cond_gan.predict(batch[0])
  lap_events_lst=lap_events_lst[:,num_of_saccades,:,:,:]
  laplacian_rec_lst= laplacian_rec_lst[:,num_of_saccades]
  int_rec_lst=int_rec_lst[:,num_of_saccades]
  color_rec_lst=color_rec_lst[:,num_of_saccades]
  
  
  
  rec_events_lst= norm_image(solve_poiss_tf((lap_events_lst[:,:,:,0])))
  
  
  
  
  real_intensity_lst= tf.expand_dims(real_intensity_lst, axis=1)
  real_color_lst= real[:,0:2,:,:]
  opp_real_lst= tf.concat([real_color_lst, real_intensity_lst], 1)
  opp_real_lst=tf.transpose(opp_real_lst, [0,2,3,1])
  opp_rec_lst= tf.concat([color_rec_lst, int_rec_lst], -1)

  for j in range(batch_size):

    rgb_pred= opp2rgb_tf(opp_rec_lst[j])
    rgb_real= opp2rgb_tf(opp_real_lst[j])
    ssim_intensity.append(tf.reduce_mean(tf.image.ssim(tf.expand_dims(opp_rec_lst[j,:,:,2],-1),tf.expand_dims(opp_real_lst[j,:,:,2],-1), 1, filter_size=5)).numpy())
    
    '''
    input_rgb= opp2rgb_tf(input_centers[j])
    input_ssim.append(tf.reduce_mean(tf.image.ssim(rgb_real, input_rgb, 1, filter_size=5)).numpy())
    '''
    ssim_pred.append(tf.reduce_mean(tf.image.ssim(rgb_pred, rgb_real, 1, filter_size=5)).numpy())
    filename =model_path+'/results/img'+str(i)+str(j)+'pred.jpg'
    im=norm_image(rgb_pred).numpy().astype(np.float32)
    plt.imsave(filename, im)
    im=norm_image(rgb_real).numpy().astype(np.float32)
    filename =model_path+'/results/img'+str(i)+str(j)+'real.jpg'
    plt.imsave(filename, im)
    '''
    input_rgb= norm_image(input_rgb).numpy().astype(np.float32)
    filename =model_path+'/results/img'+str(i)+str(j)+'input.jpg'
    plt.imsave(filename, input_rgb)
    '''
    rec_events= norm_image(rec_events_lst[j]).numpy().astype(np.float32)
    filename =model_path+'/results/img'+str(i)+str(j)+'event_rec.jpg'
    plt.imsave(filename, rec_events , cmap='gray')
    
    
    lap_events= norm_image(lap_events_lst[j,:,:,0]).numpy().astype(np.float32)
    filename =model_path+'/results/img'+str(i)+str(j)+'event_lap.jpg'
    plt.imsave(filename, lap_events , cmap='gray')




mean_intensity= np.mean(np.array(ssim_intensity))
mean_pred = np.mean(np.array(ssim_pred))
mean_input= np.mean(np.array(input_ssim))

lines = [
    'Test mean color SSIM:',
    str(mean_pred),
    '',
    'Test mean intensity only SSIM:',
    str(mean_intensity),
    ''
]



lines2 = [
    '   '
]




# Open the file in write mode ('w')
path= model_path+ '/info.txt'
lines=lines+lines2
with open(path, 'w') as file:
    # Write each line from the 'lines' list to the file
    for line in lines:
        file.write(line + '\n')


lines2 = [
    'Train mean color SSIM:',
    str(mean_pred),
    ''
]




# Open the file in write mode ('w')
path= model_path+ '/info.txt'
lines=lines+lines2
with open(path, 'w') as file:
    # Write each line from the 'lines' list to the file
    for line in lines:
        file.write(line + '\n')
