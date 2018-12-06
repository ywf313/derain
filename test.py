import tensorflow as tf
import numpy as np
from glob import glob
from net import *
import os
from PIL import Image
import skimage.measure
import matplotlib.pyplot as plt



def load_images(filelist):
    data = []
    for img in filelist:
        im = plt.imread(img)
        data.append(im.reshape(1, im.shape[0], im.shape[1],3).astype('float32')/255.) #jpg
#        data.append(im.reshape(1, im.shape[0], im.shape[1],3))                       #png
    return data

ckpt_dir = './checkpoint'
sample_dir = './data/result'

#####################      synthetic    ###########################################

test_files = sorted(glob('./data/synthetic/image/*.jpg'))
label_files = sorted(glob('./data/synthetic/label/*.jpg'))
test_data = load_images(test_files)
label_data = load_images(label_files)

sess = tf.Session()
if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
        
def load(checkpoint_dir):
    print("[*]Reading checkpoint...")

    model_dir = 'train_%s_%s'%(32, 64)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name=os.path.basename(ckpt.model_checkpoint_path)
        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False

    
X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='input')
Y = inference(X,is_training=False)

saver = tf.train.Saver()
if load(ckpt_dir):
    print('[*] Load SUCCESS')
else:
    print('[!] Load failed')
  
def forward(image):
    X_test = tf.placeholder(tf.float32, shape=image.shape, name='derain_test')
    Y_test = inference(X_test,reuse=True,is_training=False)
    return sess.run(Y_test, feed_dict={X_test:image})


ssim_sum = 0
psnr_sum = 0
print("-------Test  image-------")
for idx in range(len(test_files)):
    rainy_image = test_data[idx]
    clean = label_data[idx]
    predicted_image,_,_ = forward(rainy_image)   
    predicted_image = np.squeeze(predicted_image)
    clean = np.squeeze(clean)
    predicted_image = np.clip(255*predicted_image, 0, 255).astype('uint8')
    clean = np.clip(255*clean, 0, 255).astype('uint8')
    ssim = skimage.measure.compare_ssim(predicted_image, clean, gaussian_weights=True,sigma=1.5,use_sample_covariance=False,multichannel=True)
    psnr = skimage.measure.compare_psnr(predicted_image, clean)   
    img = Image.fromarray(predicted_image.astype('uint8'))
    img.save('{}/%d_derain.png'.format(sample_dir)%(idx+1),'png')
    print("SSIM = %.4f"  %ssim)
    print("PSNR = %.2f"  %psnr)
    ssim_sum += ssim
    psnr_sum += psnr
    
avg_ssim = ssim_sum/len(test_files)
avg_psnr = psnr_sum/len(test_files)
print("---- Average SSIM = %.4f----" % avg_ssim)
print("---- Average PSNR = %.2f----" % avg_psnr)  
    
    
#####################      real-world    ###########################################

#test_files = sorted(glob('./data/real/*.jpg'))
#test_data = load_images(test_files)
#
#sess = tf.Session()
#if not os.path.exists(sample_dir):
#        os.makedirs(sample_dir)
#        
#def load(checkpoint_dir):
#    print("[*]Reading checkpoint...")
#
#    model_dir = 'train_%s_%s'%(32, 64)
#    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)
#    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
#    if ckpt and ckpt.model_checkpoint_path:
#        ckpt_name=os.path.basename(ckpt.model_checkpoint_path)
#        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
#        return True
#    else:
#        return False
#   
#X = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='input')
#Y = inference(X,is_training=False)
#
#saver = tf.train.Saver()
#if load(ckpt_dir):
#    print('[*] Load SUCCESS')
#else:
#    print('[!] Load failed')
#  
#def forward(image):
#    X_test = tf.placeholder(tf.float32, shape=image.shape, name='derain_test')
#    Y_test = inference(X_test,reuse=True,is_training=False)
#    return sess.run(Y_test, feed_dict={X_test:image})
#
#print("-------Test  image-------")
#for idx in range(len(test_files)):
#    rainy_image = test_data[idx]    
#    predicted_image,_,_ = forward(rainy_image)   
#    predicted_image = np.squeeze(predicted_image)    
#    predicted_image = np.clip(255*predicted_image, 0, 255).astype('uint8')   
#    img = Image.fromarray(predicted_image.astype('uint8'))
#    img.save('{}/%d_derain.png'.format(sample_dir)%(idx+1),'png')
    