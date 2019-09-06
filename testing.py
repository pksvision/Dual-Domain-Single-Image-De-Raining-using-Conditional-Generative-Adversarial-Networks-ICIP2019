import os
import training as Network
import tensorflow as tf
import numpy as np
import cv2
import subprocess
import platform
import optparse
import time
import pywt

def get_wavelets(img):
  coeffs = pywt.dwt2(img, 'haar')
  return coeffs

st = time.time()
cmd = "nvidia-smi" if platform.system() == "Windows" else "which"
try: 
    available_gpus=str(np.argmax( [int(x.split()[2]) 
                  for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", 
                                             shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
    os.environ['CUDA_VISIBLE_DEVICES']=available_gpus
    print("Available GPU added : "+available_gpus)
except: 
    print("No CUDA/GPU available in system")
    os.environ['CUDA_VISIBLE_DEVICES']="-1"
    print("Executing in CPU mode")

parser = optparse.OptionParser()
parser.add_option('-m', '--modelroot', default="./models/ours_-42")
parser.add_option('-v', '--valroot', default="./facades/test_dir_syn/")
parser.add_option('-r', '--resultroot', default="./facades/test_syn_ours_icip/")
opts, args = parser.parse_args()
if not os.path.exists(str(opts.resultroot)):
    os.makedirs(str(opts.resultroot))
total_files=os.listdir(opts.valroot)

image = tf.placeholder(tf.float32, shape=(1, 512, 512, 1))
image_h = tf.placeholder(tf.float32, shape=(1, 512, 512, 1))
image_v = tf.placeholder(tf.float32, shape=(1, 512, 512, 1))
image_d = tf.placeholder(tf.float32, shape=(1, 512, 512, 1))
output = Network.Generator(image, image_h, image_v, image_d, is_training=False)
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, str(opts.modelroot))
    for i in total_files:
        img = cv2.imread(opts.valroot+str(i))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        y_,cr,cb = cv2.split(img)
        y_norm = np.float32(y_)/255.0
        coeffs = get_wavelets(y_)
        H = cv2.resize(coeffs[1][0], (512, 512), interpolation=cv2.INTER_CUBIC)
        V = cv2.resize(coeffs[1][1], (512, 512), interpolation=cv2.INTER_CUBIC)
        D = cv2.resize(coeffs[1][2], (512, 512), interpolation=cv2.INTER_CUBIC)
        final_output  = sess.run(output, feed_dict={image:y_norm.reshape((1,512,512,1)), image_h:H.reshape((1,512,512,1)), image_v:V.reshape((1,512,512,1)), image_d:D.reshape((1,512,512,1))})
        Y = final_output[0,:,:,0]
        Y *= 255.0
        out = np.uint8(Y)
        img = cv2.merge((out,cr,cb))
        img = np.uint8(img)
        img = cv2.cvtColor(img, cv2.COLOR_YCR_CB2BGR)
        words = str(i).split('.')
        cv2.imwrite(opts.resultroot+str(words[0])+'.png',img)
        print('saved image '+str(i)+' at '+opts.resultroot)
end = time.time()
print('Total time taken in secs : '+str(end-st))
print('Per image (avg): '+ str(float((end-st)/1201)))