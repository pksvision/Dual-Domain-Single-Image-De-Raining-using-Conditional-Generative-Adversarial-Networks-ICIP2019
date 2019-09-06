import os
import h5py
import re
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from vgg16 import *
import subprocess
import platform
import pywt 

cmd = "nvidia-smi" if platform.system() == "Windows" else "which"
try: 
    available_gpus=str(np.argmax( [int(x.split()[2]) 
                  for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", 
                                             shell=True, stdout=subprocess.PIPE).stdout.readlines()]))

    os.environ['CUDA_VISIBLE_DEVICES']=available_gpus
    print("LOGGING: Available GPU added : "+available_gpus)
except: 
    print("LOGGING: No CUDA/GPU available in system")
    os.environ['CUDA_VISIBLE_DEVICES']="-1"
    print("LOGGING: Executing in CPU mode")

MOVING_AVERAGE_DECAY = 0.9997
BN_EPSILON = 0.001
BN_DECAY = MOVING_AVERAGE_DECAY
UPDATE_OPS_COLLECTION = 'Derain_update_ops'
DERAIN_VARIABLES = 'Derain_variables'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('num_patches', 20,"""number of patches to read from disk at a time.""")
tf.app.flags.DEFINE_float('learning_rate', 0.00001,"""learning rate.""")
tf.app.flags.DEFINE_integer('epoch', 300,"""epoch.""")
tf.app.flags.DEFINE_integer('batch_size', 20,"""Batch size.""")
tf.app.flags.DEFINE_integer('input_size', 192000,"""Train + Val size.""")
tf.app.flags.DEFINE_integer('train_size', 192000,"""Train size.""")
tf.app.flags.DEFINE_integer('num_channels', 1,"""Number of the input's channels.""")
tf.app.flags.DEFINE_integer('image_size', 128,"""Size of the images.""")
tf.app.flags.DEFINE_integer('label_size', 128,"""Size of the labels.""")
tf.app.flags.DEFINE_string("label_path", "/home/prasen/Rain/Dataset/Training_Dataset_New/Label/", "The path of labels")
tf.app.flags.DEFINE_string("input_path", "/home/prasen/Rain/Dataset/Training_Dataset_New/Input/", "The path of inputs")
tf.app.flags.DEFINE_string("save_model_path", "./saved_models/", "The path of saving model")

if not os.path.exists(FLAGS.save_model_path):
  print("LOGGING: model directory does not exists")
  os.makedirs(FLAGS.save_model_path)
  print("LOGGING: created model dir at "+ str(FLAGS.save_model_path))
else:
  print("LOGGING: model dir exists")

def _get_variable(name, shape, initializer, trainable, weight_decay=0.0, dtype='float'):
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, DERAIN_VARIABLES]
    return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dtype, regularizer=regularizer, collections=collections, trainable=trainable)

def _get_variable_two(name, initializer, trainable, weight_decay=0.0, dtype='float' ):
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.GLOBAL_VARIABLES, DERAIN_VARIABLES]
    return tf.get_variable(name, initializer=initializer, dtype=dtype, regularizer=regularizer, collections=collections, trainable=trainable)    

def bn_pre(x, c, beta_name, beta_val, gamma_name, gamma_val, moving_mean_name, moving_mean_val, moving_variance_name, moving_variance_val):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]
    axis = list(range(len(x_shape) - 1))
    beta = _get_variable_two(beta_name, initializer=beta_val, trainable=False)
    gamma = _get_variable_two(gamma_name, initializer=gamma_val, trainable=False)
    moving_mean =  _get_variable_two(moving_mean_name, initializer=moving_mean_val, trainable=False)
    moving_variance =  _get_variable_two(moving_variance_name, initializer=moving_variance_val, trainable=False)
    mean, variance = tf.nn.moments(x, axis)
    # update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
    # update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
    # tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    # tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
    mean, variance = control_flow_ops.cond(c, lambda: (mean, variance),lambda: (moving_mean, moving_variance))
    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    return x

def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]
    axis = list(range(len(x_shape) - 1))
    beta = _get_variable('beta', params_shape, initializer=tf.zeros_initializer, trainable=True)
    gamma = _get_variable('gamma', params_shape, initializer=tf.ones_initializer, trainable=True)
    moving_mean = _get_variable('moving_mean', params_shape, initializer=tf.zeros_initializer, trainable=False)
    moving_variance = _get_variable('moving_variance', params_shape, initializer=tf.ones_initializer, trainable=False)
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_DECAY)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_DECAY)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)
    mean, variance = control_flow_ops.cond(c, lambda: (mean, variance),lambda: (moving_mean, moving_variance))
    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
    return x

def create_kernel(name, shape, initializer=tf.glorot_uniform_initializer()):
    new_variables = tf.get_variable(name=name, shape=shape, initializer=initializer, trainable=True)
    return new_variables

def read_data(input_path, label_path, start, end):
  train_x = np.zeros((end - start, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))  
  train_y = np.zeros((end - start, FLAGS.image_size, FLAGS.image_size, FLAGS.num_channels))
  for i in range(start, end):
    img = cv2.imread(input_path + str(i) + '.jpg')
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    y,cr,cb = cv2.split(img1)    
    train_x[i-start,:,:,0] = y    
    img = cv2.imread(label_path + str(i) + '.jpg')
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    y,cr,cb = cv2.split(img1)
    train_y[i-start,:,:,0] = y      
    train_x[i-start] /= 255.0
    train_y[i-start] /= 255.0
  return train_x, train_y

ch_weights = np.load("./sub-bands-npzs/model_h.npz")
cv_weights = np.load("./sub-bands-npzs/model_v.npz")
cd_weights = np.load("./sub-bands-npzs/model_d.npz")
ch_mean_var = np.load("./sub-bands-npzs/mean_var_h.npy").item()
cv_mean_var = np.load("./sub-bands-npzs/mean_var_v.npy").item()
cd_mean_var = np.load("./sub-bands-npzs/mean_var_d.npy").item()

def Generator(images, images_h, images_v, images_d, is_training):
   with tf.variable_scope('gen'):
    c = tf.convert_to_tensor(is_training, dtype='bool', name='is_training')
    ################################## Clean A ############################################################
    with tf.variable_scope('inp_conv_1'):
      kernel = create_kernel(name='weights_1', shape=[3, 3, FLAGS.num_channels, 4])
      biases = tf.Variable(tf.constant(0.0, shape=[4], dtype=tf.float32), trainable=True, name='biases_1')      
      conv1 = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
      bias1 = tf.nn.bias_add(conv1, biases)
      bias1 = bn(bias1,c)
      inp1 = tf.nn.relu(bias1)

    with tf.variable_scope('inp_conv_2'):
      kernel = create_kernel(name='weights_2', shape=[3, 3, 4, 8])
      biases = tf.Variable(tf.constant(0.0, shape=[8], dtype=tf.float32), trainable=True, name='biases_2')      
      conv2 = tf.nn.conv2d(inp1, kernel, [1, 1, 1, 1], padding='SAME')
      bias2 = tf.nn.bias_add(conv2, biases)
      bias2 = bn(bias2,c)
      inp2 = tf.nn.relu(bias2)

    with tf.variable_scope('inp_conv_3'):
      kernel = create_kernel(name='weights_3', shape=[3, 3, 8, 16])
      biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name='biases_3')      
      conv3 = tf.nn.conv2d(inp2, kernel, [1, 1, 1, 1], padding='SAME')
      bias3 = tf.nn.bias_add(conv3, biases)
      bias3 = bn(bias3,c)
      inp3 = tf.nn.relu(bias3)

    with tf.variable_scope('inp_conv_4'):
      kernel = create_kernel(name='weights_4', shape=[3, 3, 16, 32])
      biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases_4')      
      conv4 = tf.nn.conv2d(inp3, kernel, [1, 1, 1, 1], padding='SAME')
      bias4 = tf.nn.bias_add(conv4, biases)
      bias4 = bn(bias4,c)
      inp4 = tf.nn.relu(bias4)

    with tf.variable_scope('inp_conv_5'):
      kernel = create_kernel(name='weights_5', shape=[3, 3, 32, 64])
      biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases_5')      
      conv5 = tf.nn.conv2d(inp4, kernel, [1, 1, 1, 1], padding='SAME')
      bias5 = tf.nn.bias_add(conv5, biases)
      bias5 = bn(bias5,c)
      inp5 = tf.nn.relu(bias5)

    with tf.variable_scope('inp_conv_6'):
      kernel = create_kernel(name='weights_6', shape=[3, 3, 64, FLAGS.num_channels])
      biases = tf.Variable(tf.constant(0.0, shape=[FLAGS.num_channels], dtype=tf.float32), trainable=True, name='biases_6')      
      conv6 = tf.nn.conv2d(inp5, kernel, [1, 1, 1, 1], padding='SAME')
      bias6 = tf.nn.bias_add(conv6, biases)
      bias6 = bn(bias6,c)
      clean_a = tf.nn.relu(bias6)
##################################### Clean A ############################################################

########################################## CH ############################################################
    with tf.variable_scope('ch_conv_1'):
      c1 = tf.convert_to_tensor(False, dtype='bool', name='is_training')
      kernel = tf.get_variable(name='weights_1', initializer=tf.constant(ch_weights["gen/ch_conv_1/weights_1:0"]),  trainable=False)
      biases = tf.Variable(tf.constant(ch_weights["gen/ch_conv_1/biases_1:0"]), trainable=False, name='biases_1')      
      conv1 = tf.nn.conv2d(images_h, kernel, [1, 1, 1, 1], padding='SAME')
      bias1 = tf.nn.bias_add(conv1, biases)
      bias1 = bn_pre(bias1, c1, "beta", tf.constant(ch_weights["gen/ch_conv_1/beta:0"]), "gamma", 
                 tf.constant(ch_weights["gen/ch_conv_1/gamma:0"]), "moving_mean", tf.constant(ch_mean_var["gen/ch_conv_1/moving_mean:0"]),
                 "moving_variance", tf.constant(ch_mean_var["gen/ch_conv_1/moving_variance:0"]))
      ch1 = tf.nn.relu(bias1)

    with tf.variable_scope('ch_conv_2'):
      c2 = tf.convert_to_tensor(False, dtype='bool', name='is_training')
      kernel = tf.get_variable(name='weights_2', initializer=tf.constant(ch_weights["gen/ch_conv_2/weights_2:0"]),  trainable=False)
      biases = tf.Variable(tf.constant(ch_weights["gen/ch_conv_2/biases_2:0"]), trainable=False, name='biases_2')      
      conv2 = tf.nn.conv2d(ch1, kernel, [1, 1, 1, 1], padding='SAME')
      bias2 = tf.nn.bias_add(conv2, biases)
      bias2 = bn_pre(bias2, c2, "beta", tf.constant(ch_weights["gen/ch_conv_2/beta:0"]), "gamma", 
                 tf.constant(ch_weights["gen/ch_conv_2/gamma:0"]), "moving_mean", tf.constant(ch_mean_var["gen/ch_conv_2/moving_mean:0"]),
                 "moving_variance", tf.constant(ch_mean_var["gen/ch_conv_2/moving_variance:0"]))
      ch2 = tf.nn.relu(bias2)

    with tf.variable_scope('ch_conv_3'):
      c3 = tf.convert_to_tensor(False, dtype='bool', name='is_training')
      kernel = tf.get_variable(name='weights_3', initializer=tf.constant(ch_weights["gen/ch_conv_3/weights_3:0"]),  trainable=False)
      biases = tf.Variable(tf.constant(ch_weights["gen/ch_conv_3/biases_3:0"]), trainable=False, name='biases_3')      
      conv3 = tf.nn.conv2d(ch2, kernel, [1, 1, 1, 1], padding='SAME')
      bias3 = tf.nn.bias_add(conv3, biases)
      bias3 = bn_pre(bias3, c3, "beta", tf.constant(ch_weights["gen/ch_conv_3/beta:0"]), "gamma", 
                 tf.constant(ch_weights["gen/ch_conv_3/gamma:0"]), "moving_mean", tf.constant(ch_mean_var["gen/ch_conv_3/moving_mean:0"]),
                 "moving_variance", tf.constant(ch_mean_var["gen/ch_conv_3/moving_variance:0"]))
      ch3 = tf.nn.relu(bias3)

    with tf.variable_scope('ch_conv_4'):
      c4 = tf.convert_to_tensor(False, dtype='bool', name='is_training')
      kernel = tf.get_variable(name='weights_4', initializer=tf.constant(ch_weights["gen/ch_conv_4/weights_4:0"]),  trainable=False)
      biases = tf.Variable(tf.constant(ch_weights["gen/ch_conv_4/biases_4:0"]), trainable=False, name='biases_4')      
      conv4 = tf.nn.conv2d(ch3, kernel, [1, 1, 1, 1], padding='SAME')
      bias4 = tf.nn.bias_add(conv4, biases)
      bias4 = bn_pre(bias4, c4, "beta", tf.constant(ch_weights["gen/ch_conv_4/beta:0"]), "gamma", 
                 tf.constant(ch_weights["gen/ch_conv_4/gamma:0"]), "moving_mean", tf.constant(ch_mean_var["gen/ch_conv_4/moving_mean:0"]),
                 "moving_variance", tf.constant(ch_mean_var["gen/ch_conv_4/moving_variance:0"]))
      ch4 = tf.nn.relu(bias4)
######################################### CH ############################################################# 

########################################## CV ############################################################
    with tf.variable_scope('cv_conv_1'):
      c1 = tf.convert_to_tensor(False, dtype='bool', name='is_training')
      kernel = tf.get_variable(name='weights_1', initializer=tf.constant(cv_weights["gen/cV_conv_1/weights_1:0"]),  trainable=False)
      biases = tf.Variable(tf.constant(cv_weights["gen/cV_conv_1/biases_1:0"]), trainable=False, name='biases_1')      
      conv1 = tf.nn.conv2d(images_v, kernel, [1, 1, 1, 1], padding='SAME')
      bias1 = tf.nn.bias_add(conv1, biases)
      bias1 = bn_pre(bias1, c1, "beta", tf.constant(cv_weights["gen/cV_conv_1/beta:0"]), "gamma", 
                 tf.constant(cv_weights["gen/cV_conv_1/gamma:0"]), "moving_mean", tf.constant(cv_mean_var["gen/cV_conv_1/moving_mean:0"]),
                 "moving_variance", tf.constant(cv_mean_var["gen/cV_conv_1/moving_variance:0"]))
      cv1 = tf.nn.relu(bias1)

    with tf.variable_scope('cv_conv_2'):
      c2 = tf.convert_to_tensor(False, dtype='bool', name='is_training')
      kernel = tf.get_variable(name='weights_2', initializer=tf.constant(cv_weights["gen/cV_conv_2/weights_2:0"]),  trainable=False)
      biases = tf.Variable(tf.constant(cv_weights["gen/cV_conv_2/biases_2:0"]), trainable=False, name='biases_2')      
      conv2 = tf.nn.conv2d(cv1, kernel, [1, 1, 1, 1], padding='SAME')
      bias2 = tf.nn.bias_add(conv2, biases)
      bias2 = bn_pre(bias2, c2, "beta", tf.constant(cv_weights["gen/cV_conv_2/beta:0"]), "gamma", 
                 tf.constant(cv_weights["gen/cV_conv_2/gamma:0"]), "moving_mean", tf.constant(cv_mean_var["gen/cV_conv_2/moving_mean:0"]),
                 "moving_variance", tf.constant(cv_mean_var["gen/cV_conv_2/moving_variance:0"]))
      cv2 = tf.nn.relu(bias2)

    with tf.variable_scope('cv_conv_3'):
      c3 = tf.convert_to_tensor(False, dtype='bool', name='is_training')
      kernel = tf.get_variable(name='weights_3', initializer=tf.constant(cv_weights["gen/cV_conv_3/weights_3:0"]),  trainable=False)
      biases = tf.Variable(tf.constant(cv_weights["gen/cV_conv_3/biases_3:0"]), trainable=False, name='biases_3')      
      conv3 = tf.nn.conv2d(cv2, kernel, [1, 1, 1, 1], padding='SAME')
      bias3 = tf.nn.bias_add(conv3, biases)
      bias3 = bn_pre(bias3, c3, "beta", tf.constant(cv_weights["gen/cV_conv_3/beta:0"]), "gamma", 
                 tf.constant(cv_weights["gen/cV_conv_3/gamma:0"]), "moving_mean", tf.constant(cv_mean_var["gen/cV_conv_3/moving_mean:0"]),
                 "moving_variance", tf.constant(cv_mean_var["gen/cV_conv_3/moving_variance:0"]))
      cv3 = tf.nn.relu(bias3)

    with tf.variable_scope('cv_conv_4'):
      c4 = tf.convert_to_tensor(False, dtype='bool', name='is_training')
      kernel = tf.get_variable(name='weights_4', initializer=tf.constant(cv_weights["gen/cV_conv_4/weights_4:0"]),  trainable=False)
      biases = tf.Variable(tf.constant(cv_weights["gen/cV_conv_4/biases_4:0"]), trainable=False, name='biases_4')      
      conv4 = tf.nn.conv2d(cv3, kernel, [1, 1, 1, 1], padding='SAME')
      bias4 = tf.nn.bias_add(conv4, biases)
      bias4 = bn_pre(bias4, c4, "beta", tf.constant(cv_weights["gen/cV_conv_4/beta:0"]), "gamma", 
                 tf.constant(cv_weights["gen/cV_conv_4/gamma:0"]), "moving_mean", tf.constant(cv_mean_var["gen/cV_conv_4/moving_mean:0"]),
                 "moving_variance", tf.constant(cv_mean_var["gen/cV_conv_4/moving_variance:0"]))
      cV4 = tf.nn.relu(bias4)
######################################### CV #############################################################    

########################################## CD ############################################################
    with tf.variable_scope('cd_conv_1'):
      c1 = tf.convert_to_tensor(False, dtype='bool', name='is_training')
      kernel = tf.get_variable(name='weights_1', initializer=tf.constant(cd_weights["gen/cd_conv_1/weights_1:0"]),  trainable=False)
      biases = tf.Variable(tf.constant(cd_weights["gen/cd_conv_1/biases_1:0"]), trainable=False, name='biases_1')      
      conv1 = tf.nn.conv2d(images_d, kernel, [1, 1, 1, 1], padding='SAME')
      bias1 = tf.nn.bias_add(conv1, biases)
      bias1 = bn_pre(bias1, c1, "beta", tf.constant(cd_weights["gen/cd_conv_1/beta:0"]), "gamma", 
                 tf.constant(cd_weights["gen/cd_conv_1/gamma:0"]), "moving_mean", tf.constant(cd_mean_var["gen/cd_conv_1/moving_mean:0"]),
                 "moving_variance", tf.constant(cd_mean_var["gen/cd_conv_1/moving_variance:0"]))
      cd1 = tf.nn.relu(bias1)

    with tf.variable_scope('cd_conv_2'):
      c2 = tf.convert_to_tensor(False, dtype='bool', name='is_training')
      kernel = tf.get_variable(name='weights_2', initializer=tf.constant(cd_weights["gen/cd_conv_2/weights_2:0"]),  trainable=False)
      biases = tf.Variable(tf.constant(cd_weights["gen/cd_conv_2/biases_2:0"]), trainable=False, name='biases_2')      
      conv2 = tf.nn.conv2d(cd1, kernel, [1, 1, 1, 1], padding='SAME')
      bias2 = tf.nn.bias_add(conv2, biases)
      bias2 = bn_pre(bias2, c2, "beta", tf.constant(cd_weights["gen/cd_conv_2/beta:0"]), "gamma", 
                 tf.constant(cd_weights["gen/cd_conv_2/gamma:0"]), "moving_mean", tf.constant(cd_mean_var["gen/cd_conv_2/moving_mean:0"]),
                 "moving_variance", tf.constant(cd_mean_var["gen/cd_conv_2/moving_variance:0"]))
      cd2 = tf.nn.relu(bias2)

    with tf.variable_scope('cd_conv_3'):
      c3 = tf.convert_to_tensor(False, dtype='bool', name='is_training')
      kernel = tf.get_variable(name='weights_3', initializer=tf.constant(cd_weights["gen/cd_conv_3/weights_3:0"]),  trainable=False)
      biases = tf.Variable(tf.constant(cd_weights["gen/cd_conv_3/biases_3:0"]), trainable=False, name='biases_3')      
      conv3 = tf.nn.conv2d(cd2, kernel, [1, 1, 1, 1], padding='SAME')
      bias3 = tf.nn.bias_add(conv3, biases)
      bias3 = bn_pre(bias3, c3, "beta", tf.constant(cd_weights["gen/cd_conv_3/beta:0"]), "gamma", 
                 tf.constant(cd_weights["gen/cd_conv_3/gamma:0"]), "moving_mean", tf.constant(cd_mean_var["gen/cd_conv_3/moving_mean:0"]),
                 "moving_variance", tf.constant(cd_mean_var["gen/cd_conv_3/moving_variance:0"]))
      cd3 = tf.nn.relu(bias3)

    with tf.variable_scope('cd_conv_4'):
      c4 = tf.convert_to_tensor(False, dtype='bool', name='is_training')
      kernel = tf.get_variable(name='weights_4', initializer=tf.constant(cd_weights["gen/cd_conv_4/weights_4:0"]),  trainable=False)
      biases = tf.Variable(tf.constant(cd_weights["gen/cd_conv_4/biases_4:0"]), trainable=False, name='biases_4')      
      conv4 = tf.nn.conv2d(cd3, kernel, [1, 1, 1, 1], padding='SAME')
      bias4 = tf.nn.bias_add(conv4, biases)
      bias4 = bn_pre(bias4, c4, "beta", tf.constant(cd_weights["gen/cd_conv_4/beta:0"]), "gamma", 
                 tf.constant(cd_weights["gen/cd_conv_4/gamma:0"]), "moving_mean", tf.constant(cd_mean_var["gen/cd_conv_4/moving_mean:0"]),
                 "moving_variance", tf.constant(cd_mean_var["gen/cd_conv_4/moving_variance:0"]))
      cd4 = tf.nn.relu(bias4)
######################################### CD #############################################################

    maps = tf.concat([ch4, cV4, cd4], axis=3)

##################################################Merge Maps#########################################################

    with tf.variable_scope('merge_maps_conv_1'):
      kernel = create_kernel(name='weights_1', shape=[3, 3, FLAGS.num_channels+2, 16])
      biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name='biases_1')      
      conv1 = tf.nn.conv2d(maps, kernel, [1, 1, 1, 1], padding='SAME')
      bias1 = tf.nn.bias_add(conv1, biases)
      bias1 = bn(bias1,c)
      conv_shortcut = tf.nn.relu(bias1)
  
    for i in range(4):
      with tf.variable_scope('merge_maps_conv_%s'%(i*2+2)):
        kernel = create_kernel(name=('weights_%s'%(i*2+2)), shape=[3, 3, 16, 16])
        biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name=('biases_%s'%(i*2+2)))
        conv_tmp1 = tf.nn.conv2d(conv_shortcut, kernel, [1, 1, 1, 1], padding='SAME')     
        bias_tmp1 = tf.nn.bias_add(conv_tmp1, biases)
        bias_tmp1 = bn(bias_tmp1,c)
        out_tmp1 = tf.nn.relu(bias_tmp1)

      with tf.variable_scope('merge_maps_conv_%s'%(i*2+3)): 
        kernel = create_kernel(name=('weights_%s'%(i*2+3)), shape=[3, 3, 16, 16])
        biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name=('biases_%s'%(i*2+3))) 
        conv_tmp2 = tf.nn.conv2d(out_tmp1, kernel, [1, 1, 1, 1], padding='SAME')     
        bias_tmp2 = tf.nn.bias_add(conv_tmp2, biases)
        bias_tmp2 = bn(bias_tmp2,c)
        bias_tmp2 = tf.nn.relu(bias_tmp2)  
        conv_shortcut = tf.add(conv_shortcut, bias_tmp2)

    with tf.variable_scope('merge_maps_conv_12'):
      kernel = create_kernel(name='weights_12', shape=[3, 3, 16, FLAGS.num_channels])   
      biases = tf.Variable(tf.constant(0.0, shape=[FLAGS.num_channels], dtype=tf.float32), trainable=True, name='biases_12')
      conv_final = tf.nn.conv2d(conv_shortcut, kernel, [1, 1, 1, 1], padding='SAME')
      bias_final = tf.nn.bias_add(conv_final, biases)
      final = bn(bias_final,c)
      final_map= tf.nn.relu(final)

##################################################Merge Maps#########################################################

    clean_b = tf.subtract(images, final_map)
    clean_c = tf.subtract(images, ch4)
    clean_d = tf.subtract(images, cV4)
    clean_e = tf.subtract(images, cd4)
    clean_concat = tf.concat([clean_a, clean_b, clean_c, clean_d, clean_e], axis=3)

    with tf.variable_scope('conv_1'):
      kernel = create_kernel(name='weights_1', shape=[3, 3, FLAGS.num_channels+4, 16])
      biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name='biases_1')      
      conv1 = tf.nn.conv2d(clean_concat, kernel, [1, 1, 1, 1], padding='SAME')
      bias1 = tf.nn.bias_add(conv1, biases)
      bias1 = bn(bias1,c)
      conv_shortcut = tf.nn.relu(bias1)
  
    for i in range(16):
      with tf.variable_scope('conv_%s'%(i*2+2)):
        kernel = create_kernel(name=('weights_%s'%(i*2+2)), shape=[3, 3, 16, 16])
        biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name=('biases_%s'%(i*2+2)))
        conv_tmp1 = tf.nn.conv2d(conv_shortcut, kernel, [1, 1, 1, 1], padding='SAME')     
        bias_tmp1 = tf.nn.bias_add(conv_tmp1, biases)
        bias_tmp1 = bn(bias_tmp1,c)
        out_tmp1 = tf.nn.relu(bias_tmp1)

      with tf.variable_scope('conv_%s'%(i*2+3)): 
        kernel = create_kernel(name=('weights_%s'%(i*2+3)), shape=[3, 3, 16, 16])
        biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name=('biases_%s'%(i*2+3))) 
        conv_tmp2 = tf.nn.conv2d(out_tmp1, kernel, [1, 1, 1, 1], padding='SAME')     
        bias_tmp2 = tf.nn.bias_add(conv_tmp2, biases)
        bias_tmp2 = bn(bias_tmp2,c)
        bias_tmp2 = tf.nn.relu(bias_tmp2)  
        conv_shortcut = tf.add(conv_shortcut, bias_tmp2)

    with tf.variable_scope('conv_36'):
      kernel = create_kernel(name='weights_36', shape=[3, 3, 16, FLAGS.num_channels])   
      biases = tf.Variable(tf.constant(0.0, shape=[FLAGS.num_channels], dtype=tf.float32), trainable=True, name='biases_36')
      conv_final = tf.nn.conv2d(conv_shortcut, kernel, [1, 1, 1, 1], padding='SAME')
      bias_final = tf.nn.bias_add(conv_final, biases)
      final_image = bn(bias_final,c)
      final= tf.nn.relu(final_image)
   
   return final
    
def Discriminator(image, reuse):
    with tf.variable_scope('dis',reuse=reuse):
        k1 = create_kernel(name='disweights_1', shape=[3, 3, FLAGS.num_channels, FLAGS.num_channels])   
        b1 = tf.Variable(tf.constant(0.0, shape=[FLAGS.num_channels], dtype=tf.float32), trainable=True, name='disbiases_1')
        c1 = tf.nn.conv2d(image, k1, [1, 1, 1, 1], padding='SAME')
        cout1 = tf.nn.bias_add(c1, b1)
        out1= tf.nn.relu(cout1)

        k2 = create_kernel(name='disweights_2', shape=[3, 3, FLAGS.num_channels, 2*(FLAGS.num_channels)])   
        b2 = tf.Variable(tf.constant(0.0, shape=[2*(FLAGS.num_channels)], dtype=tf.float32), trainable=True, name='disbiases_2')
        c2 = tf.nn.conv2d(out1, k2, [1, 1, 1, 1], padding='SAME')
        cout2 = tf.nn.bias_add(c2, b2)
        out2= tf.nn.relu(cout2)

        k3 = create_kernel(name='disweights_3', shape=[3, 3,  2*(FLAGS.num_channels), 4*(FLAGS.num_channels)])   
        b3 = tf.Variable(tf.constant(0.0, shape=[4*(FLAGS.num_channels)], dtype=tf.float32), trainable=True, name='disbiases_3')
        c3 = tf.nn.conv2d(out2, k3, [1, 1, 1, 1], padding='SAME')
        cout3 = tf.nn.bias_add(c3, b3)
        out3= tf.nn.relu(cout3)

        k4 = create_kernel(name='disweights_4', shape=[3, 3, 4*(FLAGS.num_channels), 8*(FLAGS.num_channels)])   
        b4 = tf.Variable(tf.constant(0.0, shape=[8*(FLAGS.num_channels)], dtype=tf.float32), trainable=True, name='disbiases_4')
        c4 = tf.nn.conv2d(out3, k4, [1, 1, 1, 1], padding='SAME')
        cout4 = tf.nn.bias_add(c4, b4)
        out4= tf.nn.relu(cout4)

        k5 = create_kernel(name='disweights_5', shape=[3, 3, 8*(FLAGS.num_channels), FLAGS.num_channels])   
        b5 = tf.Variable(tf.constant(0.0, shape=[FLAGS.num_channels], dtype=tf.float32), trainable=True, name='disbiases_5')
        c5 = tf.nn.conv2d(out4, k5, [1, 1, 1, 1], padding='SAME')
        cout5 = tf.nn.bias_add(c5, b5)
        out5= tf.nn.relu(cout5)

        out5_shape = out5.get_shape().as_list()

        out5 = tf.reshape(out5, [-1, out5_shape[1]*out5_shape[2]*FLAGS.num_channels])
        hidden=tf.layers.dense(inputs=out5,units=128,activation=tf.nn.relu)
        logits=tf.layers.dense(hidden,units=1)
        output=tf.sigmoid(logits)
        return output,logits

def get_wavelets(img):
  coeffs = pywt.dwt2(img, 'haar')
  return coeffs

# shape is a list
def _smaller(img, shape):
  return cv2.resize(img, (shape[0], shape[1]), interpolation=cv2.INTER_CUBIC)