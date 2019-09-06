import os
import numpy as np
import cv2
from sewar.full_ref import uqi, mse, msssim, vifp, ssim

cnt=0
clean_dir = './facades/test_syn_clean/'
result_dir = './facades/test_syn_ours_icip/' 
total_files=os.listdir(clean_dir)

uqi_list=[]
mse_list=[]
vif_list=[]
msssim_list=[]

for i in total_files:
    f_name = str(i).split('.')[0]
    img_clean = cv2.imread(clean_dir+str(f_name)+'.jpg')
    img_pred = cv2.imread(result_dir+str(f_name)+'.png')

    uq=uqi(img_clean, img_pred)
    uqi_list.append(uq)

    ms= mse(img_clean, img_pred)
    mse_list.append(ms)

    vi=vifp(img_clean, img_pred)
    vif_list.append(vi)

    mss=msssim(img_clean, img_pred)
    msssim_list.append(mss)

    cnt+=1

    print('Mean  Our  UQI '+str(np.mean(uqi_list)))
    print('Mean  Our  MSE '+str(np.mean(mse_list)))
    print('Mean  Our  VIF '+str(np.mean(vif_list)))
    print('Mean  Our  MSSSIM '+str(np.mean(msssim_list)))

    print(str(cnt)+'\n')


   











































# msssim_list=[]
# mse_list=[]
# tv_list=[]

# total_files=os.listdir(clean_dir)

# cnt=0

# with tf.Session() as sess:
        
#     sess.run(tf.initialize_all_variables())

#     for i in total_files:
#     	img_clean = cv2.imread(clean_dir+str(i))
    	
#     	img_pred = cv2.imread(result_dir+str(i))
    	
#     	img_clean_p = tf.placeholder(tf.float32, shape=[512,512,3])
    	
#     	img_pred_p = tf.placeholder(tf.float32, shape=[512,512,3])
    	
#     	ms = tf.image.ssim_multiscale(img_pred_p, img_clean_p, max_val=255, power_factors=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333])

#     	mse = tf.reduce_mean(tf.square(img_clean_p - img_pred_p))


#     	ssim_ms, mse_val = sess.run([ms, mse], feed_dict={img_clean_p:img_clean, img_pred_p:img_pred})
#     	msssim_list.append(ssim_ms)
#     	mse_list.append(mse_val)
    	
#     	# tv_loss = tf.image.total_variation(img_pred_p, name=None)
#     	# tv_loss_val = sess.run(tv_loss, feed_dict={img_pred_p:img_pred})
#     	# tv_list.append(tv_loss_val)
#     	cnt+=1
#     	print('Mean MS_SSIM '+str(np.mean(msssim_list))+ ' Mean MSE '+str(np.mean(mse_list))+ ' Done:'+str(cnt))
