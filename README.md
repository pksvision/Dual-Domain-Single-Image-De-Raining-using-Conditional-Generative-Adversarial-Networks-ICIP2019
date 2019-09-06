# DD-cGAN 
## Dual-Domain Single Image De-Raining using Conditional Generative Adversarial Networks
[Prasen Kumar Sharma](https://sites.google.com/view/pksvision/home), [Priyankar Jain](), [Arijit Sur](https://www.iitg.ac.in/arijit/)

[[Paper Link](https://ieeexplore.ieee.org/document/8803353)] (ICIP'19)

This paper presents a novel method for a single image rain
streak problem which exploits the spatial as well as wavelet
transformed coefficients of the rainy images. The proposed
method adopts the Conditional Generative Adversarial Network [1] framework and consists of two following networks:
Generator and Discriminator. The generator model receives
the input from both spatial, frequency domain of the rainy
image and yields five de-rained image candidates. A Deep
Residual Network [2] has been used to merge these de-rained
candidates and predict a single de-rained image. To ensure the
visual quality of the de-rained image, Perceptual loss function
[3] in addition to adversarial training has been incorporated.
Extensive experiments on the synthetic and real-world rainy
images dataset reveal an improvement over the existing stateof-the-art methods [4], [5] by ∼ 1.08%, 2.57% in Structural
Similarity Index [6] and ∼ 7.39%, 9.95% in Peak signal-tonoise ratio respectively
	
  
 @INPROCEEDINGS{8803353, 
author={P. K. {Sharma} and P. {Jain} and A. {Sur}}, 
booktitle={2019 IEEE International Conference on Image Processing (ICIP)}, 
title={Dual-Domain Single Image De-Raining Using Conditional Generative Adversarial Network}, 
year={2019}, 
volume={}, 
number={}, 
pages={2796-2800}, 
keywords={Image De-raining;Conditional Generative Adversarial Network (cGAN);Haar Wavelets;Perceptual Loss;Deep Learning}, 
doi={10.1109/ICIP.2019.8803353}, 
ISSN={2381-8549}, 
month={Sep.},}






## Prerequisites:
1. Linux
2. Python 2 or 3
3. CPU or NVIDIA GPU + CUDA CuDNN (CUDA 8.0)
 
## Installation:
1. Install PyTorch and dependencies from http://pytorch.org (Ubuntu+Python2.7)
   (conda install pytorch torchvision -c pytorch)

2. Install Torch vision from the source.
   (git clone https://github.com/pytorch/vision
   cd vision
   python setup.py install)

3. Install python package: 
   numpy, scipy, PIL, pdb
   

