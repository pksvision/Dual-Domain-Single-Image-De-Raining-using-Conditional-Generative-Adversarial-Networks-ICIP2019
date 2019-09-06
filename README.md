# DD-cGAN 
## Dual-Domain Single Image De-Raining using Conditional Generative Adversarial Networks
[Prasen Kumar Sharma](https://sites.google.com/view/pksvision/home), [Priyankar Jain](), [Arijit Sur](https://www.iitg.ac.in/arijit/)

[[Paper Link](https://ieeexplore.ieee.org/document/8803353)] (ICIP'19)
	
 	@inproceedings{8803353,
	  author={P. K. {Sharma} and P. {Jain} and A. {Sur}}, 
          booktitle={2019 IEEE International Conference on Image Processing (ICIP)}, 
          title={Dual-Domain Single Image De-Raining Using Conditional Generative Adversarial Network}, 
	   year={2019}, 
	   volume={}, 
	   number={}, 
	   pages={2796-2800}, 
	   keywords={Image De-raining;Conditional Generative Adversarial Network (cGAN);Haar Wavelets;Perceptual Loss}, 
	   doi={10.1109/ICIP.2019.8803353}, 
	   ISSN={2381-8549}, 
	   month={Sep.},}
	  

## Prerequisites:
1. Linux
2. Python 2 or 3
3. CPU or NVIDIA GPU + CUDA CuDNN (CUDA 8.0)
 
## Installation:
1. Install Tensorflow and dependencies from https://www.tensorflow.org/install
   (conda install -c conda-forge tensorflow)
   
2. Install python packages: 
   numpy, scipy, PIL, pdb, sewar
   
## Demo using pre-trained model
   python3 testing.py
   
Pre-trained model can be downloaded at (put it in the folder 'models'): https://drive.google.com/drive/folders/13WJn0gjpanrhd07Rv3oO3sHb0i3KaTtv?usp=sharing

Pre-trained models related to wavelets can be downloaded at (put it in the folder 'sub-bands-npzs'): 
https://drive.google.com/drive/folders/1LwqnsJqvCKA-BP44otLJuA1X1Udg2WGj?usp=sharing
   

