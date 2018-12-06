
## Single Image Rain Removal Using Image Decomposition and Dense Network

	
## Prerequisites:
1. Linux(Ubuntu 14.04 LTS)
2. Python 2.7.6
3. CPU or NVIDIA GPU + CUDA CuDNN
 
## Installation:
1. Install Tensorflow 
2. Install python package: 
   numpy, PIL, skimage
   
## Demo using pre-trained model
1. Preparing image data: 
   
   synthetic data: put rainy images into "/data/synthetic/image/" and label images into "/data/synthetic/label/". 
   
   real-world data: put rainy images into "/data/real/"

2. for test:

    We have offered test.py and pre-trained model  for test
    You can just use 'python test.py' with GPU/CPU, you can obtain your results in /data/result.

3. Datasets:
   
   We release our rainy image dataset Rain100 for tesing.We selected 100 clear images from BSDS500 to synthesize a test dataset using Photoshop, denoted as Rain100. This dataset contains heavy and light rain with different streak directions.
   
   Our training datasets are from the authors of "Removing Rain from Single Images via a Deep Detail Network". 
   It is publicly available at http://smartdsp.xmu.edu.cn/cvpr2017.html   and   百度云 (https://pan.baidu.com/s/1snmg8Kt). 



## Acknowledgments
   We would like to thank the authors of "Removing Rain from Single Images via a Deep Detail Network" for sharing their datasets.
   
   
## Contact

   If you have questions, you can contact YanWF1993@163.com.

