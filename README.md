# RefVAE
Reference based Image Super-Resolution via Variational AutoEncoder

By Zhi-Song Liu, Li-Wen Wang and Wan-Chi Siu

This repo only provides simple testing codes, pretrained models and the network strategy demo.

We propose a Reference based Image Super-Resolution via Variational AutoEncoder (RefVAE)

We participate CVPRW [Learning the Super-Resolution Space](https://data.vision.ee.ethz.ch/cvl/ntire21/)

Please check our [paper](https://arxiv.org/pdf/2106.04090.pdf)

# BibTex

        @InProceedings{Liu2021refvae,
            author = {Zhi-Song Liu, Wan-Chi Siu and Li-Wen Wang},
            title = {Reference based Image Super-Resolution via Variational AutoEncoder},
            booktitle = {IEEE International Conference on Computer Vision and Pattern Recognition Workshop(CVPRW)},
            month = {June},
            year = {2021}
        }
        
## For proposed RefVAE model, we claim the following points:

• First working on using Variational AutoEncoder for reference based image super-resolution.

• Our proposed RefVAE can expand the SR space so that multiple SR images can be generated.

# Dependencies
    Python > 3.0
    OpenCV library
    Pytorch > 1.0
    NVIDIA GPU + CUDA

# Complete Architecture
The complete architecture is shown as follows,

![network](/figure/figure1.PNG)

# Implementation
## 1. Quick testing
---------------------------------------
1. Download pre-trained from 
https://drive.google.com/file/d/1R3vR7PiFNT26sIBorVoq6Mf-F4pMHfmh/view?usp=sharing

then put the pre-trained models under the "models" folder.

2. Modify "test.py" and run 
```sh
$ python test.py
```


## 2. Training
---------------------------
### s1. Download DIV2K and Flickr2K training images from
    
https://data.vision.ee.ethz.ch/cvl/DIV2K/

https://github.com/LimBee/NTIRE2017

### s2. Download reference images from

https://www.wikiart.org/

### s3. Modify "test.py" and run
```sh
$ python main_GAN.py
```
---------------------------

## Partial SR image comparison

## 1. Visualization comparison
Results on 8x image SR on DIV2K validation dataset
![figure2](/figure/figure2.PNG)

## 2. Quantitative comparison
![figure3](/figure/figure3.PNG)

