# S18

# UNet and Conditional VAE

This repository contains the implementation and results of the UNet and Conditional VAE assignment. This consists of two main parts: training a UNet model from scratch and designing a Conditional VAE.

## Table of Contents

- [UNet Training](#unet-training)
- [Conditional Variational Autoencoder (CVAE) Design](#conditional-variational-autoencoder-cvae-design)
- [Definitions](#definitions)

## UNet Training

The first part involves training a UNet model from scratch using the dataset and strategy provided [here](link-to-dataset).
![image](https://github.com/Delve-ERAV1/S18/assets/11761529/f4a09b01-8669-4b11-99ef-05d5f8e4912a)

The model is trained four times with the following configurations:

### Results

The results for each configuration are reported below-

##### Max Pooling + Transpose Convolution + Binary Cross Entropy
#
```python
model = uNetLightning(bilinear=True, pool=True, dice=False)
```
```
   | Name  | Type             | Params
--------------------------------------------
0  | inc   | DoubleConv       | 38.8 K
1  | down1 | Down             | 221 K 
2  | down2 | Down             | 885 K 
3  | down3 | Down             | 3.5 M 
4  | down4 | Down             | 4.7 M 
5  | up1   | Up               | 5.9 M 
6  | up2   | Up               | 1.5 M 
7  | up3   | Up               | 369 K 
8  | up4   | Up               | 110 K 
9  | outc  | OutConv          | 195   
10 | BCE   | CrossEntropyLoss | 0     
--------------------------------------------
17.3 M    Trainable params
0         Non-trainable params
17.3 M    Total params
69.052    Total estimated model params size (MB)
Epoch 0:   5%|▌         | 3/58 [00:08<02:28,  2.69s/it, v_num=10, train_loss_step=0.952]
Epoch 19:   5%|▌         | 3/58 [00:07<02:26,  2.66s/it, v_num=10, train_loss_step=0.208, val_loss=0.265, train_loss_epoch=0.234] 
/opt/conda/lib/python3.10/site-packages/lightning/pytorch/trainer/call.py:53: UserWarning: Detected KeyboardInterrupt, attempting graceful shutdown...
  rank_zero_warn("Detected KeyboardInterrupt, attempting graceful shutdown...")
```
![image](https://github.com/Delve-ERAV1/S18/assets/11761529/194dbba4-d976-41ba-8dc4-d9c064cf53a0)

##### Max Pooling + Transpose Convolution + Dice Loss
#
```python
model = uNetLightning(bilinear=False, pool=True, dice=True)
```
```
   | Name  | Type             | Params
--------------------------------------------
0  | inc   | DoubleConv       | 38.8 K
1  | down1 | Down             | 221 K 
2  | down2 | Down             | 885 K 
3  | down3 | Down             | 3.5 M 
4  | down4 | Down             | 14.2 M
5  | up1   | Up               | 9.2 M 
6  | up2   | Up               | 2.3 M 
7  | up3   | Up               | 574 K 
8  | up4   | Up               | 143 K 
9  | outc  | OutConv          | 195   
10 | BCE   | CrossEntropyLoss | 0     
--------------------------------------------
31.0 M    Trainable params
0         Non-trainable params
31.0 M    Total params
124.151   Total estimated model params size (MB)
Epoch 0:   3%|▎         | 2/58 [00:07<03:28,  3.72s/it, v_num=0, train_loss_step=0.606]
Epoch 7:   5%|▌         | 3/58 [00:08<02:34,  2.82s/it, v_num=0, train_loss_step=0.148, val_loss=0.190, train_loss_epoch=0.152] 
```
![image](https://github.com/Delve-ERAV1/S18/assets/11761529/aaabb7d2-fba9-4294-b5f3-e2ec09036472)

##### Strided Convolution + Transpose Convolution + Binary Cross Entropy
#
```python
model = uNetLightning(bilinear=False, pool=False, dice=False)
```
```
   | Name  | Type             | Params
--------------------------------------------
0  | inc   | DoubleConv       | 38.8 K
1  | down1 | Down             | 369 K 
2  | down2 | Down             | 1.5 M 
3  | down3 | Down             | 5.9 M 
4  | down4 | Down             | 23.6 M
5  | up1   | Up               | 9.2 M 
6  | up2   | Up               | 2.3 M 
7  | up3   | Up               | 574 K 
8  | up4   | Up               | 143 K 
9  | outc  | OutConv          | 195   
10 | BCE   | CrossEntropyLoss | 0     
--------------------------------------------
43.6 M    Trainable params
0         Non-trainable params
43.6 M    Total params
174.286   Total estimated model params size (MB)
Epoch 11:   5%|▌         | 3/58 [00:09<02:48,  3.07s/it, v_num=0, train_loss_step=0.374, val_loss=0.406, train_loss_epoch=0.358] 
```
![image](https://github.com/Delve-ERAV1/S18/assets/11761529/597f8c43-d164-4802-9016-fa098df1cabb)

##### Strided Convolution + Upsampling + Dice Loss
#
```python
model = uNetLightning(bilinear=False, pool=False, dice=True)
```

```
   | Name  | Type             | Params
--------------------------------------------
0  | inc   | DoubleConv       | 38.8 K
1  | down1 | Down             | 369 K 
2  | down2 | Down             | 1.5 M 
3  | down3 | Down             | 5.9 M 
4  | down4 | Down             | 23.6 M
5  | up1   | Up               | 9.2 M 
6  | up2   | Up               | 2.3 M 
7  | up3   | Up               | 574 K 
8  | up4   | Up               | 143 K 
9  | outc  | OutConv          | 195   
10 | BCE   | CrossEntropyLoss | 0     
--------------------------------------------
43.6 M    Trainable params
0         Non-trainable params
43.6 M    Total params
174.286   Total estimated model params size (MB)
Epoch 7:   3%|▎         | 2/58 [00:05<02:40,  2.87s/it, v_num=0, train_loss_step=0.167, val_loss=0.173, train_loss_epoch=0.164] 
```
![image](https://github.com/Delve-ERAV1/S18/assets/11761529/c8de43af-9c08-482e-ae7b-560aa5db1333)

### Train 
![image](https://github.com/Delve-ERAV1/S18/assets/11761529/03121973-f209-49bd-a958-358d6904d32d)

### Validation
![image](https://github.com/Delve-ERAV1/S18/assets/11761529/ccab60c2-8706-442f-8d45-351d18f30269)


## Conditional Variational Autoencoder (CVAE) Design

The second part involves designing a Conditional VAE with the following specifications:

- The CVAE takes in two inputs:
  1. An image (either from the MNIST and CIFAR10 dataset).
  2. The label of the image (one-hot encoded vector sent through an embedding layer).

- The CVAE is trained as a typical VAE would be.

- For testing, an image is randomly sent to the CVAE with an incorrect label. This is done 25 times, and the outputs are visualized.

##### MNIST Handwritten
![image](https://github.com/Delve-ERAV1/S18/assets/11761529/324f3398-fc6b-40d2-81c6-3ae930146209)
##### Incorrect Label
![image](https://github.com/Delve-ERAV1/S18/assets/11761529/03cc6f48-f3ca-49a3-ab8c-2ed7157c1eef)

##### CIFAR10
![image](https://github.com/Delve-ERAV1/S18/assets/11761529/e58eb960-f2d4-4602-8b3a-39984d2689dd)


### Results
The results for the MNIST and CIFAR10 datasets are provided below. Additionally, the 25 outputs for each dataset are stacked into a single image and labeled appropriately.
