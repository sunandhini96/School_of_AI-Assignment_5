# School_of_AI-Assignment_5
Classification of MNIST dataset: Dataset contains 60,000 images in Training and 10,000 images in Testing. Number of classes 10(0-9 numbers).
In this code we implemeted 3 models:
# case 1: Layer Normalization:

Model summary for layer normalization:

            Conv2d-1           [-1, 16, 26, 26]             144
              ReLU-2           [-1, 16, 26, 26]               0
         LayerNorm-3           [-1, 16, 26, 26]          21,632
           Dropout-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 32, 24, 24]           4,608
              ReLU-6           [-1, 32, 24, 24]               0
         LayerNorm-7           [-1, 32, 24, 24]          36,864
           Dropout-8           [-1, 32, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             320
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 12, 10, 10]           1,080
             ReLU-12           [-1, 12, 10, 10]               0
        LayerNorm-13           [-1, 12, 10, 10]           2,400
          Dropout-14           [-1, 12, 10, 10]               0
           Conv2d-15             [-1, 12, 8, 8]           1,296
             ReLU-16             [-1, 12, 8, 8]               0
        LayerNorm-17             [-1, 12, 8, 8]           1,536
          Dropout-18             [-1, 12, 8, 8]               0
           Conv2d-19             [-1, 12, 6, 6]           1,296
             ReLU-20             [-1, 12, 6, 6]               0
        LayerNorm-21             [-1, 12, 6, 6]             864
          Dropout-22             [-1, 12, 6, 6]               0
           Conv2d-23             [-1, 16, 6, 6]           1,728
             ReLU-24             [-1, 16, 6, 6]               0
        LayerNorm-25             [-1, 16, 6, 6]           1,152
          Dropout-26             [-1, 16, 6, 6]               0
        AvgPool2d-27             [-1, 16, 1, 1]               0
           Conv2d-28             [-1, 10, 1, 1]             160
================================================================
Total params: 75,080
Trainable params: 75,080
Non-trainable params: 0

# case 2: Group Normalization
Model summary for group normalization:

            Conv2d-1           [-1, 16, 26, 26]             144
              ReLU-2           [-1, 16, 26, 26]               0
         GroupNorm-3           [-1, 16, 26, 26]              32
           Dropout-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 32, 24, 24]           4,608
              ReLU-6           [-1, 32, 24, 24]               0
         GroupNorm-7           [-1, 32, 24, 24]              64
           Dropout-8           [-1, 32, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             320
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 12, 10, 10]           1,080
             ReLU-12           [-1, 12, 10, 10]               0
        GroupNorm-13           [-1, 12, 10, 10]              24
          Dropout-14           [-1, 12, 10, 10]               0
           Conv2d-15             [-1, 12, 8, 8]           1,296
             ReLU-16             [-1, 12, 8, 8]               0
        GroupNorm-17             [-1, 12, 8, 8]              24
          Dropout-18             [-1, 12, 8, 8]               0
           Conv2d-19             [-1, 12, 6, 6]           1,296
             ReLU-20             [-1, 12, 6, 6]               0
        GroupNorm-21             [-1, 12, 6, 6]              24
          Dropout-22             [-1, 12, 6, 6]               0
           Conv2d-23             [-1, 16, 6, 6]           1,728
             ReLU-24             [-1, 16, 6, 6]               0
        GroupNorm-25             [-1, 16, 6, 6]              32
          Dropout-26             [-1, 16, 6, 6]               0
        AvgPool2d-27             [-1, 16, 1, 1]               0
           Conv2d-28             [-1, 10, 1, 1]             160
================================================================
Total params: 10,832
Trainable params: 10,832
Non-trainable params: 0
# Case 3: L1 + Batch Normalization:
Model summary for L1 + Batch Normalization:

            Conv2d-1           [-1, 16, 26, 26]             144
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
           Dropout-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 32, 24, 24]           4,608
              ReLU-6           [-1, 32, 24, 24]               0
       BatchNorm2d-7           [-1, 32, 24, 24]              64
           Dropout-8           [-1, 32, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             320
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 12, 10, 10]           1,080
             ReLU-12           [-1, 12, 10, 10]               0
      BatchNorm2d-13           [-1, 12, 10, 10]              24
          Dropout-14           [-1, 12, 10, 10]               0
           Conv2d-15             [-1, 12, 8, 8]           1,296
             ReLU-16             [-1, 12, 8, 8]               0
      BatchNorm2d-17             [-1, 12, 8, 8]              24
          Dropout-18             [-1, 12, 8, 8]               0
           Conv2d-19             [-1, 12, 6, 6]           1,296
             ReLU-20             [-1, 12, 6, 6]               0
      BatchNorm2d-21             [-1, 12, 6, 6]              24
          Dropout-22             [-1, 12, 6, 6]               0
           Conv2d-23             [-1, 16, 6, 6]           1,728
             ReLU-24             [-1, 16, 6, 6]               0
      BatchNorm2d-25             [-1, 16, 6, 6]              32
          Dropout-26             [-1, 16, 6, 6]               0
        AvgPool2d-27             [-1, 16, 1, 1]               0
           Conv2d-28             [-1, 10, 1, 1]             160
================================================================
Total params: 10,832
Trainable params: 10,832
Non-trainable params: 0
# 1) Validation Loss Graphs for Layer, Group and L1+ BN Normalization:
<img width="674" alt="sc1_new" src="https://user-images.githubusercontent.com/63030539/215498715-242c6657-8a83-405d-aa82-221e2ffe1b11.png">
# 2) Validation Accuracy Graphs for Layer, Group and L1+ BN Normalization:
<img width="670" alt="sc2_new" src="https://user-images.githubusercontent.com/63030539/215499249-f7994a11-1fee-4ba6-b40f-6c7a03784f96.png">

From graphs we observed group normalization performed better than layer and batch normalization. Number of parameters for group and batch normalization are same. for Layer normalization number of parameters are more. Here we used number of groups 2.

# Performance of the Normalization techniques:
# 1) Layer Normalization:
In Layer Normalization for each batch image mean and variance for all the channels. Number of parameters equal to number of images in batch * 2.
# 2) Group Normalization: 
In Group Normalization for each batch image, number of parameters equal to Number of groups * Layers (number of images in batch * 2)
# 3) L1+ Batch Normalization;
In Batch normalization for mean and variance for each channel of batch and number of parameters equal to number of images in batch.






