# Neural Style Transfer

## What is Neural Style Transfer
NST is an  algorithm that generates an image by combining the content from one image and style from another image.

## Implementation
I've used a pre-trained VGG19 model in .mat format to create a graph from the required layers to calculate the content and style cost. The code vgg_model.py which is used to create a graph is credited in part to the MatConvNet team.

Neural_Style_Transfer.ipynb is an implementation of the following steps:
1. Load the content and style image.
2. Pre-process the images by resizing and subtracting the means from the RGB channels.
3. Generate a random image by adding noise to the content image. This will act as the initialization for the generated image.
4. Calculate the content cost.
5. Calculate the gram matrix and use that to calculate the style cost for one layer. The results are better if we use multiple layers to calculate the average style cost.
6. Define **Total cost = Content cost + Style cost**.
7. Minimize the total cost using the model defined.
8. The result image is obtained by adding back the means to the channels.

## Results
### Style 1

Content image (Me at the Brooklyn bridge, NYC)              |  Style image (Femme nue assise by Pablo Picasso)
:----------------------------------------------------------:|:------------------------------------------------:
<img width="460" height="300" src="https://github.com/mithun-bharadwaj/Neural_Style_Transfer/blob/master/Input/Content.jpg"> |<img width="460" height="500" src="https://github.com/mithun-bharadwaj/Neural_Style_Transfer/blob/master/Input/Style1.jpg">  

**Result**

<p align="center">
  <img width="460" height="450" src="https://github.com/mithun-bharadwaj/Neural_Style_Transfer/blob/master/Output/generated_image_style1.jpg">
</p>

### Style 2

Content image (Me at the Brooklyn bridge, NYC)              |  Style image (Painting by Pablo Picasso)
:----------------------------------------------------------:|:------------------------------------------------:
<img width="460" height="300" src="https://github.com/mithun-bharadwaj/Neural_Style_Transfer/blob/master/Input/Content.jpg"> |<img width="500" height="400" src="https://github.com/mithun-bharadwaj/Neural_Style_Transfer/blob/master/Input/Style2.jpg">  

**Result**

<p align="center">
  <img width="500" height="400" src="https://github.com/mithun-bharadwaj/Neural_Style_Transfer/blob/master/Output/generated_image_style2.jpg">
</p>


## Hyper-parameters
1. Number of epochs (Decent results between 500-1500 epochs- Used GPU to compute).
2. Weights for the content and style cost to calculate the total cost.
3. Weights for the style layers.
4. Learning rate.

## How to run
1. Download the weights of the pre-trained VGG19 model using the link given below.
2. Place the 2 script files in the src folder of this repository in the same root folder as the .mat file
3. Place the input images in the "Input" folder or provide the appropriate path for the script to read the images.
4. Provide a path for the script to write the generated image.

## Pre-trained VGG19 model
http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat
