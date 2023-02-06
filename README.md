# Structured-Light-Fringe-Inpainting
### Introduction:
In recent years, the gray code pattern structured light projection technology has been widely used in industrial inspection because of its good robustness and anti-noise. Gray code encoding technology directly projects a sequence of fringe pattern with specific intensity onto the scanned object, It is used to measure height distribution of the scanned object for further in-depth research operations. However, If the scanned object itself is a highly reflective metal object with strong specular reflection surface properties, tend to cause the acquired encoded fringe image to have local area encoding information lost. As a result, the measured point clouds ha a serious accuracy gap. To improve the quality of reconstruction results of highly reflective objects, we proposes a new encoded fringe image inpainting technology. This technology develops a fringe-inpainting system based on generative adversarial network framework, neural network is used to detect the area where the information is lost in the fringe images to repair the fringe features.<br> 


## Required libraries
## Required libraries
Python 3.8  
PyTorch 1.10.1  
tensorboard 2.8.0  
torchvision 0.11.2  
numpy 1.21.2  
opencv-python 4.5.5.62  
Pillow 8.4.0  
scikit-image 0.14.0  

