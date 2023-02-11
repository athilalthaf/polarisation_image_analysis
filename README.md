# Polarisation image Analyis

This project models how insects views and process polarised images. the model remaps the skylight inputs into an insects visual field based on anatomical features of insect eye.

files and their respective uses are :

## pol_img_func.py
Script act as a function library for processing skylight images.
The functions in this script can be used for elevation distortion correction, map elevation and azimuth of the sky regions. Along with that we can project the skylight images into a equirectangular projection for the ease of processing. 
There is a partial implementation gaussian kernel function to convolve a equirectangular image to further analyse the images    

## lib_importer.py
An import file that imports all the necessary python libraries needed for the project. Use :

```python
from lib_importer import 
```
to import all the libraries necessary for the project. If needed, any new libraries  can be added to this script making it automatically available for all the other files in this project.


## plots_examples.py
A script that plots the funtions in the pol_img_func file. Majority of the functions accepts a skylight image input. the sample image has been made in [Blender](https://www.blender.org/) and can be used in the from the folder called test_images folder. 



## bee_eye_subsampling.py

A modified version of [create_bee_eye.py](https://github.com/InsectRobotics/InvertPy/blob/version-1.1-alpha/examples/create_bee_eye.py) written by [Evripidis Gkanias](https://github.com/evgkanias). plots ommatidial co-ordinates of bee right eye.

## dummy_file_create_bee_eye.py

A dummy file of [create_bee_eye.py](https://github.com/InsectRobotics/InvertPy/blob/version-1.1-alpha/examples/create_bee_eye.py) just to refer the original version

## Installation

Clone the project

```bash
  git clone https://github.com/athilalthaf/polarisation_image_analysis
```

Go to the project directory

```bash
  cd polarisation_image_analysis
```

Install the required libraries using pip

```bash
  pip install -r requirements.txt
```
## Example
A case of correcting for elevation distortion of a given image.

This requires use of pixel_map_func , azimuth_mapping and elevation_mapping functions respectively.


loading libraries and functions
```python
from lib_importer import 
from pol_img_functions import  azimuth_mapping,elevation_mapping,pixel_map_func
```
loading image
```python
blend_img_low = cv2.imread("test_images/test_img_voronoi_image_low_res.png") 
blend_img_low = blend_img_low[:,:,[2,1,0]] 
```
image related arguments
```python 
centre = [54, 54]                
radius = 51
thresh = 1
```
applying functions

```python
azimuth_map = azimuth_mapping(blend_img_low, radius, centre)
elevation_map_src, elevation_map_corr = elevation_mapping(blend_img_low, radius, centre)
mapped_img = pixel_map_func(blend_img_low, centre, radius,elevation_map_src,elevation_map_corr,azimuth_map,thresh) 
```
to plot the functions 

```python
plt.imshow(mapped_img)
```
## Author

- The code is written by [athilalthaf](https://github.com/athilalthaf)




## Acknowledgements
- [InvertPy](https://github.com/InsectRobotics/InvertPy)
 - [JJFosterLab](https://github.com/JJFosterLab)
 - [Evripidis Gkanias](https://github.com/evgkanias)
 
