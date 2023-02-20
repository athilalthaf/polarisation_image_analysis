# Polarisation image Analyis

This project models how insects views and process polarised images. the model remaps the skylight inputs into an insects visual field based on anatomical features of insect eye.

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
files and their respective uses are :

## pol_img_func.py
Script act as a function library for processing skylight images.
The functions in this script can be used for elevation distortion correction, map elevation and azimuth of the sky regions. Along with that we can project the skylight images into a equirectangular projection for the ease of processing. 
There is a partial implementation gaussian kernel function to convolve a equirectangular image to further analyse the images    

## lib_importer.py
An import file that imports all the necessary python libraries needed for the project. Use :

```python
from lib_importer import *
```
to import all the libraries necessary for the project. If needed, any new libraries  can be added to this script making it automatically available for all the other files in this project.


## plots_examples.py
A script that plots the funtions in the pol_img_func file. Majority of the functions accepts a skylight image input. the sample image has been made in [Blender](https://www.blender.org/) and can be used in the from the folder called test_images folder. 



## bee_eye_subsampling.py

A modified version of [create_bee_eye.py](https://github.com/InsectRobotics/InvertPy/blob/version-1.1-alpha/examples/create_bee_eye.py) written by [Evripidis Gkanias](https://github.com/evgkanias). plots ommatidial co-ordinates of bee right eye.

## dummy_file_create_bee_eye.py

A dummy file of [create_bee_eye.py](https://github.com/InsectRobotics/InvertPy/blob/version-1.1-alpha/examples/create_bee_eye.py) just to refer the original version


## Example

Here we will try an example of correcting an image’s elevation distortion. Elevation distortion happens when we are taking a fisheye image, where the image squishes a bit as we go from the centre to the horizon of the fisheye image. 


Start with the following python command to import all the necessary python libraries.Also import the necessary functions from the pol_img_functions


```python
from lib_importer import * 
from pol_img_functions import  azimuth_mapping,elevation_mapping,pixel_map_func
```
Import a fisheye image or a sample image that mimics it, here we have some test images in the folder test_images which will use here.After that we invert the color channel from BGR to RGB.
```python
blend_img_low = cv2.imread("test_images/test_img_voronoi_image_low_res.png") 
blend_img_low = blend_img_low[:,:,[2,1,0]] 
```
these functions require some arguments, like the zenith or the centre of the image in pixel coordinates, radius or the distance from centre to the horizon of the image in pixel distance.  This can be found from using a image editing/processing software like fiji calculating the radius of circle that encompasses the horizon in pixel distance. And centre is the normally the centre most pixel coordinate for a fisheye image.  Thresh is threshold value while mapping the pixels based on the azimuth and elevation mapping. The lower the value, stricter the mapping. Normally low res images have higher threshold values like 1, and it scales down by factor of 10 , when the image scales by a factor of 10 (along the width and height). [very crude method, play around with the value to tune the mapping]. For the image we are using here these are the centre , radius and thresh values.

```python 
centre = [54, 54]                
radius = 51
thresh = 1
```
To map the image, we need to find the azimuth map and the elevation map , so first we generate these maps.

```python
azimuth_map = azimuth_mapping(blend_img_low, radius, centre)
elevation_map_src, elevation_map_corr = elevation_mapping(blend_img_low, radius, centre)
```

Now we map the image, ie. Correct the elevation distortion that is inherent to a fisheye image.
```python
mapped_img = pixel_map_func(blend_img_low, centre, radius,elevation_map_src,elevation_map_corr,azimuth_map,thresh) 
```
After that we can plot the image.
```python
plt.imshow(mapped_img)
plt.show()
```
<img width="201" alt="image" src="https://user-images.githubusercontent.com/77848234/220186171-56bf46d1-1f92-418d-9de3-224c22fbad87.png">

## Author

- The code is written by [athilalthaf](https://github.com/athilalthaf)




## Acknowledgements
- [InvertPy](https://github.com/InsectRobotics/InvertPy)
 - [JJFosterLab](https://github.com/JJFosterLab)
 - [Evripidis Gkanias](https://github.com/evgkanias)
 
