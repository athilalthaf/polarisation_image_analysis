"""
function library for polarisation images processing and handling of image data.
"""

from lib_importer import *   # import libraries


def img_masking_func(src, num, sample_size, random_color=False):
    """
    function that samples from an image and returns a sampled image
    :param src: np.ndarray
        input image that needs sampling
    :param num: int
        number of the samples along the x and y
    :param sample_size: int
        width of single mask
    :param random_color: bool, optional
        if True returns sample with colored cells if false return a black and white mask
    :return masked image: np.ndarray
          sampled image visualised from a random image
    """
    img_mask = np.zeros(src.shape, dtype="uint8")  # initialising the mask with input image shape
    img_mask_vis = np.zeros(src.shape, dtype="uint8")
    rows, cols = map(int, (src.shape[0], src.shape[1])) # getting the rows and column data of the input image

    x_spacing = np.arange(0, cols, int(np.ceil(cols / num))) #spacing
    y_spacing = np.arange(0, rows, int(np.ceil(rows / num)))
    for x in x_spacing:
        for y in y_spacing:
                cv2.rectangle(img_mask, (x, y), (x + sample_size, y + sample_size), (1, 1, 1), -1) # draw rectangular masks in
                if random_color == True :
                    cv2.rectangle(img_mask_vis, (x, y), (x + sample_size, y + sample_size), (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)), -1)
                else:
                    cv2.rectangle(img_mask_vis, (x, y), (x + sample_size, y + sample_size), (255, 255, 255), -1)


    return img_mask, img_mask_vis


def circular_index(src, radius, num):
    """
    takes an image input and returns list of x,y coordinates arranged in a circular fashion
    :param src: np.ndarray
        input image that whose circular indices are needed
    :param radius: int , pixel units
        radius of the circle needed
    :param num: int
        number of points on the circle needed
    :return x: list
        x coordinates of the points in the circle
    :return y: list
        y coordinates of the points in the circle
    """
    # takes an image input and returns list of x,y coordinates arranged in a circular fashion
    centre = [int(src.shape[0]/2), int(src.shape[1]/2)]      # centre of the image
    angs = np.linspace(0, 2 * np.pi - 2 * np.pi / num, num)  # equi-spaced angles from num of points given
    x = np.zeros(num, dtype=int)                             # initialising output array
    y = np.zeros(num, dtype=int)

    for theta in range(num):
        x[theta] = centre[0] + np.floor(radius * np.cos(angs[theta]))   # converting the polar coordinate info to respective image cordinate
        y[theta] = centre[1] + np.floor(radius * np.sin(angs[theta]))
    print(x, y)
    return x, y



def curve_indexing(src, radius, num, centre=None, outer_angle=360, inner_angle=0, degrees=True):
    """
    getting the coordinates of circular sector in the given image
    :param src: np.nadarray
        input image in which we need the indices
    :param radius: int , pixel units
        radius of the curve
    :param num: int
        number of points on the curve
    :param centre: list ,2 elements, optional
        centre of the curve
    :param outer_angle: float, degree ,optional
    ending angle
    :param inner_angle:  float, degree, optional
        starting angle
    :return x: list
        x coordinates of the points in the sector
    :return y: list
        y coordinates of the points in the sector
    """
    if centre is None:
        centre = [int(src.shape[0]/2), int(src.shape[1]/2)] # in case the centre needs to be specified
    if degrees is True:
        outer_angle = np.deg2rad(outer_angle)             #convert angles to degrees by default case
        inner_angle = np.deg2rad(inner_angle)

    angs = np.linspace(inner_angle, outer_angle - 2*np.pi / num, num) # angle for representing curve ranging from inner to outer
    x = centre[0] + np.floor(radius * np.cos(angs))         # getting the image coordinate from the polar cordinates
    y = centre[1] + np.floor(radius * np.sin(angs))
    x = np.array(x, dtype=int)                              #converting to integers as the image coordinates cannot be float
    y = np.array(y, dtype=int)
    return x, y

def pol_2_equirect(src, radius, centre=None, outer_angle= 180 +360, inner_angle=180, degrees=True,channel=True):
    """
    Function that converts input skylight image into an equi-rectangular projection. by default it unrwraps from north and projects anticlockwise.

    :param src: np.ndarray
        input image that needed to be projected into the equi-rectangular image
    :param radius: int, pixel units
        length from centre to the horizon of the image
    :param centre: list, 2 elements, optional
        zenith of the sky . usually centre in the image
    :param outer_angle: float, optional
        angle at which the projection ends
    :param inner_angle: float, optional
        angle at which the projection starts, 180 denotes north i.e  12'0' clock
    :param degrees: bool, optional
        angle parameters are in degrees or not
    :return polar: np.ndarray
        an image array that is a projection of input image
    """

    num = int(2 * np.pi * radius) # length of the projected image would be the perimeter
    if channel == True:
        polar = np.zeros((radius, num, 3)) # initialising the output image
        polar[:][:] = [255, 0, 0] # initialsing with red color so to check errors while projecting
    else:
        polar = np.zeros((radius, num)) # initialising the output image

    if centre is None:
        centre = [int(src.shape[0]/2), int(src.shape[1]/2)] # take center of the image as the default zenith

    if degrees is True:
        outer_angle = np.deg2rad(outer_angle) # degree conversion and making degree as the default unit
        inner_angle = np.deg2rad(inner_angle)

    angs = np.linspace(inner_angle, outer_angle, num)[:-1] # defining the range of the angles

    for r in range(radius):                    # generate coordinates for different radius
        for i,theta in enumerate(angs):
            x = centre[0] + r * np.sin(theta)  # generate coordinates value for different angles
            y = centre[1] + r * np.cos(theta)
            if channel == True:
                polar[r][i][:] = src[int(y)][int(x)][:]  #mapping the coordinates to the final image
            else:
                polar[r][i] = src[int(y)][int(x)]
    if channel ==True:
        polar[:,-1,:] = polar[:,0,:] #first column is equal to last column 0=360
    else:
        polar[:,-1] = polar[:,0]
    return polar #converting the 8 bit values and avoiding floating points

def gauss_filter(x,y,c,sigma_deg):
    """
    a 2d gaussian function sample
    :param x:  float
        x value
    :param y: float
        y value
    :param c:  float
        correction factor
    :param sigma_deg: float
        sigma of the gaussian
    :return g: float
        the corresponding gaussian value
    """
    g = np.exp(-(x**2/(2*sigma_deg**2)) + y**2/(2*(sigma_deg*c)**2)) #gaussian expression
    return g

def elevation_mapping(src,radius,centre=None):
    """
    function that maps an elevation from a skylight image
    :param src: np.ndarray
        the input image that whose elevation info needs to extracted
    :param radius: int, pixel units
        distance from horizon to zenith of the skylight image
    :param centre: list, optional
        zenith point in the image , default is image centre
    :return ele_map_src: np.ndarray
        corresponding elevation values in the input image
    :return ele_map_corr: np.ndarray
        elevation values corrected for the distortion
    """

    if centre is None: #centre is taken as centre of iamge centre if not specified
        centre = [int(src.shape[0]/2), int(src.shape[1]/2)]

    ele_map_src = np.zeros((src.shape[0], src.shape[1])) # initialising  elevation map values
    ele_map_src[:] = np.nan # replacing with nan so that every point outside the horizon is not included
    corrected_im = np.zeros(src.shape)
    for x in np.arange(centre[0]-radius, centre[0] + radius+1): # only needs the indexes that falls inside the area of the circle
        for y in np.arange(centre[1]-radius, centre[1] + radius+1):
            ele_map_src[x][y] = np.sqrt((x - centre[0]) ** 2 + (y - centre[1]) ** 2) # calculating the hypotenuse
            if abs(np.sqrt((x - centre[0])**2 + (y - centre[1])**2)) > radius: # hypotenuse greater than radius are not valid
                ele_map_src[x][y] = np.nan
            corrected_im[x][y] = src[x][y]
    ele_map_src = ele_map_src / np.nanmax(ele_map_src) # normalised with maximum value to make the range from 0 to 1
    ele_map_corr = (1 - ele_map_src) * np.pi / 2      # normalised value is inverted and multplied with 90 degree to correct for distortion
    ele_map_src = np.arccos(ele_map_src)                #src elevation will be a arccos of normalised hypotenuse

    return ele_map_src, ele_map_corr

def azimuth_mapping(src, radius,centre=None, angle=np.pi/2):
    """
    function for mapping the azimuth from the input skylight images. maps in a clockwise manner,
    :param src: np.ndarray
        the input image that whose azimuth info needs to extracted
    :param radius: int, pixel units
        distance from horizon to zenith of the skylight image
    :param centre: list, 2 elements, optional
        zenith point in the image , default is image centre
    :param angle: float, optional
        angle where 0 degree starts from. default set to north 12 '0' clock
    :return azimuth_map: np.ndarray
        azimuth values corresponding to the skylight image
    """
    if centre is None: #centre is taken as centre of iamge centre if not specified
        centre = [int(src.shape[0]/2), int(src.shape[1]/2)]
    azimuth_map  = np.zeros((src.shape[0],src.shape[1])) #initialising
    azimuth_map[:] = np.nan # replacing with nan so that every point outside the horizon is not included
    for x in np.arange(centre[0] - radius, centre[0] + radius + 1):  # only needs the indexes that falls inside the area of the circle
        for y in np.arange(centre[1] - radius, centre[1] + radius + 1):
                azimuth_map[x][y] =angle - np.arctan2(y - centre[1], x - centre[0]) # azimuth can be estimated from the arctan of a point from the the centre
                if abs(np.sqrt((x - centre[0])**2 + (y - centre[1])**2)) > radius:
                    azimuth_map[x][y] = np.nan
    return azimuth_map

def pixel_map_func(src,centre,radius, elevation_map_src, elevation_map_corr, azimuth_map,thresh=1):
    """
    function to correct for the elevation distortion of the image
    :param src: np.ndarray
        input image that needs to be corrected for elevation
    :param centre: list, 2 elements
        zenith point in the image , default is image centre
    :param radius: int, pixel units
        distance from horizon to zenith of the skylight image
    :param elevation_map_src: np.ndarray
        elevation map of the input image
    :param elevation_map_corr: np.ndarray
        corrected elevation map of the input image
    :param azimuth_map: np.ndarray
        azimuth map of the input image
    :param thresh: float
        threshold value of elevation comparison to avoid floating point error
    :return mapped_img: np.ndarray
        mapped image corrected for elevation distortion
    """
    mapped_img = np.zeros(src.shape) #initialising mapped image
    mapped_img[:] = np.nan
    thresh = np.deg2rad(thresh)
    for x in range(centre[0] - radius, centre[0] + radius + 1): # for every cordinated falls inside the specified area of the circle
        for y in range(centre[1] - radius, centre[1] + radius + 1):
            if abs(np.sqrt((x - centre[0]) ** 2 + (y - centre[1]) ** 2)) > radius:
                pass
            else:
                lookup_values = [azimuth_map[x][y], elevation_map_corr[x][y]] # values that corresponds to the elevation and azimuth
                diff = np.abs(azimuth_map - lookup_values[0]) # difference between azimuth and threshold

                a = np.argwhere(diff<=thresh) # returns indices of azimuth that is very nea
                # a = np.argwhere(azimuth_map == lookup_values[0]) # returns indices of azimuth that is equal to lookupvalue
                b = np.abs(elevation_map_src - lookup_values[1]) # difference between elevation_corr and elvation of lookup_values
                c = b[a[:, 0], a[:, 1]]     #subset of difference from lookup elevation and indices of the correct azimuth
                d = np.nanargmin(c)         # index that returns minimum difference of values of correct azimuth
                lookup_idx = a[d]           # index that returns minimum difference and belongs to the original image
                mapped_img[x][y] = src[lookup_idx[0], lookup_idx[1]] # indexing the lookup values to the mapped image
    # print(x_new, y_new)

    return mapped_img.astype("uint8")


def gauss_kernel(sigma, ele_val,kern_size_x, kern_size_y=None,degrees=True):
    """
    gaussian kernel for an equi-rectangular projected image
    :param sigma: float
        sigma of the kernel
    :param ele_val: float
        value of the elevation
    :param kern_size_x: int
        kernel width
    :param kern_size_y: int, optional
        kernel height , set to same as width unless specified
    :param degrees : bool, optional
        check whether input angles are in degrees or not. by default taken as degrees
    :return kernel: np.ndarray
        gaussian kernel for performing convolution
    """
    if degrees== True: # degree taken as default unit for elevation values
        ele_val = np.deg2rad(ele_val)


    sigma_y = sigma  #sigma values for y axis taken as the same


    if kern_size_y is None:         # setting kernel height same as width if not specified
        kern_size_y = kern_size_x

    if kern_size_x % 2 == 0 or kern_size_y % 2 ==0: # keeping kernel sizes odd
        print("kernel size is not odd!!") #raises warning if not
    else:
        x_mean = kern_size_x // 2                          # middle value for normalizing the centre
        y_mean = kern_size_y // 2
        kernel = np.zeros((kern_size_x, kern_size_y))            #initialising kernel
        sigma_x = sigma * 1 / np.cos(ele_val)             # correcting sigma based on elevation
        for x in range(kern_size_x):
            for y  in range(kern_size_y):

                kernel[x, y] = np.exp(-((x - x_mean)**2/(2 * sigma_y**2) + (y - y_mean)**2 / (2 * sigma_x**2))) # assigning gaussian weights to each kernel pixels


    return kernel

def sub_sampling_func(src, num, sample_size):
    """
    a function to simplify the image and use as a dummy test image
    :param src: np.ndarray
        input image that needs to subsampled
    :param num: int
        num of the samples
    :param sample_size: int
        width of each block
    :return subsample_img: np.ndarray
        subsampled image
    """
    if sample_size == 0:
        remap_img = np.zeros((num, num, 3))
    else:
        remap_img = np.zeros((num * sample_size, num * sample_size, 3))
    subsample_img = src[::int(np.ceil(src.shape[0]/num)), ::int(np.ceil(src.shape[1]/num)), :] #slicing an image based on the num points specified
    return subsample_img, remap_img

def image_tile_function(src,x_tile,y_tile):
    """
    tiling function designed to account for azimuthal wrapping and polar flipping       # partial implementation
    :param src: np.ndarray
        source image that needs to be tiled
    :param x_tile: int
        tiling along the azimuth
    :param y_tile: int
        tiling along the elevation
    :return tiled_img: np.ndarray
        tiled image
    """

    top_rows = src[:y_tile,:,:]           # select the top rows for flipping
    inverted_top_rows = np.flip(top_rows, axis=0 )  # flip over rows and columns and preserve the color data
    inverted_top_rows_shifted = np.roll(inverted_top_rows,src.shape[1]//2,axis=1) # shifting the the azimuth values while it crosses the zenith
    tiled_image = np.vstack([inverted_top_rows_shifted, src]) # stack the flipped slice over the left right tiled image


    x_left_cols = tiled_image[:, :x_tile,:]   # select left columns sets to mapped on the right
    x_right_cols = tiled_image[:, -x_tile:,:] # select right columns sets to mapped on the right

    tiled_image = np.hstack([x_right_cols, tiled_image, x_left_cols]) # joining the azimuth tiling


    return tiled_image




def azimuth_to_idx(src, azi, degree= True):
    """
    calculate the image index corresponding to the azimuth value from an equirectangular image. assumes image width is 360 degree
    :param src: np.ndaaray
        input image whose azimuth index needed to be found
    :param azi: float
        azimuth value whose index needed to be find out, value goes 0 to 360 from left to right of the image
    :param degree: bool
        assumes input azimuth is in degrees
    :return img_azi_val: int
        index value of the corresponding azimuth
    """
    if degree is True:
        azi = np.deg2rad(azi)

    img_azi_val = int(src.shape[1]/ (2 * np.pi) * azi) # normalise image width by max azi value and multiplied by azi value of interest
    return img_azi_val

def elevation_to_idx(src, ele, degree=True):
    """
    calculate the image index corresponding to the elevation  value from equirect image .assumes image width is 90 degree
    :param src:  np.ndarray
        input image whose elevation index needed to be found
    :param ele: float
        elevation value whose index needed to be find out, value goes from 90 to 0 from top to bottom of the image
    :param degree: bool, optional
        assumes input elevation is in degrees
    :return img_azi_val: int
        index value of the corresponding elevation
    """
    if degree is True:
        ele = np.deg2rad(ele)   # normalise image width by max ele value and multiplied by ele value of interest

    img_ele_val = src.shape[0] - (src.shape[0] / ( np.pi/ 2 ) * ele)
    return img_ele_val


def implot_func(imlist,title_list,suptitle = None,lim_lab_list = None ):
    """
    plotting multiple images as subplots
    :param imlist: list, np.ndarray
        list of images needed as a single plot
    :param title_list:  list, str
        list of titles
    :param suptitle: str, optional
    Supertitle of the plot
    :param lim_lab_list: list, list of 2 elements
        list of x and y labels of each image
    :return : list (plot)
    plot that contains all the listed image
    """
    plt.subplots(constrained_layout=True)
    subplot_idx = str("1"+str(len(imlist)))
    for i in range(len(imlist)):
        plt.subplot(int(subplot_idx + str(i+1))) #setting subplot index
        plt.imshow(imlist[i])                   #showing each images
        plt.title(title_list[i])                # titles of each images
        if lim_lab_list is not None :        #setting each x and y label
            plt.xlabel(lim_lab_list[i][0])
            plt.ylabel(lim_lab_list[i][1])
    if suptitle is not None:                    #setting a supertitle
        plt.suptitle(suptitle)

    plt.show()


# scale_factor = .1
# img_bgr = cv2.imread("/Users/athil/Downloads/sample_img.jpg")
# rows, cols, channel = map(int, img_bgr.shape)
# img_bgr_down_samp = cv2.pyrDown(img_bgr, dstsize = (cols // 2, rows // 2))
# img_rgb = img_bgr_down_samp[:, :, [2, 1, 0]]
# img_gray = cv2.cvtColor(img_bgr_down_samp, cv2.COLOR_BGR2GRAY)
#
# num = 10
# sample_size = 0
# img_size = 30
# rand_img = np.random.randint(0, 255, (img_size, img_size, 3),dtype="uint8")
# blank_img = np.zeros((img_size, img_size, 3), dtype="uint8")
# img_mask, img_mask_vis = img_masking_func(rand_img, num, sample_size, random_color=True)
#
#
# sub_sample, canvas = sub_sampling_func(rand_img, num, sample_size)
#
# composite_img = np.ma.multiply(rand_img, img_mask)
# dir= "/Users/athil/Desktop/codes_trial/polarisation_image_analysis/plots/subsampling"
# print(os.getcwd())
#
# pwd = os.getcwd()
#
#
# dict_omm_rho = {"ant": 5.4, "honey bee": 14, "cricket": 35}
# white = [255, 255, 255]
# red = [255, 0, 0]
# green = [0, 255, 0]
# blue = [0, 0, 255]
# black = [0, 0, 0]
#
# img_holder = [None] * len(dict_omm_rho)
#
#
# known_angle =1 #2*np.rad2deg(np.arctan(4.5/71))
# conversion_pix_per_angle = 1  #384/known_angle
#
# sig_fwhm_factor = 2.355
# dict_data = np.array([i for i in dict_omm_rho.values()])
# SIGMA = dict_data / sig_fwhm_factor * conversion_pix_per_angle
#
# kernel_size = int(4 * max(SIGMA))
# if kernel_size % 2 == 0:
#     kernel_size += 1
#
# for aoi, i in zip(dict_omm_rho.keys(), range(len(dict_omm_rho))):
#     img_holder[i] = cv2.GaussianBlur(rand_img, (kernel_size, kernel_size), SIGMA[i])
#
#
# plt_images = img_holder
# plt_images.insert(0, rand_img)
# img_legends = list(dict_omm_rho)
# img_legends.insert(0, "original")
# centre = [blank_img.shape[0], blank_img.shape[1]]
#
# x, y = curve_indexing(blank_img, radius=10, num=50, outer_angle=360, inner_angle=0)
# col_list = np.array([red, green, blue])
# blank_img[(x, y)] = col_list[np.random.randint(col_list.shape[0])]
# trail_x = int(np.ceil(np.sqrt(len(x)))) - len(x)
# trail_y = int(np.ceil(np.sqrt(len(y)))) - len(y)
#
# RAD = np.arange(5,10,2)
# X, Y = [None]*len(RAD), [None]*len(RAD)
# for radius, i in enumerate(RAD):
#     x,y = curve_indexing(blank_img, radius=radius, num=10)
#     print(X)
