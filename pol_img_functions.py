from lib_importer import *
# import os
# # from basic import dict_omm_rho
#
#
# import numpy as np
# from numpy import sin, cos, linspace,ceil,floor, max, min, round, abs,exp
# import cv2
# import matplotlib.pyplot as plt
# def down_sampling(src,iter):
#     init_size = src.shape
#     img_down_samp = src
#     for i in range(iter):
#         img_down_samp = cv2.pyrDown(img_down_samp,dstsize = (cols // 2, rows // 2))
#     print("size from "+str(init_size) +"to"+str(img_down_samp.shape))
#     return img_down_samp


def img_masking_func(src, num, sample_size, plot=False, random_color=False):
    img_mask = np.zeros(src.shape, dtype="uint8")
    img_mask_vis = np.zeros(src.shape, dtype="uint8")
    rows, cols = map(int, (src.shape[0], src.shape[1]))
    # #spacing
    x_spacing = np.arange(0, cols, int(np.ceil(cols / num)))
    y_spacing = np.arange(0, rows, int(np.ceil(rows / num)))
    for x in x_spacing:
        for y in y_spacing:
                cv2.rectangle(img_mask, (x, y), (x + sample_size, y + sample_size), (1, 1, 1), -1)
                if random_color == True :
                    cv2.rectangle(img_mask_vis, (x, y), (x + sample_size, y + sample_size), (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)), -1)
                else:
                    cv2.rectangle(img_mask_vis, (x, y), (x + sample_size, y + sample_size), (255, 255, 255), -1)

    # mask_plot = cv2.imshow("Rect_masks", img_mask)
    # if plot == True:
    #     plt.imshow(img_mask, cmap= "gray",vmin=0, vmax= 255)
    #     plt.show()
    return img_mask, img_mask_vis


def circular_index(src, radius, num):
    # takes an image input and returns list of x,y coordinates arranged in a circular fashion
    centre = [int(src.shape[0]/2), int(src.shape[1]/2)]
    angs = np.linspace(0, 2 * np.pi - 2 * np.pi / num, num)
    x = np.zeros(num, dtype=int)
    y = np.zeros(num, dtype=int)

    for theta in range(num):
        x[theta] = centre[0] + np.floor(radius * np.cos(angs[theta]))
        y[theta] = centre[1] + np.floor(radius * np.sin(angs[theta]))
    print(x, y)
    return x, y
# def img_resize_func(mask,num,sample_size):
#     thresh_vals = mask[mask!=[0,0,0]]
#     img_resize = thresh_vals.reshape(sample_size * num, sample_size * num,3)
#     return img_resize


def curve_indexing(src, radius, num, centre=None, outer_angle=360, inner_angle=0, degrees=True):
    if centre is None:
        centre = [int(src.shape[0]/2), int(src.shape[1]/2)]
    if degrees is True:
        outer_angle = np.deg2rad(outer_angle)
        inner_angle = np.deg2rad(inner_angle)

    angs = np.linspace(inner_angle, outer_angle - 2*np.pi / num, num)
    x = centre[0] + np.floor(radius * np.cos(angs))
    y = centre[1] + np.floor(radius * np.sin(angs))
    x = np.array(x, dtype=int)
    y = np.array(y, dtype=int)
    return x, y

def pol_2_equirect(src, radius, num=None, centre=None, outer_angle=360, inner_angle=0, degrees=True):
    if num is None:
        num = int(2 * np.pi * radius)
    polar = np.zeros((radius, num, 3))
    polar[:][:] = [255, 0, 0]
    if centre is None:
        centre = [int(src.shape[0]/2), int(src.shape[1]/2)]
    if degrees is True:
        outer_angle = np.deg2rad(outer_angle)
        inner_angle = np.deg2rad(inner_angle)

    angs = np.linspace(inner_angle, outer_angle, num)[:-1]
    #
    # x = np.array(x, dtype=int)
    # y = np.array(y, dtype=int)
    for r in range(radius): #
        for i,theta in enumerate(angs):
            x = centre[0] + r * np.sin(theta)
            y = centre[1] + r * np.cos(theta)
            # i = np.asarray(np.where(angs==theta))[0]
            # polar[r][int(np.rad2deg(theta))][:] = src[int(x)][int(y)][:]
            polar[r][i][:] = src[int(y)][int(x)][:]


    polar[:,-1,:] = polar[:,0,:]

    return polar.astype("uint8")

def gauss_filter(x,y,c,sigma_deg):
    g = np.exp(-(x**2/(2*sigma_deg**2)) + y**2/(2*(sigma_deg*c)**2))
    return g

def elevation_mapping(src,radius,centre=None):
    if centre is None:
        centre = [int(src.shape[0]/2), int(src.shape[1]/2)]

    ele_map_src = np.zeros((src.shape[0], src.shape[1]))
    ele_map_src[:] = np.nan
    corrected_im = np.zeros(src.shape)
    for x in np.arange(centre[0]-radius, centre[0] + radius+1):
        for y in np.arange(centre[1]-radius, centre[1] + radius+1):
            ele_map_src[x][y] = np.sqrt((x - centre[0]) ** 2 + (y - centre[1]) ** 2)
            if abs(np.sqrt((x - centre[0])**2 + (y - centre[1])**2)) > radius:
                ele_map_src[x][y] = np.nan
            corrected_im[x][y] = src[x][y]
    ele_map_src = ele_map_src / np.nanmax(ele_map_src)
    ele_map_corr = (1 - ele_map_src) * np.pi / 2
    ele_map_src = np.arccos(ele_map_src)


    # np.where(ele_ma[:,]**2 + ele_map[,:]**2 >radius, np.nan, ele_map)
    # mat = plt.matshow(ele_map)
    # plt.colorbar(mat)q
    return ele_map_src, ele_map_corr#, azi_map, corrected_im

def azimuth_mapping(src, radius,centre=None, angle=np.pi/2):
    if centre is None:
        centre = [int(src.shape[0]/2), int(src.shape[1]/2)]
    azimuth_map  = np.zeros((src.shape[0],src.shape[1]))
    azimuth_map[:] = np.nan
    for x in np.arange(centre[0] - radius, centre[0] + radius + 1):
        for y in np.arange(centre[1] - radius, centre[1] + radius + 1):
                azimuth_map[x][y] =angle - np.arctan2(y - centre[1], x - centre[0])
                if abs(np.sqrt((x - centre[0])**2 + (y - centre[1])**2)) > radius:
                    azimuth_map[x][y] = np.nan
    return azimuth_map

def pixel_map_func(src,centre,radius, elevation_map_src, elevation_map_corr, azimuth_map,thresh=1):

    # centre = [int(src.shape[0] / 2), int(src.shape[1] / 2)]
    mapped_img = np.zeros(src.shape)
    mapped_img[:] = np.nan
    thresh = np.deg2rad(thresh)
    for x in range(centre[0] - radius, centre[0] + radius + 1):
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


def gauss_filter_function(sigma,kern_size_x,kern_size_y=None):
                                        # kernel size should be odd
    if kern_size_y is None:
        kern_size_y = kern_size_x

    if kern_size_x % 2 == 0 or kern_size_y % 2 ==0:
        print("kernel size is not odd!!")
    else:
        x_mean = kern_size_x // 2                          # middle value for normalizing the centre
        y_mean = kern_size_y // 2
        kernel = np.zeros((kern_size_x, kern_size_y))            #### check the color value is needed for kernel
        for x in range(kern_size_x):
            for y  in range(kern_size_y):
                kernel[x][y] = np.exp(-((x - x_mean)**2/(2 * sigma**2) + (y - y_mean)**2/(2 * sigma**2)))


    return kernel

def sub_sampling_func(src, num, sample_size):
    if sample_size == 0:
        remap_img = np.zeros((num, num, 3))
    else:
        remap_img = np.zeros((num * sample_size, num * sample_size, 3))
    subsample_img = src[::int(np.ceil(src.shape[0]/num)), ::int(np.ceil(src.shape[1]/num)), :]
    # x_spacing = np.arange(0, cols, int(np.ceil(cols / num)))
    # y_spacing = np.arange(0, rows, int(np.ceil(rows / num)))
    # canvas[::sample_size][::sample_size][:] = src[x_spacing,y_spacing][:]
    # for x in x_spacing:
    #     for y in y_spacing:
    #         canvas[int(np.ceil(x/(num * sample_size)))][int(np.ceil(y/(num * sample_size)))] = src[x][y]
    # # canvas[::sample_size] = src[x_spacing]
    print(subsample_img.shape, remap_img.shape)
    return subsample_img, remap_img



def implot_func(imlist,title_list,suptitle = None,save= False,lim_lab_list = None ):
    plt.subplots(constrained_layout=True)
    subplot_idx = str("1"+str(len(imlist)))
    for i in range(len(imlist)):
        plt.subplot(int(subplot_idx + str(i+1)))
        plt.imshow(imlist[i])
        plt.title(title_list[i])
        if lim_lab_list is not None :
            plt.xlabel(lim_lab_list[i][0])
            plt.ylabel(lim_lab_list[i][1])
    if suptitle is not None:
        plt.suptitle(suptitle)

    plt.show()


scale_factor = .1
img_bgr = cv2.imread("/Users/athil/Downloads/sample_img.jpg")
rows, cols, channel = map(int, img_bgr.shape)
img_bgr_down_samp = cv2.pyrDown(img_bgr, dstsize = (cols // 2, rows // 2))
img_rgb = img_bgr_down_samp[:, :, [2, 1, 0]]
img_gray = cv2.cvtColor(img_bgr_down_samp, cv2.COLOR_BGR2GRAY)

num = 10
sample_size = 0
img_size = 30
rand_img = np.random.randint(0, 255, (img_size, img_size, 3),dtype="uint8")
blank_img = np.zeros((img_size, img_size, 3), dtype="uint8")
img_mask, img_mask_vis = img_masking_func(rand_img, num, sample_size, random_color=True)

# img_resize = img_resize_func(img_mask_vis, num, sample_size)
sub_sample, canvas = sub_sampling_func(rand_img, num, sample_size)

# cv2.imshow("subsampled",resize)
# composite_img = cv2.bitwise_and(img_bgr_down_samp, img_bgr_down_samp,mask =img_mask)
# img_mask.astype("uint8")
composite_img = np.ma.multiply(rand_img, img_mask)
# composite_img.astype("uint8")
#
# plt.imshow(composite_img)
# plt.show()
dir= "/Users/athil/Desktop/codes_trial/polarisation_image_analysis/plots/subsampling"
print(os.getcwd())

# cv2.imshow("col", img_mask_vis)
pwd = os.getcwd()



# cv2.imwrite("resample.jpg",resize)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# implot_func([rand_img, composite_img, sub_sample], ["sample img","mask", "sub sampled"], "Subsampling")   ###

dict_omm_rho = {"ant": 5.4, "honey bee": 14, "cricket": 35}
white = [255, 255, 255]
red = [255, 0, 0]
green = [0, 255, 0]
blue = [0, 0, 255]
black = [0, 0, 0]

img_holder = [None] * len(dict_omm_rho)


known_angle =1 #2*np.rad2deg(np.arctan(4.5/71))
conversion_pix_per_angle = 1  #384/known_angle
# sigma = fwhm/2.355
sig_fwhm_factor = 2.355
dict_data = np.array([i for i in dict_omm_rho.values()])
SIGMA = dict_data / sig_fwhm_factor * conversion_pix_per_angle

kernel_size = int(4 * max(SIGMA))
if kernel_size % 2 == 0:
    kernel_size += 1

for aoi, i in zip(dict_omm_rho.keys(), range(len(dict_omm_rho))):
    print(aoi, dict_omm_rho[aoi])
    img_holder[i] = cv2.GaussianBlur(rand_img, (kernel_size, kernel_size), SIGMA[i])


plt_images = img_holder
plt_images.insert(0, rand_img)
img_legends = list(dict_omm_rho)
img_legends.insert(0, "original")
# implot_func(plt_images, img_legends)    ####

# x, y = circular_index(rand_img, 8, 10)
# print(len(x))
# rand_img[(x, y)] = black
#
# plt.imshow(rand_img, cmap=plt.cm.get_cmap("gray"), vmax=255, vmin=0)
# plt.show()
centre = [blank_img.shape[0], blank_img.shape[1]]
#centre = [0, 0]
x, y = curve_indexing(blank_img, radius=10, num=50, outer_angle=360, inner_angle=0)
col_list = np.array([red, green, blue])
blank_img[(x, y)] = col_list[np.random.randint(col_list.shape[0])]
trail_x = int(np.ceil(np.sqrt(len(x)))) - len(x)
trail_y = int(np.ceil(np.sqrt(len(y)))) - len(y)

# img_remaped = blank_img([x,y])
# np.reshape(img_remaped,(int(np.floor(img_remaped.shape[0])),int(np.floor(img_remaped.shape[0]))))
RAD = np.arange(5,10,2)
X, Y = [None]*len(RAD), [None]*len(RAD)
for radius, i in enumerate(RAD):
    x,y = curve_indexing(blank_img, radius=radius, num=10)
    # X[i]= x             #?
    print(X)
# plt.imshow(blank_img)
# # plt.scatter(x, y)
# # plt.xlim([0, blank_img.shape[0]])
# # plt.ylim([0, blank_img.shape[1]])
# plt.show()
# plt.subplot(121)
# plt.title("mask")
# plt.imshow(img_mask_vis)
# plt.subplot(122)
# plt.title("sub sampled")
# plt.imshow(sub_sample)
# plt.show()

