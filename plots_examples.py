
"""
Plotting different functions in pol_img_functions.
separated by sections.

uncomment sections to check different functions and its plots.

loads different test images from
"""
import scipy.signal

from lib_importer import * #import base libraries
from pol_img_functions import  azimuth_mapping,elevation_mapping,pixel_map_func,pol_2_equirect,gauss_kernel,\
    image_tile_function,  azimuth_to_idx,elevation_to_idx   # import relevant functions




# blend_img_high = cv2.imread("test_images/test_img_voronoi_image_high_res.png")   ## high resolution image
# blend_img_high = blend_img_high[:,:,[2,1,0]]
blend_img_low = cv2.imread("test_images/test_img_voronoi_image_low_res.png")  # load image different resolutions
blend_img_low = blend_img_low[:,:,[2,1,0]] # re-order color channels from BGR to RGB

# blend_img_med = cv2.imread("test_images/test_img_voronoi_image_half_res.png")
# blend_img_med = blend_img_med[:,:,[2,1,0]]


"""
         centre and radius values of different test images
"""

centre = [54, 54]                       ### blend_img_low
radius = 51
thresh = 1

# centre = [270, 270]                       ### blend_img_med
# radius = 250
# thresh = .1

# centre = [540, 540]                       ### blend_img_high
# radius = 503
# thresh = .05






#
azimuth_map = azimuth_mapping(src= blend_img_low, radius=radius, centre= centre) # map the azimuth of the image
elevation_map_src, elevation_map_corr = elevation_mapping(src=blend_img_low,radius=radius,centre=centre) # elevation map of the image original and corrected version
mapped_img = pixel_map_func(src=blend_img_low, centre=centre, radius=radius, elevation_map_src=elevation_map_src,
                            elevation_map_corr=elevation_map_corr,
                             azimuth_map=azimuth_map,thresh=thresh) # correct the image for elevation distortion correction
#
equi_plot_before = pol_2_equirect(src=blend_img_low, radius=radius, centre=centre,inner_angle=180,outer_angle=180+ 360) # equirectangular projection of input image
equi_plot_after = pol_2_equirect(src=mapped_img, radius=radius, centre=centre,inner_angle=180,outer_angle=180+360) # equirectangular projection of corrected image
"""
                    input image vis || plot the loaded image

"""
#
# fig1, ax1 = plt.subplots()
#
# ax1.set_title("Image with distortion")
# # pixel_map= pixel_map_func(calib_img, radius = 258, elevation_map_src=elevation_map_src, elevation_map_corr=elevation_map_corr, azimuth_map=azi_map)
# #ele_range = np.deg2rad(np.arange(0, 91, 30))
# plt.imshow(blend_img_med)
# # cbar = fig13.colorbar(mat10, ax=ax13, ticks=ele_range, label ="angles")
# # cbar.set_ticklabels(np.ceil(np.rad2deg(ele_range)))
# plt.xlabel("pixels")
# plt.ylabel("pixels")
# plt.tight_layout()
# # plt.imsave("calib_img_subsampled.png",calib_img_subsampled )
# plt.savefig("blend_med_fig.png", dpi = 300,bbox_inches = "tight")
# plt.show()


"""
                    mapped image
"""

# fig1, ax1 = plt.subplots()
#
# ax1.set_title("corrected image with pixel map function, azi_thresh:{}$^\circ$".format(thresh))
# # pixel_map= pixel_map_func(calib_img, radius = 258, elevation_map_src=elevation_map_src, elevation_map_corr=elevation_map_corr, azimuth_map=azi_map)
# ele_range = np.deg2rad(np.arange(0, 91, 30))
# plt.imshow(mapped_img)
# # cbar = fig13.colorbar(mat10, ax=ax13, ticks=ele_range, label ="angles")
# # cbar.set_ticklabels(np.ceil(np.rad2deg(ele_range)))
# plt.xlabel("pixels")
# plt.ylabel("pixels")
# plt.tight_layout()
# # plt.imsave("calib_img_subsampled.png",calib_img_subsampled )
# plt.savefig("blend_high_mapped.png", dpi = 300,bbox_inches = "tight")
# plt.imsave("blend_high_mapped_im.png",mapped_img)
# plt.show()

"""
                    equirect projection
"""

# fig1, ax1 = plt.subplots()
#
# ax1.set_title("Equirect projection of original image")
# plt.imshow(equi_plot_after)
# plt.xticks(np.linspace(0,int(2 * np.pi * radius),9))
# ax1.set_xticklabels(np.linspace(0,360,9,dtype="int"))
#
# plt.yticks(np.linspace(0, radius, 4))
# ax1.set_yticklabels(np.linspace(90, 0, 4,dtype="int"))
#
# xtick_levels =np.linspace(0,int(2 * np.pi * radius)-1,9,dtype=int)
# ytick_levels =np.linspace(0,radius-1, 4,dtype=int)
#
# sample_img_med = [255, 255, 255] * np.ones(equi_plot_after.shape)
# sample_img_med[ytick_levels, :] = [0, 0, 0]
# sample_img_med[:, xtick_levels] = [0, 0, 0]
#
# # cbar = fig13.colorbar(mat10, ax=ax13, ticks=ele_range, label ="angles")
# # cbar.set_ticklabels(np.ceil(np.rad2deg(ele_range)))
# plt.xlabel("azimuth in degrees ($^\circ$)")
# plt.ylabel("elevation in degrees ($^\circ$)")
#
# plt.tight_layout()
# plt.imsave("equirect_test_mid_res_im.png",equi_plot_after )
# ax1.set_aspect(4)
# plt.savefig("aurora_equi_before_800.png", dpi = 300, bbox_inches="tight")
# plt.show()

"""
                    validation image
"""

# fig1, ax1 = plt.subplots()
#
# ytick_levels =np.linspace(0,equi_plot_after.shape[0], 4,dtype=int)
# ytick_levels[-1] = ytick_levels[-1] - 1
# xtick_levels =np.linspace(0,equi_plot_after.shape[1] -1,9,dtype=int)
# xtick_levels[-1] = xtick_levels[-1] - 1
#
# sample_img_med = [0, 0, 0] * np.ones(equi_plot_after.shape)
# sample_img_med[ytick_levels, :] = [255, 255, 255]
# sample_img_med[:, xtick_levels] = [255, 255, 255]
#
#
# ax1.set_title("caliberated image expected longitude and latitutes ")
# plt.imshow(sample_img_med)
# # plt.imshow(sample_img_med,alpha=0.5)
# plt.xticks(np.linspace(0,int(2 * np.pi * radius),9))
# ax1.set_xticklabels(np.linspace(0, 360, 9, dtype="int"))
#
# plt.yticks(np.linspace(0, radius, 4))
# ax1.set_yticklabels(np.linspace(90, 0, 4,dtype="int"))
#
#
# # cbar = fig13.colorbar(mat10, ax=ax13, ticks=ele_range, label ="angles")
# # cbar.set_ticklabels(np.ceil(np.rad2deg(ele_range)))
# plt.xlabel("azimuth in degrees ($^\circ$)")
# plt.ylabel("elevation in degrees ($^\circ$)")
#
# plt.tight_layout()
# # plt.imsave("calib_img_subsampled.png",calib_img_subsampled )
# ax1.set_aspect(4)
# plt.savefig("valid_img_med_black.png", dpi = 300, bbox_inches="tight")
# plt.show()

"""
                    kernel visualisation 
"""
# sigma = 5
# kern_size = 4 * sigma + 1
# ele_val = 0
# kernel = gauss_kernel(sigma=sigma,ele_val=ele_val,kern_size_x=kern_size)
#
# fig1, ax1 = plt.subplots()
#
# ax1.set_title("simple gaussian kernel , elevation :{}$^\circ$".format(ele_val))
# plt1 = plt.imshow(kernel)
# cbar = fig1.colorbar(plt1, ax=ax1, label ="weights")
# # cbar.set_ticklabels(np.ceil(np.rad2deg(ele_range)))
# plt.xlabel("pixel cordinates")
# plt.ylabel("pixel cordinates")
# plt.tight_layout()
# # plt.imsave("calib_img_subsampled.png",calib_img_subsampled )
# plt.savefig("simple_gauss_kernel_0.png", dpi = 300,bbox_inches = "tight")
#
# plt.show()

"""
                    tile functions
"""

# side_tile = equi_plot_before.shape[1]//8
# top_tile = equi_plot_before.shape[0]//3
# tiled_im = image_tile_function(equi_plot_after,side_tile,top_tile)
#
# fig1, ax1 = plt.subplots()
# ax1.set_title("sample tiling function in action , side_tile:{}$^\circ$, top_tile :{}$^\circ$"
#               .format(int(np.ceil(side_tile/equi_plot_after.shape[1]*360)), int(np.ceil(top_tile/equi_plot_after.shape[0]*90))))
# azi_ticks = np.linspace(0, tiled_im.shape[1],11)
# azi_ticks_labels = np.concatenate([[360-45],np.linspace(0,360,9,dtype=int),[45]])
#
# ele_ticks = np.linspace(0, tiled_im.shape[0],5)
# ele_ticks_labels = np.concatenate([[60],np.linspace(0,90,4,dtype=int)[::-1]])
#
#
# plt.xticks(azi_ticks)
# ax1.set_xticklabels(azi_ticks_labels)
# plt.yticks(ele_ticks)
# ax1.set_yticklabels(ele_ticks_labels)
#
#
# plt1 = plt.imshow(tiled_im)
#
#
# plt.xlabel("azimuth in ($^\circ$)")
# plt.ylabel("elevation in ($^\circ$)")
# ax1.set_aspect(tiled_im.shape[1]/tiled_im.shape[0])
# # plt.tight_layout()
# plt.imsave("tiled_img_sample_im_01.png",tiled_im)
# plt.savefig("tiled_img_sample_01.png", dpi = 300,bbox_inches = "tight")
# plt.show()

"""
                    test image for convolution
"""
# sigma = 5
# kern_size = 4 * sigma + 1
# kernel = gauss_kernel(sigma=sigma, kern_size_x=kern_size)
#
# dirac_delta = np.zeros((*kernel.shape, 3),dtype="uint8")
# dirac_delta[(kernel.shape[0]) // 2][[(kernel.shape[1]) // 2]] = [255, 255, 255]
# fig1, ax1 = plt.subplots()
# ax1.set_title("test image to check convolution")
#
#
#
# plt1 = plt.imshow(dirac_delta)
#
#
# plt.xlabel("pixels")
# plt.ylabel("pixels")
# ax1.set_aspect(1)
# # plt.tight_layout()
# plt.imsave("test_dirac_img.png",dirac_delta)
# plt.savefig("test_dirac.png", dpi = 300,bbox_inches = "tight")
# plt.show()




"""
Archived code for convolution 
"""
# rand_img = np.random.randint(0, 255, (25,100, 3),dtype="uint8")
# convolved_img = np.zeros(rand_img.shape, dtype="uint8")
# dpp_ele = rand_img.shape[0] / 90          # elevation value per pixel in degrees
# ele_val = np.arange(rand_img.shape[0]) / dpp_ele  # range of elevation values feeding for the
# ele_val = ele_val[::-1]
# sigma = 1
# for c in range(equi_plot_after.shape[2]):                                      # for each  channel
# # c = 0
#     for i in range(rand_img.shape[0]):
#         kern = gauss_kernel(sigma=sigma, ele_val=ele_val[i], kern_size_x=kern_size)
#         # kern_hold[i] = kern
#         convolve_strip = sp.signal.convolve2d(rand_img[i:i+kern_size, :, c], kern, mode ="same")/kern.sum()
#         # a = equi_plot_after[i:i+kern_size,:,c].shape
#         # print(a)
#         convolved_img[i, :, c] = convolve_strip[0, :]
# sigma = 5
# kern_size = 4 * sigma + 1
# kernel = gauss_kernel(sigma=sigma, kern_size_x=kern_size,ele_val=0)
# kern_centre = (kern_size + 1)// 2
#
#   ####
# tiled_im = image_tile_function(equi_plot_after,azimuth_to_idx(equi_plot_after,45),elevation_to_idx(equi_plot_after,30))
# equi_plot_after = tiled_im  ###
# convolved_img = np.zeros(equi_plot_after.shape, dtype="uint8")
# dpp_ele = equi_plot_after.shape[0] / 90          # elevation value per pixel in degrees
# ele_val = np.arange(equi_plot_after.shape[0]) / dpp_ele  # range of elevation values feeding for the
# ele_val = ele_val[::-1]                                  # value will be reversed
# # convolve_strip = np.zeros((kern_size ,equi_plot_after.shape[1]), dtype="uint8")
# ele_val_tiled = ele_val[:(equi_plot_after.shape[0] - equi_plot_after.shape[0])] ###
# ele_val = np.hstack([ele_val_tiled[::-1],ele_val ])  ####
#
# kern_hold = (equi_plot_after.shape[0]) * [None]
#
# for c in range(equi_plot_after.shape[2]):                                      # for each  channel
# # c = 0
#     for i in range(equi_plot_after.shape[0] ):
#         kern = gauss_kernel(sigma=sigma, ele_val=ele_val[i], kern_size_x=kern_size)
#         kern_hold[i] = kern
#         convolve_strip = sp.signal.convolve2d(equi_plot_after[i:i+kern_size, :, c], kern, mode ="same")/kern.sum()
#         # a = equi_plot_after[i:i+kern_size,:,c].shape
#         # print(a)
#         convolved_img[i, :, c] = convolve_strip[0, :]
#
#
#
# fig1, ax1 = plt.subplots()
# ax1.set_title("Random image for comvolution sigma =1")
#
#
#
# plt1 = plt.imshow(rand_img)
# # azi_ticks = np.linspace(0, convolved_img.shape[1],11)
# azi_ticks = np.linspace(0, convolved_img.shape[1],9)
# # azi_ticks_labels = np.concatenate([[360-45],np.linspace(0,360,9,dtype=int),[45]])
# azi_ticks_labels = np.linspace(0,360,9,dtype=int)
#
# # ele_ticks = np.linspace(0, convolved_img.shape[0],5)
# ele_ticks = np.linspace(0, convolved_img.shape[0],4)
# # ele_ticks_labels = np.concatenate([[60],np.linspace(0,90,4,dtype=int)[::-1]])
# ele_ticks_labels = np.linspace(0,90,4,dtype=int)[::-1]
#
#
# plt.xticks(azi_ticks)
# ax1.set_xticklabels(azi_ticks_labels)
# plt.yticks(ele_ticks)
# ax1.set_yticklabels(ele_ticks_labels)
#
#
# plt.xlabel("azimuth in ($^\circ$)")
# plt.ylabel("elevation in ($^\circ$)")
# ax1.set_aspect(convolved_img.shape[1]/convolved_img.shape[0])
# plt.tight_layout()
# plt.imsave("random_img_sigma5_im.png",convolved_img)
# plt.savefig("random_img_sigma5.png", dpi = 300,bbox_inches = "tight")
# plt.show()