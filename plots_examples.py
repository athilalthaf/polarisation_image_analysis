# from bee_eye_subsampling import *
import matplotlib.pyplot as plt
import numpy as np

from lib_importer import *
from pol_img_functions import  azimuth_mapping,elevation_mapping,pixel_map_func,pol_2_equirect,gauss_filter_function,image_tile_function


blend_img_high = cv2.imread("test_img_voronoi_image_high_res.png")
blend_img_high = blend_img_high[:,:,[2,1,0]]
blend_img_low = cv2.imread("test_img_voronoi_image_low_res.png")
blend_img_low = blend_img_low[:,:,[2,1,0]]
blend_img_med = cv2.imread("test_img_voronoi_image_half_res.png")
blend_img_med = blend_img_med[:,:,[2,1,0]]


auorora_img = cv2.imread("The_Research Station_3_James_Foster.jpeg")
auorora_img = auorora_img[:,:,[2,1,0]]
"""
                image
"""

# centre = [54, 54]                       ### low_res
# radius = 51
# thresh = 1

# centre = [540, 540]                       ### high_res
# radius = 503
# thresh = .05


# centre = [auorora_img.shape[0]//2, auorora_img.shape[1]//2]                       ### aurora_im
# radius = auorora_img.shape[0]//2-712
# thresh = .05



centre = [270, 270]                       ### med_res
radius = 250
thresh = .1



azimuth_map = azimuth_mapping(src= blend_img_med, radius=radius, centre= centre)
elevation_map_src, elevation_map_corr = elevation_mapping(src=blend_img_med,radius=radius,centre=centre)
mapped_img = pixel_map_func(src=blend_img_med, centre=centre, radius=radius, elevation_map_src=elevation_map_src,
                            elevation_map_corr=elevation_map_corr,
                            azimuth_map=azimuth_map,thresh=thresh)

equi_plot_before = pol_2_equirect(src=blend_img_med, radius=radius, centre=centre,inner_angle=180,outer_angle=180+ 360)
equi_plot_after = pol_2_equirect(src=mapped_img, radius=radius, centre=centre,inner_angle=180,outer_angle=180+360)
"""
                    input image vis
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
# plt.imshow(equi_plot_before)
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
# # plt.imsave("calib_img_subsampled.png",calib_img_subsampled )
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
                    kernels
"""
# sigma = 5
# kern_size = 4 * sigma + 1
# kernel = gauss_filter_function(sigma=sigma,kern_size_x=kern_size)
#
# fig1, ax1 = plt.subplots()
#
# ax1.set_title("simple gaussian kernel , kernel_size:{}, sigma :{}".format(kern_size,sigma))
# plt1 = plt.imshow(kernel)
# cbar = fig1.colorbar(plt1, ax=ax1, label ="weights")
# # cbar.set_ticklabels(np.ceil(np.rad2deg(ele_range)))
# plt.xlabel("pixel cordinates")
# plt.ylabel("pixel cordinates")
# plt.tight_layout()
# # plt.imsave("calib_img_subsampled.png",calib_img_subsampled )
# plt.savefig("simple_gauss_kernel.png", dpi = 300,bbox_inches = "tight")
#
# plt.show()

"""
                    tile functions
"""

side_tile = equi_plot_after.shape[1]//8
top_tile = equi_plot_after.shape[0]//3
tiled_im = image_tile_function(equi_plot_after,side_tile,top_tile)

fig1, ax1 = plt.subplots()
ax1.set_title("sample tiling function in action , side_tile:{}$^\circ$, top_tile :{}$^\circ$"
              .format(int(np.ceil(side_tile/equi_plot_after.shape[1]*360)), int(np.ceil(top_tile/equi_plot_after.shape[0]*90))))
azi_ticks = np.linspace(0, tiled_im.shape[1],11)
azi_ticks_labels = np.concatenate([[360-45],np.linspace(0,360,9),[45]])

ele_ticks = np.linspace(0, tiled_im.shape[0],5)
ele_ticks_labels = np.concatenate([[-30],np.linspace(0,90,4)])


plt.xticks(azi_ticks)
ax1.set_xticklabels(azi_ticks_labels)
plt.yticks(ele_ticks)
ax1.set_yticklabels(ele_ticks_labels)


plt1 = plt.imshow(tiled_im)


plt.xlabel("azimuth in ($^\circ$)")
plt.ylabel("elevation in ($^\circ$)")
ax1.set_aspect(tiled_im.shape[1]/tiled_im.shape[0])
plt.tight_layout()
# plt.imsave("calib_img_subsampled.png",calib_img_subsampled )
plt.savefig("tiled_img_sample.png", dpi = 300,bbox_inches = "tight")
plt.show()
