# from bee_eye_subsampling import *
from pol_img_functions import  azimuth_mapping,elevation_mapping,pixel_map_func


blend_img_high = cv2.imread("test_img_voronoi_image_high_res.png")
blend_img_high = blend_img_high[:,:,[2,1,0]]
blend_img_low = cv2.imread("test_img_voronoi_image_low_res.png")
blend_img_low = blend_img_low[:,:,[2,1,0]]

"""
blender low res image
"""

centre = [54, 54]
radius = 51
thresh = 1
azimuth_map = azimuth_mapping(src= blend_img_low, radius=radius, centre= centre)
elevation_map_src, elevation_map_corr = elevation_mapping(src=blend_img_low,radius=radius,centre=centre)
mapped_img = pixel_map_func(src=blend_img_low, centre=centre, radius=radius, elevation_map_src=elevation_map_src,
                            elevation_map_corr=elevation_map_corr,
                            azimuth_map=azimuth_map)


fig1, ax1 = plt.subplots()


ax1.set_title("corrected image with pixel map function, azi_thresh:{}$^\circ$".format(thresh))
# pixel_map= pixel_map_func(calib_img, radius = 258, elevation_map_src=elevation_map_src, elevation_map_corr=elevation_map_corr, azimuth_map=azi_map)
ele_range = np.deg2rad(np.arange(0, 91, 30))
plt.imshow(calib_img_subsampled_mapped)
# cbar = fig13.colorbar(mat10, ax=ax13, ticks=ele_range, label ="angles")
# cbar.set_ticklabels(np.ceil(np.rad2deg(ele_range)))
plt.xlabel("pixels")
plt.ylabel("pixels")
plt.tight_layout()
# plt.imsave("calib_img_subsampled.png",calib_img_subsampled )
plt.savefig("blend_low_mapped.png", dpi = 300)
plt.show()