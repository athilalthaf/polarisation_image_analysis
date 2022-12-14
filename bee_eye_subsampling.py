from lib_importer import *

# from matplotlib import rcParams


# rcParams["figure.dpi"] = 120
from pol_img_functions import implot_func
from pol_img_functions import pol_2_equirect,gauss_filter,elevation_mapping,azimuth_mapping,pixel_map_func,sub_sampling_func

alpha_max_1_2 = np.deg2rad(np.array([
    [
        -74.5, -69.7, -60.2, -50.2, -40.2, -30.3, -20.3, -10.2, 0.0,
        10.2, 20.2, 30.1, 39.9, 49.4, 58.9, 67.8, 69.0
    ], [
        270.0, 191.9, 176.8, 174.6, 173.4, 166.8, 160.5, 158.5, 156.7,
        158.5, 163.4, 174.3, 189.0, 202.4, 215.8, 244.7, 270.0
    ]
]))
alpha_max_3_4 = np.deg2rad(np.array([
    [
        -74.5, -67.8, -58.1, -48.4, -38.9, -29.4, -19.8, -9.9, 0.0,
        9.9, 19.8, 29.6, 39.4, 49.2, 58.9, 68.1, 69.0
    ], [
        -90.0, -63.5, -52.5, -44.1, -36.1, -26.8, -20.6, -14.6, -14.5, -14.4,
        -15.5, -21.2, -25.5, -31.6, -42.0, -70.7, -90.0
    ]
]))
tck_1_2 = splrep(alpha_max_1_2[0], alpha_max_1_2[1], s=0.01)
tck_3_4 = splrep(alpha_max_3_4[0], alpha_max_3_4[1], s=0.01)


def build_right_bee_eye():
    """
    Following [1]_

    Notes
    -----
    .. [1] "Mimicking honeybee eyes with a 280 degrees field of view catadioptric imaging system"
            http://iopscience.iop.org/article/10.1088/1748-3182/5/3/036002/pdf
    """
    Delta_phi_min_v = np.deg2rad(1.5)
    Delta_phi_max_v = np.deg2rad(4.5)
    Delta_phi_min_h = np.deg2rad(2.4)
    Delta_phi_mid_h = np.deg2rad(3.7)
    Delta_phi_max_h = np.deg2rad(4.6)

    def norm_alpha(alpha):
        alpha = alpha % (2 * np.pi)
        if alpha > 3 * np.pi / 2:
            alpha -= 2 * np.pi
        return alpha

    def norm_epsilon(epsilon):
        epsilon = epsilon % (2 * np.pi)
        if epsilon > np.pi:
            epsilon -= 2 * np.pi
        return epsilon

    def xi(z=None, alpha=None, epsilon=None):
        if alpha is not None:
            if z in [1, 2]:  # or 0 <= alpha <= 3 * np.pi / 2:
                return 1.
            if z in [3, 4]:  # or 3 * np.pi / 2 <= alpha:
                return -1.
        if epsilon is not None:
            if z in [1, 4]:  # or 0 <= epsilon <= np.pi / 2:
                return 1.
            if z in [2, 3]:  # or 3 * np.pi / 2 <= epsilon:
                return -1.
        return 0.

    def Delta_alpha(Delta_phi_h, epsilon):
        return norm_alpha((2 * np.arcsin(np.sin(Delta_phi_h / 2.) / np.cos(epsilon))) % (2 * np.pi))

    def phi_h(z, alpha, epsilon):
        alpha = norm_alpha(alpha)
        epsilon = norm_epsilon(epsilon)
        abse = np.absolute(epsilon)
        absa = np.absolute(alpha)

        # zone Z=1/2
        if z in [1, 2]:
            if 0 <= alpha <= np.pi / 4:
                return Delta_phi_mid_h + alpha / (np.pi / 4) * (Delta_phi_min_h - Delta_phi_mid_h)
            if np.pi / 4 < alpha <= np.pi / 2:
                return Delta_phi_min_h + (alpha - np.pi / 4) / (np.pi / 4) * (Delta_phi_mid_h - Delta_phi_min_h)
            if np.pi / 2 < alpha <= np.deg2rad(150) and abse <= np.deg2rad(50):
                return Delta_phi_mid_h + (alpha - np.pi / 2) / (np.pi / 3) * (Delta_phi_max_h - Delta_phi_mid_h)
            if np.pi / 2 < alpha <= np.deg2rad(150) and abse > np.deg2rad(50):
                return Delta_phi_mid_h + (alpha - np.pi / 2) / (np.pi / 3) * (Delta_phi_max_h - Delta_phi_mid_h) \
                                             * (np.pi / 2 - abse) / np.deg2rad(40)
            if np.deg2rad(150) < alpha <= np.pi and abse <= np.deg2rad(50):
                return Delta_phi_max_h
            if np.deg2rad(150) < alpha <= np.pi and abse > np.deg2rad(50):
                return Delta_phi_mid_h + (Delta_phi_max_h - Delta_phi_mid_h) * (np.pi / 2 - abse) / np.deg2rad(40)
            if np.pi < alpha <= 3 * np.pi / 2:
                return Delta_phi_max_h

        if z in [3, 4]:
            # zone Z=3
            if -np.pi / 2 <= alpha < -np.pi / 4 and epsilon <= -np.deg2rad(50):
                return Delta_phi_mid_h + (absa - np.pi / 4) / (np.pi / 4) * (Delta_phi_max_h - Delta_phi_mid_h) \
                                         * (np.pi / 2 - abse) / np.deg2rad(40)
            if -np.pi / 4 <= alpha <= 0 and epsilon <= -np.deg2rad(50):
                return Delta_phi_mid_h
            if -np.pi / 4 < alpha <= 0 and -np.deg2rad(50) < epsilon <= 0:
                return Delta_phi_mid_h + absa / (np.pi / 4) * (Delta_phi_max_h - Delta_phi_mid_h) \
                                         * (np.deg2rad(50) - abse) / np.deg2rad(50)

            # zone Z=4
            if -np.pi / 2 <= alpha <= -np.pi / 4 and epsilon >= np.deg2rad(50):
                return Delta_phi_max_h + (Delta_phi_max_h - Delta_phi_mid_h) * (np.pi / 2 - abse) / np.deg2rad(40)
            if -np.pi / 4 <= alpha <= 0 and 0 < epsilon <= np.deg2rad(50):
                return Delta_phi_mid_h + absa / (np.pi / 4) * (Delta_phi_max_h - Delta_phi_mid_h)
            if -np.pi / 4 <= alpha <= 0 and epsilon >= np.deg2rad(50):
                return Delta_phi_mid_h + absa / (np.pi / 4) * (Delta_phi_max_h - Delta_phi_mid_h) \
                                         * (np.pi / 2 - epsilon) / np.deg2rad(40)
        return np.nan

    def alpha_max(z, epsilon):
        if z in [1, 2]:
            return splev(epsilon, tck_1_2, ext=3, der=0)
        if z in [3, 4]:
            return splev(epsilon, tck_3_4, ext=3, der=0)
        return np.nan

    ommatidia = []
    for z in range(1, 5):
        j = 0
        epsilon = 0.  # elevation
        alpha = 0.  # azimuth
        while np.absolute(epsilon) <= np.pi / 2:
            if j % 2 == 0:
                alpha = 0
            else:
                alpha = Delta_alpha(xi(z=z, alpha=alpha) * Delta_phi_mid_h / 2, epsilon)
            while np.absolute(alpha) < np.absolute(alpha_max(z, epsilon)):
                # print z, np.rad2deg(epsilon), np.rad2deg(alpha)
                Delta_phi_h = phi_h(z, alpha, epsilon)
                if np.isnan(Delta_phi_h):
                    break
                ommatidia.append(np.array([alpha, epsilon]))
                alpha = norm_alpha(alpha + Delta_alpha(xi(z=z, alpha=alpha) * Delta_phi_h, epsilon))
            Delta_phi_v = Delta_phi_min_v + (Delta_phi_max_v - Delta_phi_min_v) * np.absolute(epsilon) / (np.pi / 2)
            epsilon = norm_epsilon(epsilon + xi(z=z, epsilon=epsilon) * Delta_phi_v / 2.)
            j += 1

    ommatidia = np.array(ommatidia)
    ommatidia = ommatidia[ommatidia[:, 1] > np.pi/3]
    omm_ori = R.from_euler('ZY', ommatidia)
    omm_rho = np.deg2rad(5) * (1 + ((np.pi/2 - ommatidia[:, 1]) % (2 * np.pi)) / np.pi)
    omm_pol = np.asarray(ommatidia[:, 1] > np.pi/3, dtype=float)
    spectral = (omm_pol[..., np.newaxis] * np.array([[0, 0, 0, 0, 1]], dtype=float) +
                (1 - omm_pol)[..., np.newaxis] * np.array([[0, 0, 1, 0, 0]], dtype=float))

    return omm_ori, omm_rho, omm_pol, spectral, ommatidia

r_eye_ori, r_eye_rho, r_eye_pol, r_eye_spectral, ommatidia = build_right_bee_eye()
print(r_eye_ori)
omm_pol = ommatidia[ommatidia[:, 1] > np.pi / 3]



hue = r_eye_spectral
rgb = hue[..., 1:4]
rgb[:, [0, 2]] += hue[..., 4:5] / 2
rgb[:, 0] += hue[..., 0]
# plt.subplot(111, polar=False)
yaw, pitch, raw = r_eye_ori.as_euler('ZYX', degrees=True).T
# yaw_norm = (yaw - yaw.min()) / np.max(yaw - yaw.min())
# pitch_norm = (pitch - pitch.min()) / np.max(pitch - pitch.min())
# plt.figure(1)
# ax1 = plt.subplot(111,polar=False)
#
# plt.scatter(yaw, pitch, s=20, c=np.clip(rgb, 0, 1))
# plt.ylabel("elevation [deg]")
# plt.xlabel("azimuth [deg]")
# plt.title("DRA co-ordinates")
# plt.xlim([-180, 180])
# plt.ylim([0, 90])
# plt.xticks(np.arange(-180,181,45))
# plt.yticks(np.arange(0,91,30))
# plt.tight_layout()
#
# plt.show()
# plt.show()


test_img = cv2.imread("test_img.png")
# yaw_img_cord = np.array(yaw_norm * (test_img.shape[0]- 1), dtype="int")
# pitch_img_cord = np.array(pitch_norm * (test_img.shape[1] - 1), dtype="int")


test_img_vis = test_img
# test_img_vis[(yaw_img_cord, pitch_img_cord)] = [255, 0, 0]
# plt.imshow(test_img_vis)
# plt.scatter(yaw_img_cord,pitch_img_cord)
# # plt.xlim([0,test_img.shape[0]])
# # plt.ylim([test_img.shape[1], 0])
# plt.show()


#simplifyingh the image into a matirx to perform convolution
# dist_array = np.zeros((yaw_img_cord.shape))
# yaw_pitch = np.vstack([yaw_img_cord, pitch_img_cord]).T
#
# for i in range(yaw_img_cord.shape[0]):
#     dist_array[i] = np.linalg.norm(np.linalg.norm(yaw_pitch[i,:] - np.array([0,0])))
#
# sort_idx = np.argsort(dist_array)
#
# yaw_pitch_sort = yaw_pitch[sort_idx, :]

# yaw_pitch_resize = np.zeros((200,200,3))
# yaw_pitch_resize[0,0] = test_img[yaw_pitch_sort[0, :]]
# for i in range(yaw_pitch_resize.shape[0]):
#     for j in range(yaw_pitch_resize.shape[1]):
#             yaw_


# nan_img = np.empty(test_img.shape)
# nan_img[:] = np.nan
# nan_img[(yaw_img_cord, pitch_img_cord)] = test_img[(yaw_img_cord, pitch_img_cord)]
##########
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

####

kernel_size = int(4 * max(SIGMA))
if kernel_size % 2 == 0:
    kernel_size += 1



# for aoi, i in zip(dict_omm_rho.keys(), range(len(dict_omm_rho))):
#     print(aoi, dict_omm_rho[aoi])
#     img_holder[i] = cv2.GaussianBlur(nan_img, (kernel_size, kernel_size), SIGMA[i])


# plt_images = img_holder
# #######
# nan_img_zero = nan_img.copy()
# nan_img_zero[np.isnan(nan_img)] = 0
# # nan_img_zero_blurr = cv2.GaussianBlur(nan_img_zero, (kernel_size, kernel_size), SIGMA[0])
# #
# nan_img_one = 0 * nan_img_zero.copy() + 1
# nan_img_one[np.isnan(nan_img)] = 0
# nan_img_one_blurr = cv2.GaussianBlur(nan_img_one, (kernel_size, kernel_size), SIGMA[0])

# NAN_img_blurr = nan_img_zero_blurr/nan_img_one_blurr

# plt.imshow(nan_img)
# plt.show()

# fig1, ax1 = plt.subplots(dpi=120, constrained_layout=True)
# ax1.imshow(nan_img, cmap="inferno")
# plt.title("DRA pattern")
# plt.show()
#
# fig2 = implot_func(img_holder,list(dict_omm_rho))
index = [-2,-2]
# idx = (yaw_img_cord[index[0]],pitch_img_cord[index[1]])
#
# a = np.zeros(test_img.shape)
# a[(yaw_img_cord,pitch_img_cord)] = red

# fig2, ax2 = plt.subplots(1,2,dpi=200)
# ax2[0].scatter(pitch_img_cord,yaw_img_cord)
#
# ax2[0].set(xlim=(0,1000),ylim=(1000,0))
# ax2[0].set_aspect("equal","box")
# ax2[1].imshow(a)
#
# ax2[1].set(xlim=(0,1000),ylim=(1000,0))
# ax2[1].set_aspect("equal", "box")
# plt.show()
#
# plt.subplot(111)
# plt.imshow(test_img_vis)
# plt.scatter(pitch_img_cord, yaw_img_cord)
# plt.show()
#

# def spiral_mat(cord_list):     #not implemented properly
#
#     def up(x_idx, y_idx):
#         x_idx = x_idx
#         y_idx -= 1
#         return x_idx, y_idx
#
#     def left(x_idx, y_idx):
#         x_idx -= 1
#         y_idx =  y_idx
#         return x_idx, y_idx
#
#     def down(x_idx, y_idx):
#         x_idx = x_idx
#         y_idx += 1
#         return x_idx, y_idx
#
#     def right(x_idx, y_idx):
#         x_idx += 1
#         y_idx = y_idx
#         return x_idx, y_idx
#
#     mat_dim = int(np.ceil(np.sqrt(cord_list.shape[0])))
#     if mat_dim % 2 != 0:
#         mat_dim += 1
#
#     spr_mat = np.zeros((mat_dim, mat_dim, cord_list.shape[1]))
#     spr_mat[:] = np.nan
#     centre = int(np.ceil(mat_dim/2))
#     spr_mat[centre, centre] = cord_list[0]       ###y_img_cord is not in the porper order ,do it before img_indx normalize
#
#     rep_counter = 1
#     idx_holder = [centre, centre]
#     # for i in range(len(yaw_img_cord) - 1):
#     #     up(idx_holder[0], idx_holder[1])
#     #     spiral_mat[]
#
#
#     return spr_mat, mat_dim


# a = spiral_mat(test_img[(yaw_img_cord, pitch_img_cord)])
# print(a)

# plt.subplot(111)
# p_rev = pitch_img_cord[-1::]
# y_rev = yaw_img_cord[-1::]
# for i in range(yaw_img_cord.shape[0]):
#     plt.scatter(pitch_img_cord[i], yaw_img_cord[i])
#     plt.pause(0.001)


# plt.subplot(111)
#
# blank_img = np.zeros(test_img.shape,dtype = "int8")
# blank_img[:][:] = red
# blank_img[(yaw_img_cord, pitch_img_cord)] = test_img[(yaw_img_cord, pitch_img_cord)]
# plt.imshow(blank_img)
# plt.show()

# print(test_img[(yaw_img_cord, pitch_img_cord)])

# fig4, ax4 = plt.subplots(dpi=120,subplot_kw=dict(projection="polar"),constrained_layout=True)
# ax4.set_yticks([np.deg2rad(30), np.deg2rad(60), np.deg2rad(90)])
# ax4.yaxis.set_ticklabels([])
# ax4.tick_params(grid_color="k", grid_linewidth = 3, grid_linestyle = "--")
# ax4.xaxis.set_ticklabels([])
# ax4.set_aspect(1)
# # plt.ylim([90, 0])
# plt.show()


calib_img = cv2.imread("dummy_caliberated_img_crop.png")

radius = 281 # pixel units
ang_conv = radius/90   # pixels to angles

centre = (290, 290)
n_samples = 360
polar_img = np.zeros((radius, n_samples, 3))

polar = pol_2_equirect(calib_img, radius=radius, num=360, centre=centre)

lim_lab_list = [["pixels", "pixels"], ["deg", "pixels"]]
# implot_func([calib_img,polar],["Calibration image","Projection along radius"],lim_lab_list=lim_lab_list)

# plt.subplot(121)
#
# plt.imshow(calib_img)
# plt.title("Calibrated image")
# plt.xlabel("pixels")
# plt.ylabel("pixels")
#
# plt.subplot(122)
# plt.imshow(polar)
# plt.xticks(np.arange(0,361,45))
# plt.title("Radial unwrap")
# plt.xlabel("azimuth [deg]")
# plt.ylabel("radial distance pxls")
#
# plt.tight_layout()
# plt.show()
#
# ax6 = plt.subplot(111)
# plt.imshow(polar)
# plt.yticks([0 , int(radius/3), int(radius*2/3), radius ])
# ax6.set_yticklabels(np.arange(0,91,30)[::-1])
# plt.xticks(np.arange(0,361,45))
# plt.xlabel("azimuth [deg]")
# plt.ylabel("elevation [deg]")
# plt.title("Radial unwrap")
# plt.tight_layout()
# plt.show()



equi_rect_sample, azi_min,azi_max = pol_2_equirect(calib_img, radius, centre=centre, outer_angle=180, inner_angle=-180)
##

cnum = equi_rect_sample.shape[2]
azi = np.linspace(azi_min,azi_max,equi_rect_sample.shape[0])
ele = np.linspace(90,0,radius)
dpp_azi = np.median(np.abs(np.diff(azi)))
dpp_ele = np.median(np.abs(np.diff(ele)))
max_val = equi_rect_sample.max()
sigma_deg = dict_omm_rho["honey bee"]
kern_azi = np.arange(np.floor(-3 * sigma_deg),np.ceil(3 * sigma_deg), dpp_azi)
#
# for e in range(90):
#     ele_correction = 1/np.cos(np.deg2rad(ele[e]))
#
#     kern_ele = np.linspace(max(-90,-3 * ele_correction * sigma_deg), np.ceil(min(90, 3*ele_correction * sigma_deg), dpp_ele))
#     KA , KE = np.meshgrid(kern_azi, kern_ele)
#
#     gk = gauss_filter(KA,KE, ele_correction,sigma_deg)
#     gk = gk.T/sum(gk)


# plt.figure(2)
# ax7 = plt.subplot(111)
# plt.imshow(equi_rect_sample)
# plt.xticks(np.arange(0, 361, 45))
# plt.yticks(np.arange(0, 91, 30) * ang_conv)
# # plt.yticks([0 , int(radius/3), int(radius*2/3), radius ])
# ax7.set_yticklabels(np.arange(0,91,30)[::-1])
# plt.xticks(np.arange(0,361,45))
# ax7.set_xticklabels(np.arange(-180,181,45))
# plt.xlabel("azimuth [deg]")
# plt.ylabel("elevation [deg]")
# plt.title("Radial unwrap ")
# plt.tight_layout()
# plt.show()
#
#
# ax8 = plt.subplot(111)
# plt.imshow(polar_invert)
# plt.xticks(np.arange(0,361,45))
# plt.yticks(np.arange(0,91,30) * ang_conv)
# # plt.yticks([0 , int(radius/3), int(radius*2/3), radius ])
# ax8.set_yticklabels(np.arange(0,91,30)[::-1])
# plt.xticks(np.arange(0,361,45))
# ax8.set_xticklabels(np.arange(-180,181,45))  # changing the xlabels
#
# plt.xlabel("azimuth [deg]")
# plt.ylabel("elevation [deg]")
# plt.title("Radial unwrap with azimuthal correction")
# plt.tight_layout()
# plt.show()
aurora_img = cv2.imread("The_Research Station_3_James_Foster.jpeg")
aurora_img = aurora_img
aurora_img_rad = int(4640/2)

# aurora_img_equi = pol_2_equirect(aurora_img, centre=None, radius=aurora_img_rad-50, outer_angle=-180, inner_angle=180)
# plt.figure(3)
# ax7 = plt.subplot(111)
# plt.imshow(aurora_img_equi)
# plt.xticks(np.arange(0, 361, 45))
# plt.yticks(np.arange(0, 91, 30) * ang_conv)
# # plt.yticks([0 , int(radius/3), int(radius*2/3), radius ])
# ax7.set_yticklabels(np.arange(0,91,30)[::-1])
# plt.xticks(np.arange(0,361,45))
# ax7.set_xticklabels(np.arange(-180,181,45))
# plt.xlabel("azimuth [deg]")
# plt.ylabel("elevation [deg]")
# plt.title("Radial unwrap ")
# plt.tight_layout()
# plt.show()

dist_img = cv2.imread("test_img_voronoi_image_low_res.png")
# plt.figure(4)
# angs = np.linspace(np.pi/2, 0,100)
# plt.plot(np.cos(angs), angs, label = "distorted")
# plt.plot(np.linspace(0,1,100),angs,label = "corrected")
#
#
# plt.legend()
# plt.xlim([0,1])
# plt.ylim([0,np.pi/2])
#
# plt.yticks(ticks=np.linspace(np.pi/2, 0, 4), labels=np.arange(90, -1, -30))
# plt.xticks(ticks = np.linspace(0, 1, 4))
# plt.title("Distortion and mapping")
# plt.xlabel("Relative position from centre")
# plt.ylabel("Elevation [deg] ")
# plt.tight_layout()
# plt.show()
# #
# #
# polar = p(calib_img, radius=radius, num=360, centre=centre)
blend_dist_img = cv2.imread("test_img_voronoi_image_low_res.png")

# fig6, ax6 = plt.subplots()
# ele_map, ele_map_corr = spherical_distort_correction(blend_dist_img, radius=258)
# mat = ax6.imshow(ele_map)
# ax6.set_title("Elevation map correction")
# cbar = fig6.colorbar(mat, ax=ax6, ticks=[0, np.pi/6, np.pi/3, np.pi/2])
# cbar.set_ticklabels(np.ceil(np.rad2deg([0, np.pi/6, np.pi/3, np.pi/2])))
# plt.show()

#
# fig7, ax7 = plt.subplots()
# azi_map = azimuth_mapping(calib_img, radius=258)
# mat = ax7.imshow(azi_map)
# ax7.set_title("Azimuth values")
# azi_range = np.deg2rad(np.arange(0, 361, 45))
# cbar = fig7.colorbar(mat, ax=ax7, ticks=azi_range)
# cbar.set_ticklabels(np.ceil(np.rad2deg(azi_range)))
#
# plt.show()

# fig8, ax8 = plt.subplots()
# azi_map = azimuth_mapping(calib_img, radius=258)
# elevation_map_src, elevation_map_corr = spherical_distort_correction(calib_img, 258)
# ax8.set_title("Pixel map function")
# pixel_map= pixel_map_func(calib_img,radius = 258, elevation_map_src=elevation_map_src, elevation_map_corr=elevation_map_corr, azimuth_map=azi_map)
# # azi_range = np.deg2rad(np.arange(0, 361, 45))
# # cbar = fig8.colorbar(mat, ax=ax7, ticks=azi_range)
# # cbar.set_ticklabels(np.ceil(np.rad2deg(azi_range)))
# plt.imshow(pixel_map)
# plt.show()

fig8, ax8 = plt.subplots()
equi, angel, radiu_inrts = pol_2_equirect(calib_img,radius=258,num=1000)
elevation_map_src, elevation_map_corr = elevation_mapping(calib_img, 258)
ax8.set_title("Equirect projection of subsample image ")

# pixel_map= pixel_map_func(calib_img, radius = 258, elevation_map_src=elevation_map_src, elevation_map_corr=elevation_map_corr, azimuth_map=azi_map)
# azi_range = np.deg2rad(np.arange(0, 361, 45))
# cbar = fig8.colorbar(mat, ax=ax7, ticks=azi_range)
# cbar.set_ticklabels(np.ceil(np.rad2deg(azi_range)))
plt.imshow(equi)
plt.imsave("equi_rect.png", equi.astype("uint8"))
plt.show()


                                                                            ## ------subsampled image testing
# fig9, ax9 = plt.subplots()
calib_img_subsampled,rm= sub_sampling_func(calib_img,100,1)
# ax9.set_title("subsampled image")
# # pixel_map= pixel_map_func(calib_img, radius = 258, elevation_map_src=elevation_map_src, elevation_map_corr=elevation_map_corr, azimuth_map=azi_map)
# # azi_range = np.deg2rad(np.arange(0, 361, 45))
# # cbar = fig8.colorbar(mat, ax=ax7, ticks=azi_range)
# # cbar.set_ticklabels(np.ceil(np.rad2deg(azi_range)))
# plt.imshow(calib_img_subsampled)
# plt.xlabel("pixels")
# plt.ylabel("pixels")
# plt.tight_layout()
# plt.imsave("calib_img_subsampled.png",calib_img_subsampled)
# plt.savefig("calib_img_subsampled_fig.png",dpi = 300)
# plt.show()

sub_sampled_img_centre = [48, 48]
sub_sampled_img_radius = 47
calib_img_subsampled_azimap = azimuth_mapping(calib_img_subsampled,sub_sampled_img_radius,sub_sampled_img_centre,angle=np.pi)
calib_img_subsampled_elemap_src, calib_img_subsampled_elemap_corr = elevation_mapping(calib_img_subsampled,
                                                                                      sub_sampled_img_radius,
                                                                                      sub_sampled_img_centre)
thresh_calib_img = 1
calib_img_subsampled_mapped = pixel_map_func(calib_img_subsampled, sub_sampled_img_centre, sub_sampled_img_radius,calib_img_subsampled_elemap_src, calib_img_subsampled_elemap_corr,calib_img_subsampled_azimap,thresh=thresh_calib_img)

#
# fig10, ax10 = plt.subplots()
# ax10.set_title("azi_map of subsampled image")
# # pixel_map= pixel_map_func(calib_img, radius = 258, elevation_map_src=elevation_map_src, elevation_map_corr=elevation_map_corr, azimuth_map=azi_map)
# azi_range = np.arange(np.nanmin(np.rad2deg(calib_img_subsampled_azimap)),np.nanmax(np.rad2deg(calib_img_subsampled_azimap)),45)
# mat10 = plt.imshow(np.rad2deg(calib_img_subsampled_azimap))
# # cbar = fig10.colorbar(mat10, ax=ax10,label = "angles")
# cbar = fig10.colorbar(mat10, ax=ax10, ticks=azi_range,label = "angles")
# cbar.set_ticklabels(np.ceil(azi_range))
# plt.xlabel("pixel cordinate")
# plt.ylabel("pixel cordinate")
# plt.tight_layout()
# # plt.imsave("calib_img_subsampled.png",calib_img_subsampled )
# plt.savefig("calib_img_subsampled_azimap_fig.png", dpi = 300)
# plt.show()

# fig11, ax11 = plt.subplots()
#
#
# ax11.set_title("elevation map of subsampled image (src)")
# # pixel_map= pixel_map_func(calib_img, radius = 258, elevation_map_src=elevation_map_src, elevation_map_corr=elevation_map_corr, azimuth_map=azi_map)
# ele_range = np.deg2rad(np.arange(0, 91, 30))
# mat10 = plt.imshow(calib_img_subsampled_elemap_src)
# cbar = fig11.colorbar(mat10, ax=ax11, ticks=ele_range, label ="angles")
# cbar.set_ticklabels(np.ceil(np.rad2deg(ele_range)))
# plt.xlabel("pixel cordinate")
# plt.ylabel("pixel cordinate")
# plt.tight_layout()
# # plt.imsave("calib_img_subsampled.png",calib_img_subsampled )
# plt.savefig("calib_img_subsampled_elevationmapsrc.png", dpi = 300)
# plt.show()


# fig12, ax12 = plt.subplots()
#
#
# ax12.set_title("corrected elevation map of subsampled image")
# # pixel_map= pixel_map_func(calib_img, radius = 258, elevation_map_src=elevation_map_src, elevation_map_corr=elevation_map_corr, azimuth_map=azi_map)
# ele_range = np.deg2rad(np.arange(0, 91, 30))
# mat10 = plt.imshow(calib_img_subsampled_elemap_corr)
# cbar = fig12.colorbar(mat10, ax=ax12, ticks=ele_range, label ="angles")
# cbar.set_ticklabels(np.ceil(np.rad2deg(ele_range)))
# plt.xlabel("pixel cordinate")
# plt.ylabel("pixel cordinate")
# plt.tight_layout()
# # plt.imsave("calib_img_subsampled.png",calib_img_subsampled )
# plt.savefig("calib_img_subsampled_elevationmapcorr.png", dpi = 300)
# plt.show()

fig13, ax13 = plt.subplots()


ax13.set_title("corrected image with pixel map function, azi_thresh:{}$^\circ$".format(thresh_calib_img))
# pixel_map= pixel_map_func(calib_img, radius = 258, elevation_map_src=elevation_map_src, elevation_map_corr=elevation_map_corr, azimuth_map=azi_map)
ele_range = np.deg2rad(np.arange(0, 91, 30))
plt.imshow(calib_img_subsampled_mapped)
# cbar = fig13.colorbar(mat10, ax=ax13, ticks=ele_range, label ="angles")
# cbar.set_ticklabels(np.ceil(np.rad2deg(ele_range)))
plt.xlabel("pixels")
plt.ylabel("pixels")
plt.tight_layout()
# plt.imsave("calib_img_subsampled.png",calib_img_subsampled )
plt.savefig("calib_img_subsampled_mapped.png", dpi = 300)
plt.show()

                                                                    #### blender lowres image test
blend_img_high = cv2.imread("test_img_voronoi_image_high_res.png")
blend_img_high = blend_img_high[:,:,[2,1,0]]
blend_img_low = cv2.imread("test_img_voronoi_image_low_res.png")
blend_img_low = blend_img_low[:,:,[2,1,0]]
centre = [blend_img_low.shape[0]//2 , blend_img_low.shape[1]//2 ]
radius = 51
blend_img_low_azi = azimuth_mapping(src=blend_img_low,radius=radius,centre=centre)
blend_img_low_ele_src, blend_img_low_ele_corr = elevation_mapping(blend_img_low,radius=radius,centre=centre)

mapped_img = pixel_map_func(src=blend_img_low,centre=centre,radius=radius,elevation_map_src=blend_img_low_ele_src,elevation_map_corr=blend_img_low_ele_corr,azimuth_map=blend_img_low_azi,thresh=.1)