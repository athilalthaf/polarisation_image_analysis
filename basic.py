# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import numpy as np
import cv2

scale_factor = .1
img_bgr = cv2.imread("/Users/athil/Downloads/sample_img.jpg")
rows, cols, channel = map(int, img_bgr.shape)
img_bgr_down_samp = cv2.pyrDown(img_bgr, dstsize = (cols // 2, rows // 2))
img_rgb = img_bgr_down_samp[:, :, [2, 1, 0]]
img_gray = cv2.cvtColor(img_bgr_down_samp, cv2.COLOR_BGR2GRAY)

dict_omm_rho = {"ant": 5.4, "honey bee": 14, "cricket": 35}

# for i,animal,rho in range(len(dict_omm_rho)),dict_omm_rho.items():
#     print(i)
    # subplot_idx = int("13"+ str(i+1))
    # plt.subplot(subplot_idx)
# plt.imshow(img_gauss_blurr)
# plt.title("Honey bee : " + str(dict_omm_rho["honey bee"]))
# plt.show()
known_angle = 2*np.rad2deg(np.arctan(4.5/71))
conversion_pix_per_angle = 384/known_angle
# sigma = fwhm/2.355
sig_fwhm_factor = 2.355


# plt.hist(img_gray)
# plt.imshow(img_rgb)
vals = img_rgb.mean(axis=2).flatten()
# b,bins,patches = plt.hist(vals,255)
# plt.title(aoi+" fov :" + "" + str(dict_omm_rho[aoi]))
# #plt.hist(img_gray[...])
# # plt.imshow(img_gray)
# plt.show()
legend = ["original"]
i = 0
hist_dat = []
hist_dat.append(img_rgb.mean(axis=2).flatten())
img_fig = plt.subplot(141)
img_fig.imshow(img_rgb)
plt.title("downsampled image")
plt.suptitle("Image processed under different acceptance angle")
plt.xticks([])
plt.yticks([])
dict_data = np.array([i for i in dict_omm_rho.values()])
SIGMA = dict_data / sig_fwhm_factor * conversion_pix_per_angle

kernel_size = int(4 * max(SIGMA))
if kernel_size % 2 == 0:
    kernel_size += 1

for aoi in dict_omm_rho.keys():
    sigma = dict_omm_rho[aoi] / sig_fwhm_factor * conversion_pix_per_angle

    img_gauss_blurr = cv2.GaussianBlur(img_rgb, (kernel_size, kernel_size), sigma)
    hist_dat.append(img_gauss_blurr.mean(axis=2).flatten())
    img_fig = plt.subplot(int("14" + str(i+2)))
    img_fig.imshow(img_gauss_blurr)

    plt.title(aoi+" rho:" + str(dict_omm_rho[aoi]))
    plt.xticks([])
    plt.yticks([])
    legend.append(aoi)
    i = i+1
plt.show()
bins = np.linspace(0, 255)
alpha = 1
for j in range(len(hist_dat)):
    fig2 = plt.hist(hist_dat[j], bins, alpha=alpha-j/len(hist_dat))
    plt.legend(legend)
    plt.pause(0.0001)
plt.show()

# img_sample_dirac = np.zeros((1001, 1001))
# img_sample_dirac[500,500] = 1
# img_sample_gauss = cv2.GaussianBlur(img_sample_dirac,(kernel_size,kernel_size),max(SIGMA))
# img_short_blurr = cv2.GaussianBlur(img_sample_dirac,(1001,1001),250)
# plt.subplot(121)
# plt.imshow(img_sample_dirac)
# plt.colorbar()
# plt.subplot(122)
# plt.imshow(img_sample_gauss)
# plt.colorbar()
# plt.show()
