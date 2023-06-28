"""
Script to measure the camera caliberation. Collects , fits and plots all the fisheye camera calibration measurements.
"""
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import os
parent_folder_path = "skylight_images/camera_caliberation/finer_increments/new_SETS" # path of the parent folder
folder_list = os.listdir(parent_folder_path) # getting list of the subfolders
unwanted_name = [ ".DS_Store"] #unwanted folders
folder_list = [i for i in folder_list if i not in unwanted_name] # removing the unwanted folders
img_num = 3 # number of images captured per trial
trial_num = 20  # number of trials
point_num = 19 # number of points made in each trials
entry_no = img_num * trial_num * point_num # total number of data points

calib_dict = calib_df = {"fd_name":[None] * entry_no,"trial":[None] * entry_no,"ori":[None] * entry_no,
                         "x":[None] * entry_no, "y":[None] * entry_no,
                         "x_diff":[None] * entry_no,"y_diff": [None] * entry_no,
                         "ele":[None] * entry_no,
                         } # creating an empty dict

calib_df = pd.DataFrame(calib_dict) # creating a dataframe from the dict
ele_vals = 3 * (list(range(90,-1,-10)) + list(range(80,-1,-10))) # elevation values of the points
idx_file = 0
lst_df = [] # list of the dataframes in each folder
lst_fd = [] # list of folder names
lst_fd = [] # list of folder names
lst_img_num = []  # list of folder names
for id,folder in enumerate(folder_list): # with in each folder
    for i in range(img_num): # with in each image
        df = pd.read_csv(parent_folder_path +"/" +folder + f"/Results_{i}.csv") # read the csv file named "results_"
        df["x_diff"] ,df["y_diff"] = abs(df.X[0] - df.X), abs(df.Y[0] - df.Y) # assign absolute pixel difference w.r.t. image centre
        lst_df.append(df) # append this dataframe to list
        lst_fd.extend(point_num * [folder]) #append folder names
        lst_img_num.extend(point_num * [i+1]) #append img num
    df_fold = pd.concat(lst_df) # concat to a master dataframe

df_fold["ele"] = trial_num * ele_vals # assigning elevation values
df_fold["fd_name"] = [*lst_fd] # assigning folder name
df_fold["ori"] = df_fold["fd_name"].str.split("_").str[-1] # taking the folder names last element (seperated by _) as orientation of the dome
df_fold["trial"] = df_fold["fd_name"].str.split("_").str[0] # taking the folder names first element (seperated by _) as trial of the dome
df_fold["alpha"] = 90 - df_fold["ele"] # alpha values 90 - elevation values
df_fold["img_num"] = lst_img_num # image num


df_fold["pixel_diff"] = np.where(df_fold.ori=="1",df_fold.x_diff,df_fold.y_diff) # pixel difference based on orientation
df_fold["axis"] = np.where(df_fold.ori=="1","x","y") # making axis column for orientation
sns.scatterplot(data = df_fold, x = "alpha",y = "pixel_diff", hue = "axis", alpha=0.5) # scatterplot the pixel difference


#
df_fold_0 = df_fold[df_fold.ele!=90] #selecting everything except the centre point aka 90 degree elevation


alpha_x  = df_fold_0[df_fold_0.axis == "x"].alpha #isolating alpha values based on axis
alpha_y  = df_fold_0[df_fold_0.axis == "y"].alpha
pixel_diff_x  = df_fold_0[df_fold_0.axis == "x"].pixel_diff
pixel_diff_y  = df_fold_0[df_fold_0.axis == "y"].pixel_diff

model = sm.OLS(pixel_diff_x,alpha_x) # fitting with a least square
result = model.fit()
result.params


def sin_half(x):
    return np.sin(x/2)
def tan_half(x):
    return np.tan(x/2)


"""
different models to fit the fisheye camera projections
"""

mod_rectilinear = smf.ols(formula= "pixel_diff ~ np.tan(np.deg2rad(alpha)) - 1 + axis", data = df_fold_0)
res_rectilinear = mod_rectilinear.fit()
res_rectilinear.summary()

mod_equidistant = smf.ols(formula= "pixel_diff ~ np.deg2rad(alpha) - 1 + axis", data = df_fold_0)
res_equidistant = mod_equidistant.fit()
res_equidistant.summary()

mod_equisolid = smf.ols(formula= "pixel_diff ~ sin_half(np.deg2rad(alpha)) - 1 + axis", data = df_fold_0)
res_equisolid = mod_equisolid.fit()
res_equisolid.summary()

mod_orthographic = smf.ols(formula= "pixel_diff ~ np.sin(np.deg2rad(alpha)) - 1 + axis", data = df_fold_0)
res_orthographic = mod_orthographic.fit()
res_orthographic.summary()

mod_stereographic = smf.ols(formula= "pixel_diff ~ tan_half(np.deg2rad(alpha)) - 1 + axis", data = df_fold_0)
res_stereographic = mod_stereographic.fit()
res_stereographic.summary()

x_lim_lab = np.linspace(0,90,10).astype(int)
x_lim_lab = [str(i)+"$^\circ$" for i in x_lim_lab]

x_lim = np.deg2rad(np.linspace(0,90,10))
x_func = np.deg2rad(np.linspace(0,90,100))

ax = plt.figure()
ax = sns.scatterplot(data= df_fold_0, x= np.deg2rad(df_fold_0.alpha), y = "pixel_diff", hue = "axis",s = 10,palette="bright")
plt.xticks(x_lim,x_lim_lab)
plt.xlabel(" 90$^\circ$ - elevation ")
plt.ylabel("| pixel difference |")
plt.title("Camera calibration raw data")

plt.tight_layout()
plt.savefig("camera_caliberation_raw_data.png", dpi=300, bbox_inches="tight")
plt.close()


ax2 = plt.figure(2)
ax2 = sns.scatterplot(data= df_fold_0, x= np.deg2rad(df_fold_0.alpha), y = "pixel_diff", hue = "axis",s = 10,palette="bright")
plt.xticks(x_lim,x_lim_lab)
plt.xlabel(" 90$^\circ$ - elevation ")
plt.ylabel("| pixel difference |")
plt.title("Camera calibration with different fits")
plt.xlim([0, np.deg2rad(91)])
plt.ylim([0, 850])
# plt.plot(x_func,res_rectilinear.params[1] * np.tan(x_func), label= "rectilinear")
plt.plot(x_func,res_equidistant.params[2] * x_func, label= "equidistant")
plt.plot(x_func,res_equisolid.params[2] * sin_half(x_func), label= "equisolid")
plt.plot(x_func,res_orthographic.params[2] * np.sin(x_func), label= "orthographic")
plt.plot(x_func,res_stereographic.params[2] * tan_half(x_func), label= "stereographic")
plt.legend()


plt.tight_layout()
plt.savefig("camera_caliberation_with_fits.png", dpi=300, bbox_inches="tight")
plt.close()

from pol_img_functions import elevation_mapping
import matplotlib.colors as colors
path='skylight_images/camera_caliberation/finer_increments/new_SETS/5_2/HDRcapture_--9995.088us_2.tiff'
img = plt.imread(path)
img = colors.Normalize()(img)
img = colors.PowerNorm(gamma=2.5)(img)
rad = int(np.round(df_fold_0[df_fold_0.ele == 0].pixel_diff.mean()))
_, map = elevation_mapping(img,radius=rad)
map = np.rad2deg(map)
eps = 0.0001
plt.imshow(img,cmap="gray")
plt.title("alignment w.r.t elevation map")
plt.xlabel("pixel coordinates")
plt.ylabel("pixel coordinates")

# plt.imshow(img,cmap="gray")
ang_range = list(range(80,0,-10))
alpha = np.linspace(0.4,0.1,len(ang_range))
for i,a in zip(ang_range, alpha):
    ang_circle = map - i <=eps
    plt.imshow(ang_circle,alpha=a)
    plt.pause(.001)

plt.tight_layout()

plt.savefig("Aligment_with_elevation_map.png",dpi=300, bbox_inches="tight")
