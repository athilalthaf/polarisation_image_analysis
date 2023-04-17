import numpy as np
import scipy.stats  as stats
import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
#import pvlib
import seaborn as sns
import ephem

#parent directory

# a = os.path.dirname(__file__)
# b = os.chdir(os.path.abspath(a + "/.."))

# os.chdir("skylight_images/apr3_13_cloud_8/1")
trial = np.arange(1,11)
radius = 211//2

folder_name = os.listdir("skylight_images")
unwanted_name = ["test", ".DS_Store"]
folder_name = [i for i in folder_name if i not in unwanted_name]

name_list = []
trial_list = []
time_list = []
cloud_cover_list = []
mean_dolp_list = []
mean_aolp_list = []
proper_time_list = []
for fdname in folder_name:
    for i in range(10):
        file_name_dolp = "skylight_images/{}/{}/DoLP_pixels.csv".format(fdname, trial[i])
        file_name_aolp = "skylight_images/{}/{}/AoLP_pixels.csv".format(fdname, trial[i])
        dolp_data = pd.read_csv(file_name_dolp, dtype="float")
        aolp_data = pd.read_csv(file_name_aolp, dtype="float")
        dolp_data = dolp_data.to_numpy()
        aolp_data = aolp_data.to_numpy()

        # extracting time from file to get proper time of elevation and azimuth
        file_time_stamp = os.path.getmtime(file_name_dolp)
        file_time_dateobj = datetime.datetime.fromtimestamp(file_time_stamp)
        file_time_str = file_time_dateobj.strftime("%Y/%m/%d %H:%M:%S") #modified time in str format
        file_time_dtime = file_time_dateobj + datetime.timedelta(minutes=1)   # for one minute difference
        file_time_dtime = file_time_dtime.strftime("%Y-%m-%d %H:%M:%S")



        h, w = dolp_data.shape[0], dolp_data.shape[1]
        cloud_cover = fdname[-1]
        time_of_day = fdname[5:7]      ###
        centre_x = dolp_data.shape[1] // 2
        centre_y = dolp_data.shape[0] // 2
        Y, X = np.ogrid[:h, :w]
        dist_from_centre = np.sqrt((X - centre_x) ** 2 + (Y - centre_y) ** 2)

        mask = dist_from_centre >= radius

# plt.imshow(a)
# plt.show()




        dolp_data[mask] = np.nan
        aolp_data[mask] = np.nan
        mean_dolp = np.nanmean(dolp_data)
        mean_aolp = np.nanmean(aolp_data)
        # print( fdname, trial[i], time_of_day, cloud_cover,mean_dolp, mean_aolp)

        name_list.append(fdname)
        trial_list.append(trial[i])
        time_list.append(time_of_day)
        cloud_cover_list.append(cloud_cover)
        mean_dolp_list.append(mean_dolp)
        mean_aolp_list.append(mean_aolp)
        proper_time_list.append(file_time_str)
    # plt.imshow(a)
    # plt.show()
    # print(a)

df = pd.DataFrame(list(zip(name_list, trial_list, time_list, cloud_cover_list, mean_dolp_list, np.rad2deg(mean_aolp_list),proper_time_list)), columns=["Day", "Trial", "time", "Cloud cover", "DoLP", "AoLP","Timestamp"])

elevation_dict = {"10" : 30, "11" : 38, "13" : 48, "15" : 44} # degrees

ele_list = [elevation_dict[str(i)] for i in time_list]

df.insert(3, "Elevation", ele_list)

print(df.head())

ax1 = df.boxplot(column="DoLP",by="Cloud cover")
plt.show()

ax2 = df.boxplot(column="AoLP",by="Cloud cover")
plt.show()

grouped_ele_dolp = df.groupby(["Day","Elevation"]).agg({"DoLP" : ["mean"]})
grouped_ele_dolp.columns = ["DoLP mean"]
grouped_ele_dolp.reset_index()
ax2 = grouped_ele_dolp.boxplot(column="DoLP mean", by="Elevation")
# ax2 = sns.boxplot(data=grouped_ele_dolp,groupby="Elevation")
plt.xlabel(" Elevation in $^\circ$")
plt.ylabel(" Mean DoLP ")
plt.tight_layout()

grouped_ele_aolp = df.groupby(["Day","Elevation"]).agg({"AoLP" : ["mean"]})
grouped_ele_aolp.columns = ["AoLP mean"]
grouped_ele_aolp.reset_index()
ax3 = grouped_ele_aolp.boxplot(column="AoLP mean", by="Elevation")
# ax2 = sns.boxplot(data=grouped_ele_dolp,groupby="Elevation")
plt.xlabel(" Elevation in $^\circ$")
plt.ylabel(" Mean AoLP ")
plt.tight_layout()
plt.savefig("group_aolp_ele.png",dpi = 300, bbox_inches="tight")
plt.show()
plt.savefig("group_dolp_ele.png",dpi = 300, bbox_inches="tight")


grouped_clo_dolp = df.groupby(["Day","Cloud cover"]).agg({"DoLP" : ["mean"]})
grouped_clo_dolp.columns = ["DoLP mean"]
grouped_clo_dolp.reset_index()
ax4 = grouped_clo_dolp.boxplot(column="DoLP mean", by="Cloud cover")
# ax2 = sns.boxplot(data=grouped_ele_dolp,groupby="Elevation")
plt.xlabel(" Cloud cover ")
plt.ylabel(" Mean DoLP ")
plt.tight_layout()
plt.savefig("group_dolp_clo.png",dpi = 300, bbox_inches="tight")


grouped_clo_aolp = df.groupby(["Day","Cloud cover"]).agg({"AoLP" : ["mean"]})
grouped_clo_aolp.columns = ["AoLP mean"]
grouped_clo_aolp.reset_index()
ax5 = grouped_clo_aolp.boxplot(column="AoLP mean", by="Cloud cover")
# ax2 = sns.boxplot(data=grouped_ele_dolp,groupby="Elevation")
plt.xlabel(" Cloud cover ")
plt.ylabel(" Mean AoLP  $^\circ$")
plt.tight_layout()
plt.savefig("group_aolp_clo.png",dpi = 300, bbox_inches="tight")

gp = grouped_clo_aolp.groupby("Cloud cover").groups


## stat result anova

clo_aolp = grouped_clo_aolp.groupby("Cloud cover")["AoLP mean"].apply(list).reset_index()
clo_dolp = grouped_clo_dolp.groupby("Cloud cover")["DoLP mean"].apply(list).reset_index()

stat_clo_aolp = stats.f_oneway(*clo_aolp["AoLP mean"])
stat_clo_dolp = stats.f_oneway(*clo_dolp["DoLP mean"])

ele_aolp = grouped_ele_aolp.groupby("Elevation")["AoLP mean"].apply(list).reset_index()
ele_dolp = grouped_ele_dolp.groupby("Elevation")["DoLP mean"].apply(list).reset_index()

stat_ele_aolp = stats.f_oneway(*ele_aolp["AoLP mean"])
stat_ele_dolp = stats.f_oneway(*ele_dolp["DoLP mean"])


stats_list = [clo_aolp, clo_dolp, ele_aolp, ele_dolp]



# Define the observer's location
konstanz = ephem.Observer()
konstanz.lat = '47.6780'
konstanz.lon = '9.1737'
konstanz.elevation = 405 # meters above sea level
konstanz.horizon = '-0:34' # adjust for atmospheric refraction

# Define the date and time
ele = []
for i in range(len(df)):
    dt = ephem.Date(df["Timestamp"][i])  # 3:10 PM
    konstanz.date = dt
# Compute the sun's position
    sun = ephem.Sun(dt)
    sun.compute(konstanz)

    # Get the azimuth and altitude in degrees
    azimuth = float(sun.az) * 180.0 / ephem.pi
    altitude = float(sun.alt) * 180.0 / ephem.pi

    print("DT: {} elevation:{} corrected : {}".format(dt, np.round(altitude), np.round(altitude)+15))
    ele.append(np.round(altitude))
    # print("Sun altitude:", altitude)
