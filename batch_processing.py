import pandas as pd

from lib_importer import * #importing the libraries
from pvlib import solarposition
import pytz
#parent directory

# a = os.path.dirname(__file__)
# b = os.chdir(os.path.abspath(a + "/.."))

# os.chdir("skylight_images/apr3_13_cloud_8/1")
"""
folder naming scheme is as follows :
    each time a data is collected make a folder in the skylight images folder
    use the following naming shceme monthday_time_cloud_cloudcover e.g. apr13_15_cloud_8 which means 
    data is taken on april 13 starting at 3 pm and the cloud cover was 8
    important thing to not is make the cloud_cover is mentioned at the end of the filename 
    within this folder we create folders namely 1 to 10 indicating the number of trials
    make sure exactly 10 trials are taken otherwise the script don't function
     
"""
trial = np.arange(1,11)  # making a range of 1 to 10
radius = 211//2     #radius of the circular image

folder_name = os.listdir("skylight_images") # getting the list folders within skylight folders
unwanted_name = ["test", ".DS_Store"]   # list of folders needed to be excluded
folder_name = [i for i in folder_name if i not in unwanted_name] # list of all the folders that contains the images and plots
"""
the dataframe columns 
"""
name_list = [] # list of folder names
trial_list = [] # list of trial number in each bout of measurement , taken from the subfolder names
# time_list = []
cloud_cover_list = []  # list of cloud cover taken from the folder name
mean_dolp_list = [] # list of dolp
mean_aolp_list = [] # list of aolp
proper_time_list = [] # list of image aquisition time

"""
reading and modifying the SAMPA data as a dataframe
"""
df_sol  = pd.read_csv("sampa_apr_aug.csv") # reading SAMPA data
df_sol.drop("Solar Top. azimuth angle (westward from S)",axis = 1, inplace = True) #removing azimuth colum (0 to 180 )
df_sol.rename(columns = {"Solar Topocentric zenith angle" : "elevation", "Solar Top. azimuth angle (eastward from N)":"azimuth"},  inplace = True) #renaming the columns for ease of use
df_sol["Timestamp"] = pd.to_datetime(df_sol["Date"] + " " + df_sol["Time"]) #changing the format of datetime for consistency
df_sol = df_sol.drop(["Date", "Time"],axis=1) #removing the time and day as we already have the same data in timestamp column
for fdname in folder_name: #going through each measurement
    for i in range(10): # going through each trial with each measurement
        file_name_dolp = "skylight_images/{}/{}/DoLP_pixels.csv".format(fdname, trial[i]) #getting the path to DoLP.csv
        file_name_aolp = "skylight_images/{}/{}/AoLP_pixels.csv".format(fdname, trial[i]) #getting the path to AoLP.csv
        dolp_data = pd.read_csv(file_name_dolp, dtype="float")  #reading DoLP.csv
        aolp_data = pd.read_csv(file_name_aolp, dtype="float")  #reading AoLP.csv
        dolp_data = dolp_data.to_numpy() # converting to numpy object to read image
        aolp_data = aolp_data.to_numpy() # converting to numpy object to read image
        sub_folder_lst = os.listdir("skylight_images/{}/{}".format(fdname, trial[i])) #list of items with the trial folders
        middle_img_name = [i for i in sub_folder_lst if "_1.tiff" in i] #getting the tiff image with middle exposure "_1.tiff"
        # extracting time from file to get proper time of elevation and azimuth
        """
        important note: the code is written in mac which saves the file created time in the modified time list ,
        double check whether this is applicable  
        """
        file_time_stamp = os.path.getmtime("skylight_images/{}/{}/{}".format(fdname, trial[i],*middle_img_name)) # modified time of middle exposed file
        file_time_dateobj = datetime.datetime.fromtimestamp(file_time_stamp)  # converting it to datetime format
        file_time_str = file_time_dateobj.strftime("%Y-%m-%d %H:%M:%S") #modified time in str format
        # file_time_dtime = file_time_dateobj + datetime.timedelta(minutes=1)   # for one minute difference
        # file_time_dtime = file_time_dtime.strftime("%Y-%m-%d %H:%M:%S")

        """
        processing and handling of AOLP and DOLP goes here
        """
        h, w = dolp_data.shape[0], dolp_data.shape[1] # height and width of the dolp data, dimensions are same for aolp as well

        centre_x = dolp_data.shape[1] // 2 #getting the centre data
        centre_y = dolp_data.shape[0] // 2
        Y, X = np.ogrid[:h, :w] # getting a meshgrid to mask the image
        dist_from_centre = np.sqrt((X - centre_x) ** 2 + (Y - centre_y) ** 2) #generating the distance grid from the centre of the image

        mask = dist_from_centre >= radius  # masking everything out the radius

# plt.imshow(a)
# plt.show()




        dolp_data[mask] = np.nan # assigning nan to masked out values
        aolp_data[mask] = np.nan
        mean_dolp = np.nanmean(dolp_data) # getting the mean for dolp data
        # print( fdname, trial[i], time_of_day, cloud_cover,mean_dolp, mean_aolp)
        sin_mean = np.nanmean(np.sin(aolp_data))
        cos_mean = np.nanmean(np.cos(aolp_data))
        mean_aolp = np.arctan2(sin_mean,cos_mean) # getting the circular mean for aolp


        """
        gathering all the datas to the list
        """
        cloud_cover = int(fdname[-1])  # getting the cloud cover data from the folder name
        # time_of_day = fdname[5:7]  # getting a roug
        name_list.append(fdname)  # appending foldernames to a  list
        trial_list.append(trial[i])  # appending trial number to a  list
        # time_list.append(time_of_day)
        cloud_cover_list.append(cloud_cover) # appending cloud to a  list
        mean_dolp_list.append(mean_dolp) # appending dolp to a  list
        mean_aolp_list.append(mean_aolp) # appending aolp to a  list
        proper_time_list.append(file_time_str) # appending timestamp to a  list

    # plt.imshow(a)
    # plt.show()
    # print(a)

df = pd.DataFrame(list(zip(name_list, trial_list, cloud_cover_list, mean_dolp_list, np.rad2deg(mean_aolp_list),proper_time_list)),
                  columns=["Day", "Trial", "Cloud cover", "DoLP", "AoLP","Timestamp"]) # combining the lists to a dataframe

# elevation_dict = {"10" : 30, "11" : 38, "13" : 48, "15" : 44} # degrees
df["Timestamp"] = pd.to_datetime(df["Timestamp"]) # converting the timestamp str to datetime object
df["Timestamp"] = df["Timestamp"].dt.round("min") # rounding the time to a minute wise resolution
merged_df = pd.merge(df,df_sol, on = "Timestamp") # merging two dataframes sampa and polarisation df
merged_df["elevation"] = np.round(90 - merged_df["elevation"]) # elevation data is in altitude format

print(merged_df.head()) #priting the final dataframe

# ax1 = df.boxplot(column="DoLP",by="Cloud cover")
# plt.show()
#
# ax2 = df.boxplot(column="AoLP",by="Cloud cover")
# plt.show()
#
# grouped_ele_dolp = df.groupby(["Day","Elevation"]).agg({"DoLP" : ["mean"]})
# grouped_ele_dolp.columns = ["DoLP mean"]
# grouped_ele_dolp.reset_index()
# ax2 = grouped_ele_dolp.boxplot(column="DoLP mean", by="Elevation")
# # ax2 = sns.boxplot(data=grouped_ele_dolp,groupby="Elevation")
# plt.xlabel(" Elevation in $^\circ$")
# plt.ylabel(" Mean DoLP ")
# plt.tight_layout()
#
# grouped_ele_aolp = df.groupby(["Day","Elevation"]).agg({"AoLP" : ["mean"]})
# grouped_ele_aolp.columns = ["AoLP mean"]
# grouped_ele_aolp.reset_index()
# ax3 = grouped_ele_aolp.boxplot(column="AoLP mean", by="Elevation")
# # ax2 = sns.boxplot(data=grouped_ele_dolp,groupby="Elevation")
# plt.xlabel(" Elevation in $^\circ$")
# plt.ylabel(" Mean AoLP ")
# plt.tight_layout()
# plt.savefig("group_aolp_ele.png",dpi = 300, bbox_inches="tight")
# plt.show()
# plt.savefig("group_dolp_ele.png",dpi = 300, bbox_inches="tight")
#
#
# grouped_clo_dolp = df.groupby(["Day","Cloud cover"]).agg({"DoLP" : ["mean"]})
# grouped_clo_dolp.columns = ["DoLP mean"]
# grouped_clo_dolp.reset_index()
# ax4 = grouped_clo_dolp.boxplot(column="DoLP mean", by="Cloud cover")
# # ax2 = sns.boxplot(data=grouped_ele_dolp,groupby="Elevation")
# plt.xlabel(" Cloud cover ")
# plt.ylabel(" Mean DoLP ")
# plt.tight_layout()
# plt.savefig("group_dolp_clo.png",dpi = 300, bbox_inches="tight")
#
#
# grouped_clo_aolp = df.groupby(["Day","Cloud cover"]).agg({"AoLP" : ["mean"]})
# grouped_clo_aolp.columns = ["AoLP mean"]
# grouped_clo_aolp.reset_index()
# ax5 = grouped_clo_aolp.boxplot(column="AoLP mean", by="Cloud cover")
# # ax2 = sns.boxplot(data=grouped_ele_dolp,groupby="Elevation")
# plt.xlabel(" Cloud cover ")
# plt.ylabel(" Mean AoLP  $^\circ$")
# plt.tight_layout()
# plt.savefig("group_aolp_clo.png",dpi = 300, bbox_inches="tight")
#
# gp = grouped_clo_aolp.groupby("Cloud cover").groups
#
#
# ## stat result anova
#
# clo_aolp = grouped_clo_aolp.groupby("Cloud cover")["AoLP mean"].apply(list).reset_index()
# clo_dolp = grouped_clo_dolp.groupby("Cloud cover")["DoLP mean"].apply(list).reset_index()
#
# stat_clo_aolp = stats.f_oneway(*clo_aolp["AoLP mean"])
# stat_clo_dolp = stats.f_oneway(*clo_dolp["DoLP mean"])
#
# ele_aolp = grouped_ele_aolp.groupby("Elevation")["AoLP mean"].apply(list).reset_index()
# ele_dolp = grouped_ele_dolp.groupby("Elevation")["DoLP mean"].apply(list).reset_index()
#
# stat_ele_aolp = stats.f_oneway(*ele_aolp["AoLP mean"])
# stat_ele_dolp = stats.f_oneway(*ele_dolp["DoLP mean"])
#
#
# stats_list = [clo_aolp, clo_dolp, ele_aolp, ele_dolp]



# Define the observer's location
# konstanz = ephem.Observer()
# konstanz.lat = '47.67795'
# konstanz.lon = '9.173324'
# konstanz.elevation = 0 # meters above sea level
# konstanz.horizon = '-0:34' # adjust for atmospheric refraction

# lon = 9.1737
# lat = 47.6780
# elevation = 0 # meters above sea level
# # Define the date and time
# ele = []
# timezone = "Europe/Berlin"
#
# for i in range(len(df)):
#     # dt = ephem.Date(df["Timestamp"][i])  # 3:10 PM
#     dt = df["Timestamp"][i]  # 3:10 PM
#     t = pytz.timezone(timezone)
#     sol_pos = solarposition.get_solarposition(dt,lat,lon)
#     sun_ele = sol_pos["elevation"]
#
#
# #     konstanz.date = dt
# # # Compute the sun's position
# # #     sun = ephem.Sun(konstanz)
# #     sun = ephem.Sun(konstanz)
# #     sun.compute(konstanz)
#
#     # Get the azimuth and altitude in degrees
#     # azimuth = float(sun.az) * 180.0 / ephem.pi
#     # altitude = float(sun.alt) * 180.0 / ephem.pi
#     # altitude = float(np.rad2deg(sun.alt))
#
#     # print("DT: {} elevation:{} corrected : {}".format(dt, np.round(sun_ele.values), 90-np.round(sun_ele)))
#     print("DT: {} elevation:{}".format(dt, np.round(sun_ele.values[0])))
#
#     ele.append(np.round(sun_ele))
#     # print("Sun altitude:", altitude)
sort_cloud = merged_df["Cloud cover"].sort_values().unique()
sns.regplot(data= merged_df, x = "elevation", y ="DoLP",color = "black",scatter=False )
sns.scatterplot(data = merged_df,x = "elevation",y = "DoLP", hue = "Cloud cover",hue_order = sort_cloud,palette = "cool")
plt.xlim([0,50])
plt.ylim([0,.5])
plt.xlabel("elevation in $^\circ$")
plt.ylabel("DoLP")
plt.title("DoLP v/s elevation")
plt.tight_layout()

plt.savefig("DoLP_ele_scatter_reg.png",dpi = 300, bbox_inches="tight")

merged_df["Cloud cover"] = merged_df["Cloud cover"].astype(int)
merged_df.sort_values("Cloud cover")
sort_elevation = merged_df["elevation"].sort_values().unique()
sns.regplot(data= merged_df, x = "Cloud cover", y ="DoLP",color = "black",scatter=False )
sns.scatterplot(data = merged_df,x = "Cloud cover",y = "DoLP")
# plt.xlim([0,50])
plt.ylim([0,.5])
plt.xlabel("Cloud cover")
plt.ylabel("DoLP")
plt.title("DoLP v/s cloud cover")
plt.tight_layout()

plt.savefig("DoLP_cloud_scatter_reg.png",dpi = 300, bbox_inches="tight")

slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df['elevation'], merged_df['DoLP'])
# extract the regression equation and R-squared value
print("Regression Equation: y = {:.2f}x + {:.2f}".format(slope, intercept))
print("R-squared value: {:.2f}".format(r_value**2))
print("p value: {:.4f}".format(p_value))

slope, intercept, r_value, p_value, std_err = stats.linregress(merged_df['Cloud cover'], merged_df['DoLP'])
# extract the regression equation and R-squared value
print("Regression Equation: y = {:.2f}x + {:.2f}".format(slope, intercept))
print("R-squared value: {:.2f}".format(r_value**2))
print("p value: {:.8f}".format(p_value))


# sns.regplot(data= merged_df, x = "azimuth", y ="AoLP",color = "black",scatter=False )
sns.scatterplot(data = merged_df,x = "azimuth",y = "AoLP", hue = "Cloud cover",hue_order = sort_cloud,palette = "cool")
# plt.xlim([0,50])
# plt.ylim([0,.5])
plt.xlabel("Azimuth in $^\circ$")
plt.ylabel("AoLP in $^\circ$")
plt.title("AoLP v/s Azimuth")
plt.tight_layout()

plt.savefig("AoLP_cloud_scatter.png",dpi = 300, bbox_inches="tight")
