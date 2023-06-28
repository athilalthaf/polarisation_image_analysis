"""
Script to process all the aolp and dolp data from the skylight images . these files are across different trial and different bouts of experiments.
The script gathers all the files, process each files and then stores metrics into a dataframe which is later cleaned and used for statistical analysis
folder naming scheme is as follows :
    each time a data is collected make a folder in the skylight images folder
    use the following naming shceme monthday_time_cloud_cloudcover e.g. apr13_15_cloud_8 which means
    data is taken on april 13 starting at 3 pm and the cloud cover was 8
    important thing to not is make the cloud_cover is mentioned at the end of the filename
    within this folder we create folders namely 1 to 10 indicating the number of trials
    make sure exactly 10 trials are taken otherwise the script don't function

"""


import statsmodels.formula.api as smf
from lib_importer import * #importing the libraries


trial = np.arange(1,11)  # making a range of 1 to 10
radius = 211//2     #radius of the circular image

folder_name = os.listdir("skylight_images") # getting the list folders within skylight folders
unwanted_name = ["test", ".DS_Store","camera_caliberation"]   # list of folders needed to be excluded
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
df_sol["timestamp"] = pd.to_datetime(df_sol["Date"] + " " + df_sol["Time"]) #changing the format of datetime for consistency
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
        double check whether this is applicable  +}~@/
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
        if "cloud_cover.rtf" not in sub_folder_lst: #checking if a there is a another filed named cloud_cover in the dir
            cloud_cover = int(fdname[-1])  # getting the cloud cover data from the folder name
        else: #if yes read that as cloud cover instead foldername
            rescore_cloud_cover = pd.read_csv("skylight_images/{}/{}/cloud_cover.rtf".format(fdname, trial[i]))
            cloud_cover = int(rescore_cloud_cover.iloc[5][0][-2])
        # time_of_day = fdname[5:7]  # getting a roug
        name_list.append(fdname)  # appending foldernames to a  list
        trial_list.append(trial[i])  # appending trial number to a  list
        # time_list.append(time_of_day)
        cloud_cover_list.append(cloud_cover) # appending cloud to a  list
        mean_dolp_list.append(mean_dolp) # appending dolp to a  list
        mean_aolp_list.append(mean_aolp) # appending aolp to a  list
        proper_time_list.append(file_time_str) # appending timestamp to a  list



df = pd.DataFrame(list(zip(name_list, trial_list, cloud_cover_list, mean_dolp_list, np.rad2deg(mean_aolp_list),proper_time_list)),
                  columns=["name", "trial", "cloud_cover", "dolp", "aolp","timestamp"]) # combining the lists to a dataframe

# elevation_dict = {"10" : 30, "11" : 38, "13" : 48, "15" : 44} # degrees
df["timestamp"] = pd.to_datetime(df["timestamp"]) # converting the timestamp str to datetime object
df["timestamp"] = df["timestamp"].dt.round("min") # rounding the time to a minute wise resolution
merged_df = pd.merge(df,df_sol, on = "timestamp") # merging two dataframes sampa and polarisation df
merged_df["elevation"] = np.round(90 - merged_df["elevation"]) # elevation data is in altitude format
merged_df["day"] = merged_df.timestamp.dt.day_of_year
print(merged_df.head()) #priting the final dataframe
clean_df = merged_df[merged_df.name == "apr18_17_cloud_7"]
merged_df.to_excel("raw_data.xlsx")


palette2 = sns.dark_palette("#69d", reverse=True, n_colors =9).as_hex() #color palette from blue to grey for cloud cover
# sns.lmplot(data= b, x = "elevation", y ="dolp",hue = "cloud_cover" ,palette = palette2)
palette_ele = sns.color_palette(n_colors=len(clean_df.elevation.unique())).as_hex()  # color palette for elevation
# clean_df = clean_df.astype({"day":"string"})
model = smf.mixedlm("aolp ~ azimuth*elevation*cloud_cover + (1| day)", data=clean_df,groups = clean_df.day) # model for glmm
result = model.fit() # fittting
null_model = smf.mixedlm("aolp ~  + (1 | day) ", data=clean_df,groups = clean_df.day) # null model to compare
null_result = null_model.fit() # fitting
print(result.summary()) # getting the summary

LR_stat = -2 * (null_result.llf - result.llf) # checking the significance w.r.t null model
p_val = stats.chi2.sf(LR_stat,result.df_modelwc)  # p value of model selecation
print(f"no. of parameters : {result.df_modelwc}")
print(f"p_value w.r.t. null model : {p_val}")





ele_range = np.arange(clean_df.elevation.min(), clean_df.elevation.max()+ 1) # getting the range of elevation
counts = clean_df.elevation.value_counts() # counting each elevation

counts = counts.reindex(ele_range,fill_value=0) # reindexing


date = datetime.datetime.now() # getting the current date time  for saving images with date  and preventing file overwriting
date = date.strftime("%Y-%m-%d %H:%M:%S") # converting to str format

"""
PLOTS !!
"""
"""
DoLP v/s Cloud Cover
"""
plt.figure(1)
sns.stripplot(data = clean_df,x="cloud_cover",y="dolp",hue="cloud_cover",palette=palette2,legend=False,size = 2)
plt.xlabel("Cloud Cover")
plt.ylabel("DoLP")
plt.title("DoLP v/s Cloud cover")
plt.tight_layout()
plt.savefig("cloud_cover_dolp "+date+ " .png", dpi=300, bbox_inches="tight")

"""
AoLP v/s Cloud Cover
"""
plt.figure(2)
sns.stripplot(data = clean_df,x="cloud_cover",y="aolp",hue="cloud_cover",palette=palette2,legend=False,size = 2) #creating a stripplot to show the AoLP variation on different cloud covers
plt.xlabel("Cloud Cover")
plt.ylabel("AoLP in $^\circ$")
plt.title("AoLP v/s Cloud cover")
plt.tight_layout()
plt.savefig("cloud_cover_aolp "+date+ " .png", dpi=300, bbox_inches="tight")


"""
DoLP v/s elevation
"""
plt.figure(3)
sns.lmplot(data= clean_df, x = "elevation", y ="dolp",hue = "cloud_cover" ,palette = palette2) #creating a linear fit to show the DoLP variation on elevation for different cloud covers
plt.xlabel("elevation in $^\circ$")
plt.ylabel("DoLP")
plt.title("DoLP v/s elevation")
plt.savefig("ele_dolp "+date+ " .png", dpi=300, bbox_inches="tight")

"""
AoLP v/s azimuth
"""
plt.figure(4)
sns.lmplot(data= clean_df, x = "azimuth", y ="aolp",hue = "cloud_cover" ,palette = palette2) #creating a linear fit to show the AoLP variation on azimuth for different cloud covers
plt.xlabel("azimuth in $^\circ$")
plt.ylabel("AoLP in $^\circ$")
plt.title("AoLP v/s Azimuth")
plt.savefig("azi_aolp "+date+ " .png", dpi=300, bbox_inches="tight")

"""
elevation data
"""
plt.figure(5)
counts.plot.barh(color = "green") # histogram of counts of different elevation values
plt.title("overall elevation data")
plt.ylabel("elevation in $^\circ$")
plt.xlabel("count")
plt.tick_params(axis="y", labelsize=5)
plt.tight_layout()
plt.savefig("elevation_dat "+date+ " .png", dpi=300, bbox_inches="tight")






