# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 12:12:45 2021

@author: vkrugten
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 14:36:08 2021

@author: jaap
"""
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot
import statistics


#%% Input
Trajectory_path = r"D:\210914_ExNo_68_FM4-64_60x\210914_ExNo_68_FM4-64_60x_Pos2\Transposed tracks\Results\*Molecule *.csv"
DataSet_name = "210914_ExNo_68_FM4-64_60x_Pos2_"
SavePath = r"D:\210914_ExNo_68_FM4-64_60x\210914_ExNo_68_FM4-64_60x_Pos2\Transposed tracks\Analyzed//" 
Save_figures = 'n' #'Y' will save it

dt = 50
enough = 0
too_few= 0

stylesheet= r"C:\Users\vkrugten\surfdrive\Programming\Jaapstyle_light.mplstyle"
plt.style.use(stylesheet)

#%% Listst to be filled
All_Speed = [] #euclidean p2p dist/time
All_Vel = [] # Velocity from euclidean p2p dist/time
All_Vel_Along_Spline = [] #from 1D displacement along drawn spline, MATLAB column 9
All_Vel_Transposed= [] #euclidean speed from first point till point i, MATLAB column 8
Track_length_in_Points = []
Stepsizes = []
Ort_positions = [] # position along the drawn spline, MATLAB column 7
Ort_displacements_p2pS = [] #point to point displacement
Total_Ort_displS = [] # Total ort displ
Displ_along_spline_p2pS = [] #1D displacement along drawn spline
Transposed_euclidean_dist_p2pS = [] # 

DisplS_along_spline_Cum = []

Total_Ort_positionS = []
Total_displacementS_along = []

Total_Ort_positionS_RolAVG10 = []
Tortuosities = []

#%%Analysis loop
for filename in glob.glob(Trajectory_path):
    df = pd.read_csv(filename, sep="\t")#, skiprows=1)
    
    x_position = df['x_position[nm]'].to_numpy() # put untransposed coordinates in array
    y_position = df['y_position[nm]'].to_numpy()
    diff_x = x_position[:-1] - x_position[1:] # calculate difference with previous row
    diff_y = y_position[:-1] - y_position[1:]
    df['x_displacement[nm]'] = np.insert(diff_x,0,0)        
    df['y_displacement[nm]'] = np.insert(diff_y,0,0)
    
    #Calculate euclidean distance and Speed (from untransposed coordinates)
    dist =  np.sqrt(diff_x **2 +diff_y**2)
    distMinus1st = np.insert(dist,0,np.nan)  
    df['Euclidean_p2p_distance[nm]'] = distMinus1st
    Stepsizes.append(dist)     
    Speed = dist/dt
    All_Speed.append(Speed) #appends the result of Speed calculations numpy array to the end of the All_Speed numpy array for every loop
    
    # Calculate signed Velocity from Speed (from untransposed coordinates)
    df['Speed[nm/s]'] = np.insert(Speed,0,np.nan)
    df['Velocity[nm/s]'] = df['Speed[nm/s]']*(2*(df['y_displacement[nm]'] >0)-1)
    All_Vel.append(df['Velocity[nm/s]'])
 
    
    #Convert cumulative distance along spline to p2p dustance along the spline
    Displ_along_spline_Cum = df['Cumulative_distance_along_spline_1D'].to_numpy()
    Displ_along_spline_p2p = Displ_along_spline_Cum[:-1] - Displ_along_spline_Cum[1:]
    Displ_along_spline_p2pS.append(Displ_along_spline_p2p)
    
    df['Displ_along_spline_p2p[nm]'] = np.insert(Displ_along_spline_p2p,0,np.nan)
    
    DisplS_along_spline_Cum.append(Displ_along_spline_Cum)

    #THIS DOES NOT CALCULATE WHAT i WANT IT TO, CHECK!!!
    Transposed_euclidean_dist_Cum = df['Cumulative_Euclidean_distance_2D'].to_numpy()
    Transposed_euclidean_dist_p2p = Transposed_euclidean_dist_Cum[:-1] - Transposed_euclidean_dist_Cum[1:]
    Transposed_euclidean_dist_p2pS.append(Transposed_euclidean_dist_p2p)
    
    df['Transposed_euclidean_dist_p2p[nm]'] = np.insert(Transposed_euclidean_dist_p2p,0,np.nan)
    
    
    #Orthogonal position from drawn spline  
    Ort_position = df['Orthogonal_distance_to_spline']
    Ort_positions.append(Ort_position)
    
    Ort_displacements_p2p =  Ort_position.shift(-1) - Ort_position
    Ort_displacements_p2pS.append(Ort_displacements_p2p)
    
    Total_Ort_displ = max(Ort_position) - min(Ort_position)
    Total_Ort_displS.append(Total_Ort_displ)
    
    #Get all the Velocities values calculated in the MATLAB column 5 script and add them to one list to make a histogram from     
    Vel_Transposed = df['P2P_Velocities_Cumulative_Euclidean_distance'].to_numpy()
    All_Vel_Transposed.append(Vel_Transposed/1000)  
    
    Vel_Along_Spline = df['P2P_Velocities_Distance_along_spline'].to_numpy()
    All_Vel_Along_Spline.append(Vel_Along_Spline/1000)
  
       
    #Calculate number of timepoints per trajectory and add to list of all trajectories
    Len = len(x_position)
    Track_length_in_Points.append(Len)
   
    
    ##Calculate Tortuosity
    # total euclidian displacement for one trajectory
    DisplTot_X = x_position[len(x_position)-1] - x_position[0]
    DisplTot_Y = y_position[len(y_position)-1] - y_position[0]
    DisplTot =  np.sqrt(DisplTot_X **2 + DisplTot_Y**2)
    Tortuosity = dist.sum()/DisplTot
    Tortuosities.append(Tortuosity)
    
    #Calculate max x and y displacement
    Total_Ort_position = max(Ort_position)-min(Ort_position)
    Total_Ort_positionS.append(Total_Ort_position)
    
    Total_displacement_along_spline= max(Displ_along_spline_p2p)-min(Displ_along_spline_p2p)
    Total_displacementS_along.append(Total_displacement_along_spline)
    
    #Caculate ort displacements from transposed stuff for a certain range
    # Ort_position_30_points = max(Ort_position[0:9])-min(Ort_position[0:9])
    # Ort_positions_30_points.append(Ort_position_30_points)
     
         
    # add arrays to existing dataframe 
    df["Speed_rolling_average[nm/s]"] = df['Speed[nm/s]'].rolling(window=10).mean()
    
    #check what files have Veklocities over 100 um/s in then
    if df['Speed[nm/s]'].gt(100).any():
        print("These files have a Speed value of over 100 um/s")
        print(filename)
         
    # safe new dataframe with the columns 'header' in specified location
    header = ["frame", "time[s]","x_position[nm]","y_position[nm]","Cumulative_Euclidean_distance_2D",
                                                "Cumulative_distance_along_spline_1D","Orthogonal_distance_to_spline",
                                                "P2P_Velocities_Cumulative_Euclidean_distance","P2P_Velocities_Distance_along_spline",
                                                'Euclidean_p2p_distance[nm]', 'Speed[nm/s]', 'Velocity[nm/s]', 'Displ_along_spline_p2p[nm]',
                                                'Transposed_euclidean_dist_p2p[nm]', ]
    names = DataSet_name + "Velo Molecule " + filename.split("Molecule ")[1][:-4] + ".txt"
    df.to_csv(SavePath + names, sep='\t', encoding='utf-8', index=False, columns = header)
    
    #plot figure
    # fig1 = plt.figure(1, figsize=(10,10))
    # plot(df.index, SpeedMinus1st, marker='o', linewidth=0.5, markersize=3, markerfacecolor='none', markeredgewidth=1) #label="Mol "+ filename.split("Molecule ")[1][:-4])
    # # plt.legend(framealpha=0, loc='best', bbox_to_anchor=(0.8, 0.53, 0.5, 0.5)) #labelcolor=COLOR,
    # plt.xlabel("frame #")
    # plt.ylabel("velocity (\u03BCm/s)")  
    
    # fig2 = plt.figure(2, figsize=(10,10))
    # plot(df['time[s]'], df["RolAvg_Orth(x)Displ [nm]"],linewidth=0.5, markersize=3, markerfacecolor='none', markeredgewidth=1) #label="Mol "+ filename.split("Molecule ")[1][:-4])
    # plt.xlabel("time [s]")
    # plt.ylabel("Rolling average Orthogonal displacement [nm/s]") 
       
    
     
    #If I want to check how many coordinates there are in a file with this filename
    # print("# of x_values in Mol", filename.split("Molecule ",1)[1][:-4], "=", len(x_values))
     
      
    
    
    
    # print("Number of Trajectories with > 10 datapoints = ", enough)        
    # print("Number of Trajectories with < 10 datapoints = ", too_few)   


#%% Calculate statistics and write to new dataframe,transpose it, and save it

#Concatenate arrays so it are lists with all the values behind each other

All_Speed =  np.concatenate(All_Speed) #euclidean p2p dist/time
All_Vel =  np.concatenate(All_Vel) # Velocity from euclidean p2p dist/time
All_Vel_Along_Spline =  np.concatenate(All_Vel_Along_Spline) #from 1D displacement along drawn spline, MATLAB column 9
All_Vel_Transposed=  np.concatenate(All_Vel_Transposed) #euclidean speed from first point till point i, MATLAB column 8
Stepsizes =  np.concatenate(Stepsizes)
DisplS_along_spline_Cum = np.concatenate(DisplS_along_spline_Cum)

Ort_positions =  np.concatenate(Ort_positions) # position along the drawn spline, MATLAB column 7
Ort_displacements_p2pS = np.concatenate(Ort_displacements_p2pS)
Displ_along_spline_p2pS =  np.concatenate(Displ_along_spline_p2pS) #1D displacement along drawn spline
Transposed_euclidean_dist_p2pS =  np.concatenate(Transposed_euclidean_dist_p2pS) # 


# To_root_Y_positive = len(list(filter(lambda x: (x >= 0), Par_displacements)))
# To_tip_Y_negative = len(list(filter(lambda x: (x < 0), Par_displacements)))

# #%% Write to new dataframe and save as a text file with all the statistics
# df2 = pd.DataFrame()
# df2['DataSet_name'] = [DataSet_name]
# df2['Total number of trajectories'] = len(Track_length_in_Points)
# df2["Total numer of timepoints"]  = sum(Track_length_in_Points)
# df2["Mean trajectory length [timepoints]"]  = statistics.mean(Track_length_in_Points)
# df2["Total of trajectories [sec]"] = df2["Total numer of timepoints"]* 0.05
# df2["Mean trajectory length [sec]"] = df2["Mean trajectory length [timepoints]"]*0.05
# df2['Mean Stepsize [nm]'] = statistics.mean(Stepsizes)
# df2["Mean Speed [um/s]"] = sum(All_Speed)/len(All_Speed)
# df2["Max Speed [um/s]"] = max(All_Speed)
# df2["Mean total Ort displacement per trajectory [nm]"]  = statistics.mean(Total_Ort_positionS)
# df2["Mean total Par displacement per trajectory [nm]"] = statistics.mean(Total_Par_displacementS)
# df2["Max total Ort displacement per trajectory [nm]"] = max(Total_Ort_positionS)
# df2["Max total Par displacement per trajectory [nm]"] = max(Total_Par_displacementS)
# df2['Fraction_too_root_Y_positive'] = To_root_Y_positive/len(Par_displacements)
# df2['Fraction_too_tip_Y_negative'] = To_tip_Y_negative/len(Par_displacements)

# #save dataframe
# header = ['DataSet_name', 
#         'Total number of trajectories',
#         'Total numer of timepoints', 
#         'Total of trajectories [sec]',
#         'Mean trajectory length [timepoints]', 
#         'Mean Stepsize [nm]',
#         'Mean velocity [um/s]',
#         'Max velocity [nm/s]',
#         'Mean total x displacement per trajectory [nm]',
#         'Mean total y displacement per trajectory [nm]',
#         'Max total x displacement per trajectory [nm]',
#         'Max total y displacement per trajectory [nm]',
#         'Fraction_too_root_Y_positive',
#         'Fraction_too_tip_Y_negative']
# path = r"C:\Users\vkrugten\surfdrive\Programming\Results_210818_ExNo_60_FM4-64vkrugten_20210818_fm4-64_NiceOne\0-6000_molfiles\\" 
# name = "ResultsSummary_"+ DataSet_name + ".txt"

# df_ResultSum = df2.transpose()# transposes dataframe(2) to better readable format
# df_ResultSum.to_csv(SavePath + name, sep='\t', encoding='utf-8')

# print(df_ResultSum)

#%% Create and save figures of results of loop

# # plt.legend(framealpha=0, loc='best', bbox_to_anchor=(0.8, 0.53, 0.5, 0.5)) #labelcolor=COLOR,
# # plt.xlabel("frame #")
# # plt.ylabel("velocity (\u03BCm/s)")     

## save figure at specified path and name
# nameLoopFigure = "VelocityoverFrame"
# plt.savefig(SavePath + nameLoopFigure, transparent = True, bbox_inches="tight", dpi=300)

# Histogram for Speed distribution
fig2 =plt.figure(figsize=(10,10))
ax2 = fig2.add_subplot(1, 1, 1)
ax2.hist(All_Speed, bins=np.arange(min(All_Speed), max(All_Speed) + 2, 0.15),color='r', edgecolor='black')
ax2.set_xlabel("Speeds (\u03BCm/s) (#euclidean p2p dist/time)")
ax2.set_ylabel("count")
ax2.locator_params(axis='x', nbins=18)
ax2.set_xlim([-0.5, 16])

# Histogram for Velocity # Velocity from euclidean p2p dist/time
fig21 =plt.figure(figsize=(10,10))
ax21 = fig21.add_subplot(1, 1, 1)
ax21.hist(All_Vel, bins=500,color='r', edgecolor='black')
ax21.set_xlabel("Velocity(\u03BCm/s) (# Velocity from euclidean p2p dist/time)")
ax21.set_ylabel("count")
ax21.locator_params(axis='x', nbins=18)
ax21.set_xlim([-10, 10])

# #Histogram of Velocity #from 1D displacement along drawn spline, MATLAB column 9
fig11 =plt.figure(figsize=(10,10))
ax11 = fig11.add_subplot(1, 1, 1)
ax11.hist(All_Vel_Along_Spline, bins=500, color='#ff33f6', edgecolor='black')
ax11.set_xlabel("Velocity (#from 1D displacement along drawn spline, MATLAB column 9)")
ax11.set_ylabel("count")
ax11.set_xlim([-10, 10])

# #Histogram of Velocity #euclidean speed from first point till point i, MATLAB column 8
fig12 =plt.figure(figsize=(10,10))
ax12 = fig12.add_subplot(1, 1, 1)
ax12.hist(All_Vel_Transposed, bins=500, color='#ff33f6', edgecolor='black')
ax12.set_xlabel("Velocity (#euclidean speed from first point till point i, MATLAB column 8)")
ax12.set_ylabel("count")
ax12.set_xlim([-10, 10])

# # Histogram for Total_Ort_displS
fig3 =plt.figure(figsize=(10,10))
ax3 = fig3.add_subplot(1, 1, 1)
ax3.hist(Total_Ort_displS,color='b', edgecolor='black')
ax3.set_xlabel("Total Orthogonal displacement per trajectory (nm)")
ax3.set_ylabel("count")
# fig3.plt.locator_params(axis='x', nbins=18)

# # Histogram for rolling average Orthogonal displacements
# fig4 =plt.figure(figsize=(10,10))
# ax4 = fig4.add_subplot(1, 1, 1)
# ax4.hist(Total_Ort_positionS_RolAVG10, bins=400,color='#3fff33', edgecolor='black')
# ax4.set_xlabel("Rolling average Orthogonal displacement (nm)")
# ax4.set_ylabel("count")
# ax4.set_xlim([-250, 250])

# ## Histogram ort displacements
fig5 =plt.figure(figsize=(20,10))
ax5 = fig5.add_subplot(1, 1, 1)
ax5.hist(Ort_positions, bins=90,color='#33ffff', edgecolor='black') 
ax5.set_xlabel("Orthogonal positions (\u03BCm)")
ax5.set_ylabel("count")
ax5.set_xlim([-1.5, 1.5])

# #histogram for original orth displacements
# fig51 =plt.figure(figsize=(20,10))
# ax51 = fig51.add_subplot(1, 1, 1)
# ax51.hist(Ort_ori_displacements, bins=2000, edgecolor='black') 
# ax51.set_xlabel("Orthogonal ori displacements (nm)")
# ax51.set_ylabel("count")
# ax51.set_xlim([-250, 250])

# # Histogram for Total Parallel displacements
# fig6 =plt.figure(figsize=(10,10))
# ax6 = fig6.add_subplot(1, 1, 1)
# # binsize6 = np.arange(min(Total_Par_displacementS), max(Total_Par_displacementS) + 50, 750)
# ax6.hist(Total_Par_displacementS, bins=500,color='g', edgecolor='black')
# ax6.set_xlabel("Total Parallel displacement per trajectory (nm)")
# ax6.set_ylabel("count")

## Histogram Displ_along_spline_p2pS
fig7 =plt.figure(figsize=(20,10))
ax7 = fig7.add_subplot(1, 1, 1)
ax7.hist(Displ_along_spline_p2pS, bins=200, color='m', edgecolor='black') 
ax7.set_xlabel("Displ_along_spline_p2pS (nm)")
ax7.set_ylabel("count")
# ax7.set_xlim([-750, 750])
# 

# #histogram of Transposed_euclidean_dist_p2pS
fig71 =plt.figure(figsize=(20,10))
ax71 = fig71.add_subplot(1, 1, 1)
ax71.hist(Transposed_euclidean_dist_p2pS, bins=2000,color='m', edgecolor='black') 
ax71.set_xlabel("Transposed_euclidean_dist_p2pS (nm)")
ax71.set_ylabel("count")
ax71.set_xlim([-750, 500])

# #stack bar plot for fractions anterograde en retrograde
# # fig8, ax8 = plt.subplots(figsize=(1,8))
# # Labels = ['']
# # ToRoot = df2['Fraction_too_root_Y_positive']
# # ToTip = df2['Fraction_too_tip_Y_negative']
# # Width = 0.2
# # ax8.bar(Labels, ToRoot, Width, color='g')
# # ax8.bar(Labels, ToTip, Width, bottom=ToRoot, color='r')
# # ax8.set_ylabel('Fraction too root(green) and tip(red)')
# # ax8.tick_params(bottom = False)
# # ax8.spines['top'].set_visible(False)
# # ax8.spines['right'].set_visible(False)
# # ax8.spines['bottom'].set_visible(False)
# # ax8.spines['left'].set_visible(False)
# # ax8.annotate("", xy=(0, 0.25), xycoords='data',xytext=(0, 0.05), textcoords='data', arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
# # ax8.annotate("", xy=(0, 0.35), xycoords='data',xytext=(0, 0.95), textcoords='data', arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))

# ## Histogram for Euclidian displacements/stepsize
# fig9 =plt.figure(figsize=(10,10))
# ax9 = fig9.add_subplot(1, 1, 1)
# # binsize9 = np.arange(min(Stepsizes), max(Stepsizes) + 50, 10)
# ax9.hist(Stepsizes, bins=500 ,color='#ff33f6', edgecolor='black')
# ax9.set_xlabel("Stepsize / euclidian displacement (nm)")
# ax9.set_ylabel("count")
# ax9.set_xlim([-10, 800])

# #Histogram of track lengths
# # fig10 =plt.figure(figsize=(10,10))
# # ax10 = fig10.add_subplot(1, 1, 1)
# # # binsize10 = np.arange(min(Stepsizes), max(Stepsizes) + 50, 10)
# # ax10.hist(Track_length_in_Points, bins=40, color='#ff33f6', edgecolor='black')
# # ax10.set_xlabel("Track_length_in_Points")
# # ax10.set_ylabel("count")
# # # ax10.set_xlim([-10, 800])

# Histogram of Ort_displacements_p2pS
fig12 =plt.figure(figsize=(10,10))
ax12 = fig12.add_subplot(1, 1, 1)
ax12.hist(Ort_displacements_p2pS, bins=2000, color='#ff33f6', edgecolor='black')
ax12.set_xlabel("Ort_displacements_p2pS")
ax12.set_ylabel("count")
ax12.set_xlim([-300, 300])

#%% Save figure at specified path and name if Save_figures = Y
if Save_figures =='Y':
    fig2.savefig(SavePath + "SpeedHistogram" + ".png", transparent = True, bbox_inches="tight", dpi=300)
    fig3.savefig(SavePath + "Total Orthogonal displacement per trajectory (nm)" + ".png", transparent = True, bbox_inches="tight", dpi=300)
    fig4.savefig(SavePath + "Rolling average Orthogonal displacement (nm)" + ".png", transparent = True, bbox_inches="tight", dpi=300)
    fig5.savefig(SavePath + "Orthogonal displacements (nm)" + ".png", transparent = True, bbox_inches="tight", dpi=300)
    fig6.savefig(SavePath + "Total Parallel displacement per trajectory (nm)" + ".png", transparent = True, bbox_inches="tight", dpi=300)
    fig7.savefig(SavePath + "Parallel displacements (nm)" + ".png", transparent = True, bbox_inches="tight", dpi=300)
    fig8.savefig(SavePath + "'Fraction too root(green) and tip(red)'" + ".png", transparent = True, bbox_inches="tight", dpi=300)
    fig9.savefig(SavePath + "Stepsize / euclidian displacement (nm)" + ".png", transparent = True, bbox_inches="tight", dpi=300)
    fig10.savefig(SavePath + "Track_length_in_Points" + ".png", transparent = True, bbox_inches="tight", dpi=300)
    