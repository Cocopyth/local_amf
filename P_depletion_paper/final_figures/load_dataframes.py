from amftrack.util.sys import (
    get_analysis_folders,
    get_time_plate_info_from_analysis,
    get_time_hypha_info_from_analysis,
    get_global_hypha_info_from_analysis
)
import numpy as np
import pandas as pd
import os
root_dir = r"C:\Users\coren\AMOLF-SHIMIZU Dropbox\DATA"
analysis_dir = "PRINCE_ANALYSIS"
def get_plate_info(plates, time_plate_info, folders,min_time = -10,max_time = 100):
    plate_speeds, plate_density, plate_density_biov, plate_radius, plate_radius_SA = {}, {}, {}, {}, {}
    plate_strain, plate_start, plate_SA_density, fungalC, root = {}, {}, {}, {}, {}
    
    for plate_id in plates:
        plate = int(plate_id.split('_')[0])
        time_plate_info_plate = time_plate_info.loc[time_plate_info['unique_id'] == int(plate_id)]
        folders_plate = folders.loc[folders['unique_id'] == int(plate_id)]
        
        time_plate_info_plate_median = time_plate_info_plate.loc[
            time_plate_info_plate['time_since_begin_h_shifted'].between(-min_time,max_time)]
        
        plate_speeds[plate_id] = np.nanmedian(time_plate_info_plate['hull_speed'])
        plate_density[plate_id] = np.nanmedian(time_plate_info_plate_median['density'])
        plate_density_biov[plate_id] = np.nanmedian(time_plate_info_plate_median['density_biovolume'])
        plate_radius[plate_id] = np.nanmedian(
            np.sqrt(time_plate_info_plate_median['density_biovolume'] / time_plate_info_plate_median['density'] / np.pi))
        plate_radius_SA[plate_id] = np.nanmedian(
            time_plate_info_plate_median['density_SA'] / time_plate_info_plate_median['density'] / (2 * np.pi))
        plate_strain[plate_id] = folders_plate['strain'].iloc[0]
        plate_start[plate_id] = folders_plate['CrossDate'].iloc[0]
        fungalC[plate_id] = folders_plate['fungalC'].iloc[-1]
        root[plate_id] = folders_plate['root'].iloc[-1]
        plate_SA_density[plate_id] = np.nanmedian(time_plate_info_plate_median['density_SA'])

    return plate_speeds, plate_density, plate_density_biov, plate_radius, plate_radius_SA, \
           plate_strain, plate_start, plate_SA_density, fungalC, root

def build_dataframe(plates, plate_speeds, plate_density, plate_density_biov, plate_radius, plate_radius_SA,
                    plate_strain, plate_start, plate_SA_density, fungalC, root):
    df = pd.DataFrame({
        'plate_id': plates,
        'start': [plate_start[x] for x in plates],
        'speed': [plate_speeds[x] for x in plates],
        'density_SA': [plate_SA_density[x] for x in plates],
        'density_BV': [plate_density_biov[x] for x in plates],
        'density_L': [plate_density[x] for x in plates],
        'radius': [plate_radius[x] for x in plates],
        'radius_SA': [plate_radius_SA[x] for x in plates],
        'strain': [plate_strain[x] for x in plates],
        'fungalC': [fungalC[x] for x in plates],
        'root': [root[x] for x in plates],
    })

    df['strain'] = df['strain'].replace("'A5sp3'", "'A5'")
    df['fungalC'] = df['fungalC'].replace(np.nan, "100P100N")
    df['BV_growth_coeff'] = df['speed']**2 * df['density_BV']
    df['SA_growth_coeff'] = df['speed']**2 * df['density_SA'] * np.pi
    df['L_growth_coeff'] = df['speed']**2 * df['density_L']
    df['SA_growth_coeff2'] = df['L_growth_coeff'] * df['radius_SA'] * 2 * np.pi
    df['density_SA'] /= 1e6
    df['density_BV'] /= 1e9
    df['density_L'] /= 1e3
    return df

def load_all(plates):
    all_analysis_folders = get_analysis_folders()
    analysis_folders = all_analysis_folders.loc[all_analysis_folders['unique_id'].isin(plates)]
    folders, time_plate_info = get_time_plate_info_from_analysis(analysis_folders, False)

    time_plate_info = time_plate_info.replace(-1.0, np.nan)
    time_plate_info["density"] = time_plate_info["tot_length_study"] / time_plate_info["area_sep_comp"]
    time_plate_info["density_biovolume"] = time_plate_info["tot_biovolume_study"] / time_plate_info["area_sep_comp"]
    time_plate_info["density_SA"] = time_plate_info["tot_surface_area_study"] / time_plate_info["area_sep_comp"]
    time_plate_info['radius'] = np.sqrt(time_plate_info['tot_biovolume_study']/time_plate_info['tot_length_study']/2/np.pi)
    time_plate_info['radius_SA'] = time_plate_info['tot_biovolume_study']/time_plate_info['tot_surface_area_study']*2
    df_sum = load_summary()
    df_sum['FungalSide'] = df_sum['FungalSide'].fillna("100P/100N/100C")   
    df_sum['FungalSide'] = df_sum['FungalSide'].replace("100P100N100C","100P/100N/100C")
    df_sum['FungalSide'] = df_sum['FungalSide'].replace("100P100N100C","100P/100N/100C")

    df_sum['treatment'] = df_sum['treatment'].replace("1P100N100C","1P/100N/100C")
    final_data = merge_summary_df(time_plate_info,df_sum,folders,plates)
    plate_speeds, plate_density, plate_density_biov, plate_radius, plate_radius_SA, \
    plate_strain, plate_start, plate_SA_density, fungalC, root = get_plate_info(plates, final_data, folders)

    df = build_dataframe(plates, plate_speeds, plate_density, plate_density_biov, plate_radius, plate_radius_SA,
                         plate_strain, plate_start, plate_SA_density, fungalC, root)
    merged_df = df.merge(df_sum, left_on='plate_id', right_on='unique_id', how='inner')
    merged_df['FungalSide'] = merged_df['FungalSide'].fillna("100P/100N/100C")   
    merged_df['age'] = merged_df['age'].dt.days.astype(int)
    merged_df['FungalSide'] = merged_df['FungalSide'].replace("100P100N100C","100P/100N/100C")
    merged_df['FungalSide'] = merged_df['FungalSide'].replace("100P100N100C","100P/100N/100C")

    merged_df['treatment'] = merged_df['treatment'].replace("1P100N100C","1P/100N/100C")
    return merged_df,final_data

def load_summary():
    df_sum = pd.read_excel(os.path.join(root_dir, analysis_dir, "plate_summary2.xlsx"))
    df_sum['real_root'] = df_sum['root']
    df_sum.loc[df_sum['real_root'] == "EN daurus carota", 'real_root'] = "Carrot Vasilis"
    df_sum.loc[df_sum['real_root'] == "carrotV", 'real_root'] = "Carrot Vasilis"
    df_sum.loc[df_sum['real_root'] == "CarrotEN", 'real_root'] = "Carrot Vasilis"

    df_sum.loc[df_sum['real_root'] == "Carrot", 'real_root'] = "Carrot Toby"

    df_sum.loc[df_sum['real_root'] == "carrot", 'real_root'] = "Carrot Toby"
    df_sum.loc[df_sum['real_root'] == "CH cichorium intybus", 'real_root'] = "Chichorium"
    df_sum.loc[df_sum['real_root'] == "BE nicotiana benthamiana", 'real_root'] = "Nicotiana"
    df_sum['start'] = pd.to_datetime(df_sum['start'], errors='coerce')
    df_sum['crossed']  = pd.to_datetime(df_sum['crossed'], errors='coerce')
    df_sum['date_fungus']  = pd.to_datetime(df_sum['age fungus'], errors='coerce')
    df_sum['age'] = df_sum['crossed']-df_sum['start']

    return(df_sum)

def merge_summary_df(time_plate_info,df_sum,folders,plates,thresh_area = 200):
    concatenated_data = []

    for j in range(0, len(plates)):
        plate_id = plates[j]
        # print(plate_id)
        # print(plate_id)
        plate = int(plate_id.split('_')[0])

        time_plate_info_plate = time_plate_info.loc[time_plate_info['unique_id'] == int(plate_id)]
        time_plate_info_plate = time_plate_info_plate.sort_values(by='time_since_begin_h')
        folders_plate = folders.loc[folders['unique_id'] == int(plate_id)]
        late_start = len(time_plate_info_plate[time_plate_info_plate['area_sep_comp'] >= thresh_area]['time_since_begin_h'])==len(time_plate_info_plate)
        early_stop = len(time_plate_info_plate[time_plate_info_plate['area_sep_comp'] >= thresh_area])==0
        time_plate_info_plate['late_start']=late_start
        time_plate_info_plate['early_stop']=early_stop
        if not early_stop:
            t_shift = time_plate_info_plate[time_plate_info_plate['area_sep_comp'] >= thresh_area]['time_since_begin_h'].iloc[0]
       
        else:
            t_shift = 0
        time_plate_info_plate['time_since_begin_h_shifted'] = time_plate_info_plate['time_since_begin_h'] - t_shift
        # Add the strain information to the DataFrame
        # time_plate_info_plate['is_in'] = time_plate_info_plate['timestep']<timestep[plate]
        time_plate_info_plate['ZMUV_radius'] = (time_plate_info_plate['radius']-np.mean(time_plate_info_plate['radius']))
        time_plate_info_plate['ZMUV_SA'] = (time_plate_info_plate['radius']**2-np.mean(time_plate_info_plate['radius']**2))
        time_plate_info_plate['real_root'] = df_sum[df_sum['unique_id']==plate_id]['real_root'].iloc[0]
        time_plate_info_plate['fungalC'] = df_sum[df_sum['unique_id']==plate_id]['FungalSide'].iloc[0]
        time_plate_info_plate['treatment'] = df_sum[df_sum['unique_id']==plate_id]['treatment'].iloc[0]
        

        time_plate_info_plate['strain'] = df_sum[df_sum['unique_id']==plate_id]['fungus'].iloc[0]
        
        time_plate_info_plate['strain_unique_id'] = time_plate_info_plate['strain'].astype(str) + '_' + time_plate_info_plate['unique_id'].astype(str)
        # Append this modified DataFrame to a list
        concatenated_data.append(time_plate_info_plate)

    # Concatenate all the DataFrames in the list into a single DataFrame
    final_data = pd.concat(concatenated_data)

    # Replace -1.0 with NaN
    final_data = final_data.replace(-1.0, np.nan)
    bin_size = 10
    final_data = final_data[final_data['time_since_begin_h_shifted'].notna()]
    
    final_data['time_hour_binned'] = final_data['time_since_begin_h_shifted'].astype(int)//bin_size*bin_size
    final_data['strain'] = final_data['strain'].replace("A5sp3","A5")
    final_data['tot_volume'] = final_data['spore_volume']+final_data['tot_biovolume_study']
    #returns only the long enough imaged
    select = final_data.loc[final_data['late_start']==False]
    select = select.loc[select['early_stop']==False]
    return(select)
