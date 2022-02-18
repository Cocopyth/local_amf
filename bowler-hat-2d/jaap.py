import numpy as np
import scipy.io as sio
import pandas as pd
mat_file = sio.loadmat('210914_ExNo_68_FM4-64_60x_Pos2_Transposed')
for i,mol in enumerate(mat_file['Track_Info'][0][0]):
    dataframe = pd.DataFrame(data=mol, columns=["Frame Number", "Tme","x-coordinate","y-coordinate","Distance from the origin (first point of the track)","Distance from the reference point (base) along the spline","Distance parallel to the spline","P2P Velocities calculated from column 5","P2P Velocities calculated from column 6"])
    dataframe.to_csv(f'for_jaap/track_mol{i}.csv',sep="\t")