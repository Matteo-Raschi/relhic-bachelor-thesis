# crossmatching and create file for matching data and for non_matching data

import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

# Costanti
c = 299792.458  # Speed of light in km/s
max_angular_distance = 0.05  # maximum angular separation in degrees between cataloged objects(3 arcmin)
k_tolerance = 1  # tolerance factor used in comparing redshifts

# function for angular separation
def angular_separation_with_error(ra1, dec1, ra2, dec2, err1, err2):
    ra1, dec1, ra2, dec2 = map(np.radians, [ra1, dec1, ra2, dec2]) # convert grad to radiant (numpy need rad)
    delta_ra = ra2 - ra1 # calculate difference from the 2 point 
    delta_dec = dec2 - dec1
    a = np.sin(delta_dec / 2) ** 2 + np.cos(dec1) * np.cos(dec2) * np.sin(delta_ra / 2) ** 2 # Haversine formula to calculate angular separation
    separation = 2 * np.arcsin(np.sqrt(a)) # angolar separation in radiant
    separation_deg = np.degrees(separation) # convert angolar separation in degree
    error_deg = np.sqrt(err1**2 + err2**2) # Include the positional errors in the angular separation (sdss error missing after)
    return separation_deg, error_deg

# Load data
fashi_data = pd.read_csv("C:/Users/user/Downloads/fashi_data.csv", low_memory=False)
sdss_data = pd.read_csv("C:/Users/user/Downloads/processed_sdss_data.csv", low_memory=False)

# convert ra and dec in numerical values for fashi
columns_to_convert = ['ra', 'dec', 'z', 'z_err', 'ra-dec_err']
for col in columns_to_convert:
    if col in fashi_data.columns: #check that column exist in DataFrame, convert column in numeric, if error value became NaN
        fashi_data[col] = pd.to_numeric(fashi_data[col], errors='coerce')
        
# convert ra and dec in numerical values for sdss 
sdss_data['ra'] = pd.to_numeric(sdss_data['ra'], errors='coerce')
sdss_data['dec'] = pd.to_numeric(sdss_data['dec'], errors='coerce')
sdss_data['z'] = pd.to_numeric(sdss_data['z'], errors='coerce')
sdss_data['zERR'] = pd.to_numeric(sdss_data['zERR'], errors='coerce')

# Create a KDTree for the sdss data (file was to big my pc couldn't analyze, was taking so much time)
sdss_coords = np.radians(sdss_data[['ra', 'dec']].dropna().to_numpy()) # extract ra and dec from dataframe
sdss_tree = cKDTree(sdss_coords) # build the tree

# radiant maximum angular separation in radiant for kdtree
max_distance_rad = np.radians(max_angular_distance)

# matching fashi and sdss
matches = [] # inizializza liste per memorizzare le corrispondenze e non corrispondenze
non_matches = []
for i, row in fashi_data.iterrows(): # iteration of every row of fashi to match with sdss objects
    ra_f, dec_f, z_f, z_err_f, ra_dec_err_f = row['ra'], row['dec'], row['z'], row['z_err'], row['ra-dec_err'] # extract values necessary for fashi object
    if pd.notna(ra_f) and pd.notna(dec_f) and pd.notna(z_f) and pd.notna(z_err_f) and pd.notna(ra_dec_err_f): # verify that necessary values are not missing
        indices = sdss_tree.query_ball_point([np.radians(ra_f), np.radians(dec_f)], max_distance_rad) # search in the kdtree sdss objects that are in max_distance_rad
        matched = False # flag to indicate if there is atleast one corrispondecy
        for idx in indices: # iter on every index of sdss objects found in the max distance
            sdss_row = sdss_data.iloc[idx] # take the row in the sdss dataframe
            sdss_ra, sdss_dec, sdss_z, sdss_z_err = sdss_row['ra'], sdss_row['dec'], float(sdss_row['z']), float(sdss_row['zERR']) # extract relevant data from sdss object

            # calculate angular separation and include error
            separation, separation_err = angular_separation_with_error(ra_f, dec_f, sdss_ra, sdss_dec, ra_dec_err_f, 0)  # Assumiamo errore nullo per SDSS

            # calculate tolerance on redshift
            delta_z = k_tolerance * (z_err_f + sdss_z_err)

            # Check whether the redshift difference is within tolerance and the angular separation is acceptable
            if abs(sdss_z - z_f) <= delta_z and separation <= (max_angular_distance + separation_err):
                matches.append({ # if criteria are ok add dictonary with data matching pn matches list
                    'ID_FASHI': row['ID_FASHI'],
                    'FASHI_Name': row['Name'],
                    'FASHI_ra': ra_f,
                    'FASHI_dec': dec_f,
                    'FASHI_z': z_f,
                    'FASHI_z_err': z_err_f,
                    'FASHI_ra_dec_err': ra_dec_err_f,
                    'SDSS_objID': sdss_row['objID'],
                    'SDSS_ra': sdss_ra,
                    'SDSS_dec': sdss_dec,
                    'SDSS_z': sdss_z,
                    'SDSS_z_err': sdss_z_err,
                    'Separation': separation,
                    'Separation_err': separation_err
                })
                matched = True # add flag to indicate that there is atleast one correspondency, if not after examinating every match there are no correspondecy add the line at the non correspondency list
        if not matched:
            non_matches.append(row.to_dict())

# converting matches to DataFrame (table)
matches_df = pd.DataFrame(matches)
non_matches_df = pd.DataFrame(non_matches)

# saving results
matches_df.to_csv("fashi_sdss_matches.csv", index=False)
non_matches_df.to_csv("fashi_sdss_non_matches.csv", index=False)
print("Analisi completata! Risultati salvati in 'fashi_sdss_matches.csv' e 'fashi_sdss_non_matches.csv'.")
