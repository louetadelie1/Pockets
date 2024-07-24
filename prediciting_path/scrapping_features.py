import glob
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from matplotlib import pyplot as plt
import pandas as pd
from collections import OrderedDict, Counter
from operator import itemgetter
import difflib
import re
import pickle
import mdtraj
#from scipy.spatial import distance_matrix, cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer as mlb
from scipy import stats
import mdtraj as md
from sklearn.decomposition import PCA
import os
import shutil
import subprocess
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import itertools
import statsmodels.api as sm
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import statistics
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler


path=('/Users/adelielouet/Documents/science/dd_proj/docking_other_systems/disordered_proteins/PED_and_HCG_ensembles/')

proteins = ['tau', 'piap','prion', 'abeta']
dfs = []
docking_score_df=[]

for pro in proteins:
    files_protein = glob.glob(path + pro + '/ensemble_100/pocket_input/pair*/save_2.txt', recursive=True)
    print(files_protein)
    data_dict = {}
    for filename in files_protein:
        with open(filename) as file:
            for line in file:
                first = line.rstrip()
                zn_id, score, pair_number = first.split('\t')
                pair_number = pair_number + '_' + pro
                if zn_id not in data_dict:
                    data_dict[zn_id] = {}
                data_dict[zn_id][pair_number] = float(score)

    df = pd.DataFrame(data_dict).T
    dfs.append(df)

docking_score_df = pd.concat(dfs)
docking_score_df = docking_score_df.groupby(docking_score_df.index).sum().dropna()
#docking_score_df = docking_score_df[docking_score_df['pair45_STP_5_tau'] != 0.00] # Accidentally docked full 5000 library on entire 100pockets of abeta oopsie
docking_score_df.loc['Avg_dock_score'] = docking_score_df.mean()

## THIS starts the SAR ANALYSIS
analysis_df=(docking_score_df.loc['Avg_dock_score'].to_frame()).T

data_dict = {}
df=[]

one_letter_map = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

kyte_doolittle = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}
rose = {
    'A': 0.74, 'R': 0.64, 'N': 0.63, 'D': 0.62, 'C': 0.91,
    'Q': 0.62, 'E': 0.62, 'G': 0.72, 'H': 0.78, 'I': 0.88,
    'L': 0.85, 'K': 0.52, 'M': 0.85, 'F': 0.88, 'P': 0.64,
    'S': 0.66, 'T': 0.70, 'W': 0.85, 'Y': 0.76, 'V': 0.86
}
welling = {
    'A': 0.3, 'R': -1.4, 'N': -0.5, 'D': -0.9, 'C': 2.0,
    'Q': -0.5, 'E': -0.7, 'G': 0.0, 'H': -0.5, 'I': 1.8,
    'L': 1.7, 'K': -1.8, 'M': 1.2, 'F': 1.5, 'P': -0.3,
    'S': -0.1, 'T': -0.4, 'W': 1.2, 'Y': 0.7, 'V': 1.5
}

# Use the following method--- cyrpotsite uses the same values:
wimley_white = {
    'A': 0.17, 'R': -1.01, 'N': -0.42, 'D': -0.58, 'C': 0.38,
    'Q': -0.47, 'E': -0.62, 'G': 0.01, 'H': -0.10, 'I': 1.81,
    'L': 1.70, 'K': -0.99, 'M': 1.23, 'F': 2.46, 'P': -0.13,
    'S': -0.13, 'T': -0.14, 'W': 2.07, 'Y': 1.60, 'V': 1.22
}


aromatic_pairs=['F','Y','W','H']

##### Looking at the following data:
# 1. SASA
# 2. Hydrophobicity
# 3. Polarity
# 4. Volume
# 5. Overall charge of pocket
# 6. Extracts exact residues involved in pocekts
# 7. One letter residues involved in pockets
# 8. RG
# 9. DSSP

# Goes into the original fpocket files in pocket_1.atm (or whatever its called) and extracts information directly from that file
# The rg and dssp are based on original pdb file (although local dssp comapres og pdb file to residues extracted from pocket_1.atm)

def reduce(residues_list):
    mylist=(list(re.sub("[^0-9]", "", str(x)) for x in residues_list))
    my_int_list = [int(i) for i in mylist]
    mylist2=(list(dict.fromkeys(my_int_list)))
    return(sorted(mylist2))

path_fpocket=('/Users/adelielouet/Documents/science/dd_proj/docking_other_systems/disordered_proteins/PED_and_HCG_ensembles/')

#path_fpocket=('/Users/adelielouet/Documents/science/dd_proj/docking_other_systems/disordered_proteins/PED_and_HCG_ensembles/tau/hcg_ensemble_tauk18_2022/')

for column in analysis_df.columns:
    protein=column.split('_')[-1]
    pair=column.split('_')[0]
    stp_number=column.split('_')[-2]
    if protein=='tau':
        pocket_pdb=path_fpocket+protein+'/ensemble/fpocket/'+pair+"_out/pockets/pocket1_atm.pdb"
        original_pdb=path_fpocket+protein+'/ensemble/fpocket'+'/'+pair+'.pdb'
    elif protein=='abeta':
        pocket_pdb=path_fpocket+protein+'/ensemble_100/fpocket/'+pair+"_out/pockets/pocket1_atm.pdb"
        original_pdb=path_fpocket+protein+'/ensemble_100/fpocket'+'/'+pair+'.pdb'
    else:
        pocket_pdb=path_fpocket+protein+'/ensemble/fpocket/'+pair+"_out/pockets/pocket"+stp_number+"_atm.pdb"
        #print(pocket_pdb)
        original_pdb=path_fpocket+protein+'/ensemble/fpocket/'+pair+'.pdb'

    with open(pocket_pdb) as file:
        data_dict[column] = {}
        all_indices=[]
        for line in file:
            if 'HEADER 0' in line:
                pocket_score = line.split(' ')[-1].strip()
                data_dict[column]['pocket_score'] = pocket_score
            if 'HEADER 10' in line:
                pocket_volume_CH = line.split(' ')[-1].strip()
                data_dict[column]['pocket_volume_CH'] = pocket_volume_CH
            if 'HEADER 4' in line:
                solvent = line.split(' ')[-1].strip()
                data_dict[column]['solvent'] = solvent
            if 'HEADER 6' in line:
                hydrophobicity = line.split(' ')[-1].strip()
                data_dict[column]['hydrophobicity'] = hydrophobicity
            if 'HEADER 7' in line:
                polarity = line.split(' ')[-1].strip()
                data_dict[column]['polarity'] = polarity
            if 'HEADER 11' in line:
                charge = line.split(' ')[-1].strip()
                data_dict[column]['charge'] = charge
            if 'HEADER 1' in line:
                drug_score = line.split(' ')[-1].strip()
                data_dict[column]['drug_score'] = drug_score
            if 'HEADER 2' in line:
                Number_of_alpha_spheres = line.split(' ')[-1].strip()
                data_dict[column]['Number_of_alpha_spheres'] = Number_of_alpha_spheres
            if 'HEADER 3' in line:
                Mean_alpha_sphere_radius = line.split(' ')[-1].strip()
                data_dict[column]['Mean_alpha_sphere_radius'] = Mean_alpha_sphere_radius
            if 'HEADER 14' in line:
                Proportion_alpha_sphere_radius = line.split(' ')[-1].strip()
                data_dict[column]['Proportion_alpha_sphere_radius'] = Proportion_alpha_sphere_radius
            if 'HEADER 8' in line:
                Amino_Acid_based_volume_Score = line.split(' ')[-1].strip()
                data_dict[column]['Amino_Acid_based_volume_Score'] = Amino_Acid_based_volume_Score
            if 'HEADER 9' in line:
                pocket_volume_MC = line.split(' ')[-1].strip()
                data_dict[column]['pocket_volume_MC'] = pocket_volume_MC
            if 'HEADER 12' in line:
                Local_hydrophobic_density_Score = line.split(' ')[-1].strip()
                data_dict[column]['Local_hydrophobic_density_Score'] = Local_hydrophobic_density_Score
            if 'HEADER 13' in line:
                Number_of_apolar_alpha_sphere = line.split(' ')[-1].strip()
                data_dict[column]['Number_of_apolar_alpha_sphere'] = Number_of_apolar_alpha_sphere
            if 'ATOM' in line:
                str_list=line.split(' ')
                str_list = (list(filter(None, str_list)))[1]
                all_indices.append(str_list)   #This is to calculate sasa score myself

            data_dict[column]['atom_pocket_indices'] = all_indices

    # This extracts residues in pockets
    traj_fpocket = md.load(pocket_pdb)
    residues = [residue for residue in traj_fpocket.topology.residues]
    residues_one_letter = [one_letter_map[residue.name] for residue in residues]
    data_dict[column]['residues_one_letter'] = residues_one_letter
    data_dict[column]['residues'] = residues
    n=([(re.findall(r'\d+', str(residue)))[0] for residue in residues])
    numbers = list(int(u) for u in n)
    data_dict[column]['resid'] = sorted(numbers)
    

    # Calcualting SASA:
    t = mdtraj.load(original_pdb)
    sasa = mdtraj.shrake_rupley(t)
    new_ind=(list(int(x) for x in all_indices))
    new_ind_twice = [x - 1 for x in new_ind]
    sasa_sum=sum(list(map(sasa[0].__getitem__, new_ind_twice)))
    data_dict[column]['sasa'] = sasa_sum

    # This calculates Radius of Gyr 
    traj_original_pdb = md.load(original_pdb)
    rg=md.compute_rg(traj_original_pdb, masses=None)
    data_dict[column]['rg'] = rg

    # Using Kyte-Doolittle to calculate hydrophobicty score
    resid_unique=(list(dict.fromkeys(residues_one_letter)))
    hydrophobicity_score_kd=sum(list(map(wimley_white.__getitem__, resid_unique)))
    data_dict[column]['wimley_white'] = hydrophobicity_score_kd

    # Looking at aromatic rings:
    number_aromatics=(len(list(u for u in residues_one_letter if u in aromatic_pairs)))
    data_dict[column]['number_aromatics'] = number_aromatics
     
    # Looking at number of atoms and number of unique residues involved:
    number_atoms=len(residues_one_letter)
    number_unique_residues=len(set(sorted(numbers)))
    data_dict[column]['total_pocket_atoms'] = number_atoms
    data_dict[column]['unique_residues'] = number_unique_residues

    # adding com
    res=list(set(sorted(numbers)))
    res=list(set(sorted(numbers)))
    atoms_list = []
    for res_idx in res:
        atoms = traj_original_pdb.topology.select(f'resid {res_idx}')
        atoms_list.append(atoms.tolist())
    merged = list(itertools.chain(*atoms_list))
    #atom_indices = [atom.index for atom in atoms_list.atoms]
    com = (md.compute_center_of_mass(traj_original_pdb.atom_slice(merged))).tolist()
    com=list(map(lambda x:x*10,com[0]))
    data_dict[column]['com'] = com

df = pd.DataFrame(data_dict)


regress_df = (pd.concat([analysis_df, df], axis=0)).T


## Making this data useable
for column in regress_df:
    try:
        regress_df[column] = regress_df[column].astype(float)
    except:
        pass

numeric_regress_df= regress_df.filter(['Avg_dock_score', 'pocket_score', 'drug_score',
       'Number_of_alpha_spheres', 'Mean_alpha_sphere_radius', 'solvent',
       'hydrophobicity', 'polarity', 'Amino_Acid_based_volume_Score',
       'pocket_volume_MC', 'pocket_volume_CH', 'charge',
       'Local_hydrophobic_density_Score', 'Number_of_apolar_alpha_sphere',
       'Proportion_alpha_sphere_radius', 'sasa', 'rg', 'wimley_white', 'number_aromatics','total_pocket_atoms','unique_residues'], axis=1)


comparisons=['pocket_score', 'drug_score',
       'Number_of_alpha_spheres', 'Mean_alpha_sphere_radius', 'solvent',
       'hydrophobicity', 'polarity', 'Amino_Acid_based_volume_Score',
       'pocket_volume_MC', 'pocket_volume_CH', 'charge',
       'Local_hydrophobic_density_Score', 'Number_of_apolar_alpha_sphere',
       'Proportion_alpha_sphere_radius', 'sasa', 'rg', 'wimley_white', 'number_aromatics','total_pocket_atoms','unique_residues']


# Analyze correlation - pearson

for testing_x in comparisons:
    correlation_coefficient, _ = stats.pearsonr(numeric_regress_df[testing_x],numeric_regress_df['Avg_dock_score'])
    #print("Correlation coefficient:", correlation_coefficient)

    X = ((numeric_regress_df[testing_x].to_numpy()).reshape(-1, 1))
    y = numeric_regress_df['Avg_dock_score']
    reg = LinearRegression().fit(X, y)

    alpha = reg.coef_[0]
    beta = reg.intercept_

    plt.text(0.05, 0.95, 'Correlation coefficient: ' + str(correlation_coefficient), transform=plt.gca().transAxes)
    
    plt.plot(X, reg.predict(X), color='red', label=f'Regression Line: y = {alpha:.2f} * x + {beta:.2f}')

    plt.scatter(numeric_regress_df[testing_x],numeric_regress_df['Avg_dock_score'])
    plt.xticks(rotation=90)
    plt.ylabel('Average Docked Score')
    plt.xlabel(testing_x)
    plt.title('Correlation '+ testing_x +' to average docking score')
    plt.show()
