#__init__.py

import numpy as np
import pandas as pd
import MDAnalysis as mda
from collections import OrderedDict
from operator import itemgetter
from scipy.spatial import distance_matrix,distance

# type_mol for now needs to be protein/not protein/resname X/...

class Centroid_fpocket:
    def __init__(self, apo_pockets,type_mol):
        self.apo_pockets = apo_pockets
        self.type_mol = type_mol
        
    def mda(self):
        apo_pock = mda.Universe(self.apo_pockets)
        select_atom = apo_pock.select_atoms(self.type_mol)
        select_atom_pos = select_atom.atoms.positions
        select_atom_resids = select_atom.resids
        select_atom_resids=select_atom_resids
        select_atom_pos=select_atom_pos
        return (select_atom_resids,select_atom_pos)
        
    def mda_to_df(self):
        select_atom_resids, select_atom_pos = self.mda()
        result_STP = list(OrderedDict.fromkeys(select_atom_resids))
        data_STP = {'x': select_atom_pos[:, 0], 'y': select_atom_pos[:, 1], 'z': select_atom_pos[:, 2], 'name': select_atom_resids}
        df_STP = pd.DataFrame(data_STP)
        return (df_STP,result_STP)
    
    def center(self, nested_array_list):
        a = np.array(nested_array_list)
        mean = np.mean(a, axis=0)
        return mean[0], mean[1], mean[2]

    def calculate_centroid(self):
        df_STP,result_STP = self.mda_to_df()
        centroid = []
        for l in result_STP:
            STP1 = df_STP[df_STP['name'] == l]
            stp_pos1 = STP1[['x', 'y', 'z']].values
            centroid_point = self.center(stp_pos1)
            a = list(centroid_point)
            a.append(l)
            centroid.append(a)
        return centroid
    
class Compare_centroids:
    def __init__(self, centroid,holo_pdb,pdb_number):
        self.centroid=centroid
        self.holo_pdb=holo_pdb
        self.pdb_number=pdb_number
    
    @staticmethod
    def Extract_0(x,lst):
        return list(map(itemgetter(x), lst ))
    
    @staticmethod
    def distance_finder(one,two):
        [x1,y1,z1] = one  # first coordinates
        [x2,y2,z2] = two[:3]  # second coordinates
        v=(((x2-x1)**2)+((y2-y1)**2)+((z2-z1)**2))**(1/2)
        answer=(v,two[3])
        return answer

    def calculate_distance(self, max_distance=55.5):
        centroid_instance = Centroid_fpocket(self.holo_pdb, 'not protein')
        liga_resid,liga_pos = centroid_instance.mda()
    
        cal_post = []
        for liga_centroid in liga_pos:
            cal_pre = []
            for apo_centroid in self.centroid:
                cal = self.distance_finder(liga_centroid, apo_centroid)
                if cal[0] <= max_distance:
                    extend_argument=(cal[0], cal[1],self.pdb_number)
                    cal_pre.extend([extend_argument])
                else:
                    pass
            cal_post.append(cal_pre)
        flat_list = [num for sublist in cal_post for num in sublist]
        STP_pockets = list(dict.fromkeys([item[1] for item in flat_list]))
        return (STP_pockets,self.pdb_number)

class Retrive_pocket_residues:
    def __init__(self, single_pocket_path):
        self.single_pocket_path = single_pocket_path
        
    def md_analysis_fix(self,cutoff=55.5):
        liga_centroid_instance = Centroid_fpocket(self.single_pocket_path, 'not protein')
        liga_resid,liga_pos = liga_centroid_instance.mda()      
        
        protein_centroid_instance = Centroid_fpocket(self.single_pocket_path, 'protein')
        protein_resid,protein_pos = protein_centroid_instance.mda()      
        
        #This one does each atom of the ligand * each atom of protein - size of matrix is 2017 (woth 38 nested distance for 38 atoms in ligand) and 2017 atoms in protein
        dist_matrix = distance.cdist(protein_pos, liga_pos, 'euclidean')    
            
        list_prot_resid_distance=[]
        prot_resids=[]
        for i,j in zip((enumerate(dist_matrix)),protein_resid):
            if (min(i[1])) <= cutoff:
                evalue=(j,i[1])
                list_prot_resid_distance.append(evalue)
                prot_resids.append(j)
                
        return list_prot_resid_distance,prot_resids
        
    def return_path_resids(self):
        _, prot_resids = self.md_analysis_fix()
        orga_prot_resids = list(dict.fromkeys(prot_resids))
        complete_list=[(self.single_pocket_path,orga_prot_resids)]
        return complete_list




        
