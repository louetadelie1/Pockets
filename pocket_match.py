import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster


class Writing_pocket_pdb:
    def __init__(self, pocket_path,sample_path):
        self.pocket_path=pocket_path
        self.sample_path=sample_path
        
    def generate_pdb(self):
        with open(self.pocket_path[0], 'r') as filedata:
            for line in filedata:
                for x in self.pocket_path[1]:
                    y=str(x)+'     '      
                    if y in line:
                        with open(self.sample_path+(self.pocket_path[0].split('/')[-1]), 'a') as fil:
                            fil.write(line)
                        fil.close
                        
class Evaluating_PM:
    def __init__(self,pocket_match_out):
        self.pocket_match_out=pocket_match_out
        
    def similarity_mat(self):
        pd_data = pd.read_csv(self.pocket_match_out,sep="delimiter",header=None)
        clean_array=[(row[0].split()[0].split('__________')[0], row[0].split()[1].split('__________')[0], row[0].split()[2]) for _, row in pd_data.iterrows()]
        unique_ids = sorted(set(item[0] for item in clean_array))
        id_to_index = {pdb_name: index for index, pdb_name in enumerate(unique_ids)}
        num_pdb = len(unique_ids)
        similarity_matrix = np.zeros((num_pdb, num_pdb))
        for pdb1, pdb2, sim_value1 in clean_array:
            try:
                index1 = id_to_index[pdb1]
                index2 = id_to_index[(pdb2)]
                similarity_matrix[index1, index2] = float(sim_value1)
            except:
                pass
        linkage_matrix = linkage(similarity_matrix, metric='euclidean', method='complete')
        
        return linkage_matrix,unique_ids
    
    
    @classmethod
    def create_dict(cls, linkage_matrix,unique_ids,threshold=4.5):
        clusters = fcluster(linkage_matrix, threshold, criterion='distance')
        cluster_dict = {}
        for pdb, cluster_id in zip(unique_ids, clusters):
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append(pdb)

        for cluster_id, pdb_list in cluster_dict.items():
            print(f'Cluster {cluster_id}: {pdb_list}')
            
        return cluster_dict
    
    @classmethod
    def cluster_stats(cls,flat_pocket_paths_resid,cluster_dict):
        #stats_list=[]
        for cluster_id, pdb_list in cluster_dict.items():
            cal=((len(pdb_list)/len(unique_ids))*100)
            cluster_header = f'\033[1mCluster {cluster_id}\033[0m --'
            print(cluster_header, len(pdb_list), 'structures, making', cal, '% of total population')
            #stats_list.append(cluster_header+'\n')
            
        for y in pdb_list:
            name = (re.split(r'pair|.pdb', y)[1])
            name_re=holo_path+'pair'+name+'.pdb'
            for x in flat_pocket_paths_resid:
                if name_re==x[0]:
                    print(x)
                    #stats_list.append(x)
        #return stats_list
    
    