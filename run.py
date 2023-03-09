import sys
import warnings
import os
import pandas as pd
import numpy as np

sys.path.insert(1, './EvolutionaryAlgorithms/')
sys.path.insert(2, './Classifiers/StaC')
warnings.filterwarnings('ignore')

from EvolutionaryAlgorithms.GeneticAlgorithm import GeneticAlgorithm
from Classifiers.StaC.StaCv3 import StackedChaining
from utils.CEMatrix import construct_conditional_entropy_matrix

path_to_dataset = "./Datasets"
def run_pipeline():
    print("PIPELINE INITIATED!")

    os.chdir(path_to_dataset)
    for file in os.listdir():
        split = int(file.split("_")[1].split(".")[0])
        dset = pd.read_csv(os.path.join(path_to_dataset, file))

        # Extracting the class and the feature dataframes from the data
        dset_features = dset.iloc[:, :split]
        dset_classes = dset.iloc[:, split:]

        # Generating the optimal permutation for label ordering 
        ce_matrix = construct_conditional_entropy_matrix(dset_classes.to_numpy())
        geneticObj = GeneticAlgorithm(len(dset_classes.columns), ce_matrix)
        label_order = geneticObj.genetic_algorithm()

        # Getting results on the given approach for MLC
        stacked_chaining = StackedChaining(dset_features, dset_classes, split, label_order)
        stacked_chaining.run()
        
    print("PIPELINE CLOSED!")

if __name__ == "__main__":
    print("!!!!!!!!!!!!!!!!ALGORITHM INITIATED!!!!!!!!!!!!!!!!")
    run_pipeline()
    print("!!!!!!!!!!!!!!!!ALGORITHM ENDED!!!!!!!!!!!!!!!!!!!!")


