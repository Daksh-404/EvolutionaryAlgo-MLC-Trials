import sys
import warnings
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

sys.path.insert(1, "./EvolutionaryAlgorithms/")
sys.path.insert(2, "./Classifiers/StaC")
warnings.filterwarnings("ignore")

from EvolutionaryAlgorithms.GeneticAlgorithm import GeneticAlgorithm
from Classifiers.StaC.StaCv3 import StackedChaining
from utils.CEMatrix import construct_conditional_entropy_matrix

path_to_dataset = os.getcwd() + "/Datasets"


def test_parameters(
    dset_classes,
    dset_features,
    ce_matrix,
    generation,
    mutation_rate,
    crossover_rate,
    elitism,
    split,
    metrics_list,
    iterations=1,
):
    info_table = {}
    for metric in metrics_list:
        info_table[metric]=[]
    for _ in range(iterations):
        geneticObj = GeneticAlgorithm(len(dset_classes.columns), ce_matrix)
        label_order = geneticObj.genetic_algorithm(
            population_size=100,
            num_generations=generation,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elitism_rate=elitism,
        )
        # Getting results on the given approach for MLC
        stacked_chaining = StackedChaining(
            dset_features, dset_classes, split, label_order
        )
        stacked_chaining.run()
        for metric in metrics_list:
            info_table[metric].append(stacked_chaining.metric_values[metric])
    for metric in metrics_list:
        info_table[metric] = np.mean(np.array(info_table[metric]))
    return info_table


def run(
    dset_classes,
    dset_features,
    generations,
    mutation_rate_trial_values,
    crossover_rate_trial_values,
    elitism_trial_values,
    split,
):
    metrics_list = [
        "HAMMING LOSS",
        "SUBSET ACCURACY",
        "LABEL RANKING LOSS",
        "COVERAGE ERROR",
        "ZERO ONE LOSS",
        "AVERAGE PRECISION SCORE",
        "LABEL RANKING APR",
        "JACCARD MACRO",
        "JACCARD MICRO",
        "JACCARD SAMPLES",
        "ONE ERROR",
        "F1 SCORE MACRO",
        "F1 SCORE MICRO",
        "F1 SCORE SAMPLES",
    ]
    parameter_names = ["Generations", "Mutation rate", "Crossover rate", "Elitism Rate"]

    info_table = {}
    for parameter in parameter_names:
        info_table[parameter] = []

    for metric in metrics_list:
        info_table[metric] = []

    # Generating the optimal permutation for label ordering
    ce_matrix = construct_conditional_entropy_matrix(dset_classes.to_numpy())
    for gen_index in tqdm(range(len(generations))):
        generation = generations[gen_index]
        for mutation_rate in mutation_rate_trial_values:
            for cross_index in tqdm(range(len(crossover_rate_trial_values))):
                crossover_rate = crossover_rate_trial_values[cross_index]
                print(f"Crossover rate: {crossover_rate}")
                for elite_index in tqdm(range(len(elitism_trial_values))):
                    elitism = elitism_trial_values[elite_index]
                    print(f"Elitism: {elitism}")
                    tmp_info = test_parameters(
                        dset_classes=dset_classes,
                        dset_features=dset_features,
                        ce_matrix=ce_matrix,
                        generation=generation,
                        mutation_rate=mutation_rate,
                        crossover_rate=crossover_rate,
                        elitism=elitism,
                        split=split,
                        metrics_list=metrics_list,
                        iterations=10
                    )
                    for metric in metrics_list:
                        info_table[metric].append(tmp_info[metric])
                    info_table["Generations"].append(generation)
                    info_table["Mutation rate"].append(mutation_rate)
                    info_table["Crossover rate"].append(crossover_rate)
                    info_table["Elitism Rate"].append(elitism)
    return info_table


def run_pipeline():
    print("PIPELINE INITIATED!")

    os.chdir(path_to_dataset)
    for file in os.listdir():
        if file.count("testing")==0:
            split = int(file.split("_")[1].split(".")[0])
            dset = pd.read_csv(os.path.join(path_to_dataset, file))
            generations = [100]
            mutation_rate_trial_values = list(np.linspace(start=0.01, stop=0.01, num=1))
            crossover_rate_trial_values = list(np.linspace(start=0.8, stop=0.8, num=1))
            elitism_trial_values = [14]
            # Extracting the class and the feature dataframes from the data
            dset_features = dset.iloc[:, :split]
            dset_classes = dset.iloc[:, split:]
            info_table = run(
                generations=generations,
                dset_features=dset_features,
                dset_classes=dset_classes,
                mutation_rate_trial_values=mutation_rate_trial_values,
                crossover_rate_trial_values=crossover_rate_trial_values,
                elitism_trial_values=elitism_trial_values,
                split=split,
            )
            csv = pd.DataFrame(data=info_table)
            csv.to_csv("testing_result.csv")
    print("PIPELINE CLOSED!")


if __name__ == "__main__":
    print("!!!!!!!!!!!!!!!!ALGORITHM INITIATED!!!!!!!!!!!!!!!!")
    run_pipeline()
    print("!!!!!!!!!!!!!!!!ALGORITHM ENDED!!!!!!!!!!!!!!!!!!!!")
