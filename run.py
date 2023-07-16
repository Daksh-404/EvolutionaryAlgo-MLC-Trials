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
from Classifiers.StaC.StaCv1 import StackedChaining
from utils.utils import construct_conditional_entropy_matrix

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
    name,
    iterations=1
):
    info_table = {}
    info_table["iteration"] = []
    for metric in metrics_list:
        info_table[metric] = []
    for i in range(iterations):
        geneticObj = GeneticAlgorithm(len(dset_classes.columns), ce_matrix)
        s = time.time()
        label_order = geneticObj.genetic_algorithm(
            population_size=200,
            num_generations=generation,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elitism_rate=elitism,
        )

        e = time.time()
        tr = e - s
        tp = 0
        # Getting results on the given approach for MLC
        #print("#################################")
        print(name)
        #print("#################################")
        stacked_chaining = StackedChaining(
            dset_features, dset_classes, split, label_order, name
        )

        stacked_chaining.run()
        for metric in metrics_list:
            info_table[metric].append(stacked_chaining.metric_values[metric])
        info_table["iteration"].append(i + 1)
        print("training time: ", round(tr + stacked_chaining.training_time, 3))
        print("prediction time: ", round(stacked_chaining.prediction_time, 3))
    return info_table


def run(
    dset_classes,
    dset_features,
    generations,
    mutation_rate_trial_values,
    crossover_rate_trial_values,
    elitism_trial_values,
    split,
    name,
    iterations=1
):
    # "LABEL RANKING LOSS",
    # "COVERAGE ERROR",
    # "ZERO ONE LOSS",
    # "AVERAGE PRECISION SCORE",
    # "LABEL RANKING APR",
    # "ONE ERROR",
    metrics_list = [
        "HAMMING LOSS",
        "SUBSET ACCURACY",
        "JACCARD MACRO",
        "JACCARD MICRO",
        "JACCARD SAMPLES",
        "F1 SCORE MACRO",
        "F1 SCORE MICRO",
        "F1 SCORE SAMPLES",
    ]
    parameter_names = ["Generations", "Mutation rate", "Crossover rate", "Elitism Rate"]

    info_table = {}
    info_table["iteration"] = []
    for parameter in parameter_names:
        info_table[parameter] = []

    for metric in metrics_list:
        info_table[metric] = []

    # Generating the optimal permutation for label ordering
    ce_matrix = construct_mutual_information_matrix(dset_classes.to_numpy())
    for gen_index in tqdm(range(len(generations))):
        generation = generations[gen_index]
        for mutation_rate in mutation_rate_trial_values:
            for cross_index in tqdm(range(len(crossover_rate_trial_values))):
                crossover_rate = crossover_rate_trial_values[cross_index]
                for elite_index in tqdm(range(len(elitism_trial_values))):
                    elitism = elitism_trial_values[elite_index]
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
                        name=name,
                        iterations=iterations
                    )
                    for key in tmp_info:
                        info_table[key] += tmp_info[key]
                    info_table["Generations"] += [generation] * iterations
                    info_table["Mutation rate"] += [mutation_rate] * iterations
                    info_table["Crossover rate"] += [crossover_rate] * iterations
                    info_table["Elitism Rate"] += [elitism] * iterations
    return info_table

def run_pipeline():
    print("PIPELINE INITIATED!")
    for file in os.listdir():
        if file.count("testing") == 0:
            split = int(file.split("_")[1].split(".")[0])
            name = file.split("_")[0]
            dset = pd.read_csv(os.path.join(path_to_dataset, file))
            generations = [100]
            mutation_rate_trial_values = list(np.linspace(start=0.03, stop=0.03, num=1))
            crossover_rate_trial_values = list(np.linspace(start=0.9, stop=0.9, num=1))
            elitism_trial_values = [8]

            # Extracting the class and the feature dataframes from the data
            start = 1
            if name == "genbase":
                start = start + 1
            end = split + 1

            if name == "rcv":
              start = 0
              end = end - 1
            #print(start, end)
            dset = dset.sample(frac=1).reset_index(drop=True)
            dset_features = dset.iloc[:, start : end]
            dset_classes = dset.iloc[:, end :]
            info_table = run(
                generations=generations,
                dset_features=dset_features,
                dset_classes=dset_classes,
                mutation_rate_trial_values=mutation_rate_trial_values,
                crossover_rate_trial_values=crossover_rate_trial_values,
                elitism_trial_values=elitism_trial_values,
                split=split,
                name=name,
                iterations=1
            )
            csv = pd.DataFrame(data=info_table)
            csv.to_csv(f"{path_to_dataset}/{name}_test_results.csv")
    print("PIPELINE CLOSED!")


if __name__ == "__main__":
    print("!!!!!!!!!!!!!!!!ALGORITHM INITIATED!!!!!!!!!!!!!!!!")
    run_pipeline()
    print("!!!!!!!!!!!!!!!!ALGORITHM ENDED!!!!!!!!!!!!!!!!!!!!")
