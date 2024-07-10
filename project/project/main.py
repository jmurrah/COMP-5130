import pandas as pd
import argparse
import project.algorithms as algorithms


def convert_to_dataframe(file_path: str) -> pd.DataFrame:
    with open(file_path, "r") as file:
        raw_data = file.readlines()

    data = {"from-node-id": [], "to-node-id": []}

    for row in raw_data[4:]:
        data["from-node-id"].append(row.split("\t")[0].strip())
        data["to-node-id"].append(row.split("\t")[1].strip())

    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, help="Algorithm to run")
    algorithm = parser.parse_args().algo

    test_data = convert_to_dataframe("./project/datasets/test_data.txt")
    arxiv_small_dataset = convert_to_dataframe("./project/datasets/CA-GrQc.txt")
    dblp_large_dataset = convert_to_dataframe("./project/datasets/com-dblp.ungraph.txt")

    if algorithm == "k-means":
        # algorithms.k_means.k_means(5, test_data)
        algorithms.k_means.k_means(3, arxiv_small_dataset)
        # algorithms.k_means.k_means(10, dblp_large_dataset)
