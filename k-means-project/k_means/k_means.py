import pandas as pd


def convert_to_dataframe(file_path: str) -> pd.DataFrame:
    with open(file_path, "r") as file:
        raw_data = file.readlines()

    data = {"from-node-id": [], "to-node-id": []}

    for row in raw_data[4:]:
        data["from-node-id"].append(row.split("\t")[0].strip())
        data["to-node-id"].append(row.split("\t")[1].strip())

    return pd.DataFrame(data)


def main():
    arxiv_small_dataset = convert_to_dataframe("./datasets/CA-GrQc.txt")
    dblp_large_dataset = convert_to_dataframe("./datasets/com-dblp.ungraph.txt")

    print(arxiv_small_dataset)
