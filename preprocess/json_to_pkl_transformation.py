import json
import pandas as pd
import json
from tqdm import tqdm


def load_jsonl_data(file_path):
    """Load data from a JSONL (JSON Lines) file with a progress bar."""
    data = []

    # Open the file
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Loading JSONL", unit="line"):
            data.append(json.loads(line))

    return pd.DataFrame(data)


if __name__ == '__main__':
    path_1 = '../Datasets/Movies_and_TV.jsonl'
    meta_path_1 = '../Datasets/meta_Movies_and_TV.jsonl'
    path_2 = '../Datasets/Musical_Instruments.jsonl'
    meta_path_2 = '../Datasets/meta_Musical_Instruments.jsonl'
    path_3 = '../Datasets/All_Beauty.jsonl'
    meta_path_3 = '../Datasets/meta_All_Beauty.jsonl.jsonl'
    path_4 = '../Datasets/Amazon_Fashion.jsonl'
    meta_path_4 = '../Datasets/meta_Amazon_Fashion.jsonl'

    df_1 = load_jsonl_data(path_1)
    meta_df_1 = load_jsonl_data(meta_path_1)
    df_2 = load_jsonl_data(path_2)
    meta_df_2 = load_jsonl_data(meta_path_2)
    df_3 = load_jsonl_data(path_3)
    meta_df_3 = load_jsonl_data(meta_path_3)
    df_4 = load_jsonl_data(path_4)
    meta_df_4 = load_jsonl_data(meta_path_4)

    df_1.to_pickle('../Datasets/Movies_and_TV.pkl')
    meta_df_1.to_pickle('../Datasets/meta_Movies_and_TV.pkl')
    df_2.to_pickle('../Datasets/Musical_Instruments.pkl')
    meta_df_2.to_pickle('../Datasets/meta_Musical_Instruments.pkl')
    df_3.to_pickle('../Datasets/beauty.pkl')
    meta_df_3.to_pickle('../Datasets/meta_beauty.pkl')
    df_4.to_pickle('../Datasets/fashion.pkl')
    meta_df_4.to_pickle('../Datasets/meta_fashion.pkl')
