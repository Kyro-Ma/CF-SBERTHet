import gc
import os
import sys
from transformers import (
    BertModel,
    BertTokenizer,
    DistilBertModel,
    DistilBertTokenizer,
    DistilBertTokenizerFast,
)
from sklearn.model_selection import train_test_split
from utils import load_dataset
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import warnings
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.data import HeteroData
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle
import random

# import nltk
# nltk.download('punkt_tab')

warnings.filterwarnings("ignore", category=FutureWarning, message="'DataFrame.swapaxes' is deprecated")


def train_and_evaluate(training_data, testing_data, items_dict):
    data_train = HeteroData()
    data_test = HeteroData()
    print_counter = 20000

    uid_train = {}
    iid_train = {}
    current_uid_train = 0
    current_iid_train = 0
    rate_count_train = len(training_data)
    counter = 0

    # map the id of user and items to numerical value
    for index, row in training_data.iterrows():
        if counter % print_counter == 0:
            print(str(round(counter / rate_count_train, 2) * 100) + '%')

        if row['user_id'] in uid_train.keys():
            pass
        else:
            uid_train[row['user_id']] = current_uid_train
            current_uid_train += 1

        if row['parent_asin'] in iid_train:
            pass
        else:
            iid_train[row['parent_asin']] = current_iid_train
            current_iid_train += 1

        counter += 1

    uid_test = {}
    iid_test = {}
    current_uid_test = 0
    current_iid_test = 0
    rate_count_test = len(testing_data)
    counter = 0
    # print("standardise user id and item id for testing train_edge_data")
    for index, row in testing_data.iterrows():
        # if counter % print_counter == 0:
        #     print(str(round(counter / rate_count_test, 2) * 100) + '%')

        if row['user_id'] in uid_test.keys():
            pass
        else:
            uid_test[row['user_id']] = current_uid_test
            current_uid_test += 1

        if row['parent_asin'] in iid_test:
            pass
        else:
            iid_test[row['parent_asin']] = current_iid_test
            current_iid_test += 1

        counter += 1

    # Add user node IDs (without features)
    data_train['user'].num_nodes = current_uid_train  # Number of users
    data_test['user'].num_nodes = current_uid_test
    item_features_train = []
    item_features_test = []
    counter = 0
    print("Getting item features (training)")
    for value in iid_train.keys():
        if counter % print_counter == 0:
            print(str(round(counter / len(iid_train.keys()), 2) * 100) + '%')

        target = items_dict[value]
        temp = target['title'].weight.detach().cpu().numpy().tolist()[0]
        # print(temp)
        item_features_train.append(temp)
        counter += 1

    counter = 0
    print("Getting item features (testing)")
    for value in iid_test.keys():
        if counter % print_counter == 0:
            print(str(round(counter / len(iid_test.keys()), 2) * 100) + '%')

        target = items_dict[value]
        temp = target['title'].weight.detach().cpu().numpy().tolist()[0]
        item_features_test.append(temp)
        counter += 1

    # item_features_train = np.array(item_features_train, dtype=np.float32)
    # item_features_test = np.array(item_features_test, dtype=np.float32)
    # print(item_features_train)

    # Adding item nodes with features
    data_train['item'].x = torch.tensor(item_features_train, dtype=torch.float).to(device)  # Item features (2D)
    data_test['item'].x = torch.tensor(item_features_test, dtype=torch.float).to(device)  # Item features (2D)

    # region training edges
    rating_edge_from_train, rating_edge_to_train = [], []
    rating_train = []
    verify_buy_from_train, verify_buy_to_train = [], []
    review_train = []
    review_edge_from_train, review_edge_to_train = [], []
    counter = 0
    store_item_dict_train = {key: [] for key in stores}
    same_store_edge_train = [[], []]

    for index, row in training_data.iterrows():
        if counter % print_counter == 0:
            print(str(round(counter / len(iid_test.keys()), 2) * 100) + '%')

        rating_edge_from_train.append(uid_train[row['user_id']])
        rating_edge_to_train.append(iid_train[row['parent_asin']])
        rating_train.append(row['rating'])
        store_item_dict_train[items_dict[row['parent_asin']]['store']].append(iid_train[row['parent_asin']])

        if row['text'] is not None:
            review_edge_from_train.append(uid_train[row['user_id']])
            review_edge_to_train.append(iid_train[row['parent_asin']])
            # 768 comes from A_llmrec SBERT dimensions
            review_train.append(torch.nn.Embedding(1, 768).weight.detach().cpu().numpy())

        if row['verified_purchase']:
            verify_buy_from_train.append(uid_train[row['user_id']])
            verify_buy_to_train.append(iid_train[row['parent_asin']])

        counter += 1

    # solve the repeated items in the store_item_dict and build same store edge
    for store in store_item_dict_train.keys():
        item_from_store = list(set(store_item_dict_train[store]))

        if len(item_from_store) < 2:
            pass
        for i in range(len(item_from_store)):
            for j in range(i, len(item_from_store)):
                same_store_edge_train[0].append(item_from_store[i])
                same_store_edge_train[1].append(item_from_store[j])

    # Convert List of NumPy Arrays to a Single NumPy Array
    review_train = np.array(review_train).tolist()

    # Adding edges and edge attributes
    data_train['user', 'rates', 'item'].edge_index = torch.tensor(
        [rating_edge_from_train, rating_edge_to_train], dtype=torch.long
    ).to(device)
    data_train['user', 'rates', 'item'].edge_attr = torch.tensor(rating_train, dtype=torch.float).to(device)
    data_train['item', 'rated_by', 'user'].edge_index = torch.tensor(
        [rating_edge_to_train, rating_edge_from_train], dtype=torch.long
    ).to(device)
    rating_train.reverse()
    data_train['item', 'rated_by', 'user'].edge_attr = torch.tensor(
        rating_train, dtype=torch.float
    ).to(device)

    data_train['user', 'review', 'item'].edge_index = torch.tensor(
        [review_edge_from_train, review_edge_to_train], dtype=torch.long
    ).to(device)
    data_train['user', 'review', 'item'].edge_attr = torch.tensor(review_train, dtype=torch.float).to(device)
    data_train['item', 'reviewed_by', 'user'].edge_index = torch.tensor(
        [review_edge_to_train, review_edge_from_train], dtype=torch.long
    ).to(device)
    review_train.reverse()
    data_train['item', 'reviewed_by', 'user'].edge_attr = torch.tensor(review_train, dtype=torch.float).to(device)

    data_train['user', 'buys', 'item'].edge_index = torch.tensor(
        [verify_buy_from_train, verify_buy_to_train]
    ).to(device)
    data_train['item', 'bought_by', 'user'].edge_idex = torch.tensor(
        [verify_buy_to_train, verify_buy_from_train]
    ).to(device)
    # item_random_walk_train = random_walk(data_train['item', 'rated_by', 'user']['edge_index'])
    # user_random_walk_train = random_walk(data_train['user', 'rates', 'item']['edge_index'])
    # data_train['user', 'related_to', 'user'].edge_index = torch.tensor(
    #     [user_random_walk_train[0] + user_random_walk_train[1],
    #      user_random_walk_train[1] + user_random_walk_train[0]]).to(device)
    # data_train['item', 'related_to', 'item'].edge_index = torch.tensor(
    #     [item_random_walk_train[0] + item_random_walk_train[1],
    #      item_random_walk_train[1] + item_random_walk_train[0]]).to(device)
    # build bidirectional edges for items within same store
    data_train['item', 'same_store', 'item'].edge_index = torch.tensor(
        [same_store_edge_train[0] + same_store_edge_train[1], same_store_edge_train[1] + same_store_edge_train[0]]
    ).to(device)

    print('train edge data finished')

    # region testing edges
    rating_edge_from_test, rating_edge_to_test = [], []
    rating_test = []
    verify_buy_from_test, verify_buy_to_test = [], []
    review_test = []
    review_edge_from_test, review_edge_to_test = [], []
    counter = 0
    store_item_dict_test = {key: [] for key in stores}
    same_store_edge_test = [[], []]
    for index, row in testing_data.iterrows():
        if counter % print_counter == 0:
            print(str(round(counter / rate_count_test, 2) * 100) + '%')

        rating_edge_from_test.append(uid_test[row['user_id']])
        rating_edge_to_test.append(iid_test[row['parent_asin']])
        rating_test.append(row['rating'])
        store_item_dict_test[items_dict[row['parent_asin']]['store']].append(iid_test[row['parent_asin']])

        if row['text'] is not None:
            review_edge_from_test.append(uid_test[row['user_id']])
            review_edge_to_test.append(iid_test[row['parent_asin']])
            review_test.append(torch.nn.Embedding(1, 768).weight.detach().cpu().numpy())

        if row['verified_purchase']:
            verify_buy_from_test.append(uid_test[row['user_id']])
            verify_buy_to_test.append(iid_test[row['parent_asin']])

        counter += 1

    for store in store_item_dict_test.keys():
        item_from_store = list(set(store_item_dict_test[store]))
        if len(item_from_store) < 2:
            pass
        for i in range(len(item_from_store)):
            for j in range(i, len(item_from_store)):
                same_store_edge_test[0].append(item_from_store[i])
                same_store_edge_test[1].append(item_from_store[j])

    # Convert List of NumPy Arrays to a Single NumPy Array
    review_test = np.array(review_test).tolist()

    # Adding edges and edge attributes
    data_test['user', 'rates', 'item'].edge_index = torch.tensor(
        [rating_edge_from_test, rating_edge_to_test], dtype=torch.long
    ).to(device)
    data_test['user', 'rates', 'item'].edge_attr = torch.tensor(rating_test, dtype=torch.float).to(device)
    data_test['item', 'rated_by', 'user'].edge_index = torch.tensor(
        [rating_edge_to_test, rating_edge_from_test], dtype=torch.long
    ).to(device)
    rating_test.reverse()
    data_test['item', 'rated_by', 'user'].edge_attr = torch.tensor(
        rating_test, dtype=torch.float
    ).to(device)

    data_test['user', 'review', 'item'].edge_index = torch.tensor(
        [review_edge_from_test, review_edge_to_test], dtype=torch.long
    ).to(device).to(torch.int64)
    data_test['user', 'review', 'item'].edge_attr = torch.tensor(review_test, dtype=torch.float).to(device)
    data_test['item', 'reviewed_by', 'user'].edge_index = torch.tensor(
        [review_edge_to_test, review_edge_from_test], dtype=torch.long
    ).to(device).to(torch.int64)
    review_test.reverse()
    data_test['item', 'reviewed_by', 'user'].edge_attr = torch.tensor(review_test, dtype=torch.float).to(device)

    data_test['user', 'buys', 'item'].edge_index = torch.tensor(
        [verify_buy_from_test, verify_buy_to_test]
    ).to(device).to(torch.int64)
    data_test['item', 'bought_by', 'user'].edge_idex = torch.tensor(
        [verify_buy_to_test, verify_buy_from_test]
    ).to(device).to(torch.int64)
    # item_random_walk_test = random_walk(data_test['item', 'rated_by', 'user']['edge_index'])
    # user_random_walk_test = random_walk(data_test['user', 'rates', 'item']['edge_index'])
    # data_test['user', 'related_to', 'user'].edge_index = torch.tensor(
    #     [user_random_walk_test[0] + user_random_walk_test[1], user_random_walk_test[1] + user_random_walk_test[0]]
    # ).to(device).to(torch.int64)
    # data_test['item', 'related_to', 'item'].edge_index = torch.tensor(
    #     [item_random_walk_test[0] + item_random_walk_test[1], item_random_walk_test[1] + item_random_walk_test[0]]
    # ).to(device).to(torch.int64)
    data_test['item', 'same_store', 'item'].edge_index = torch.tensor(
        [same_store_edge_test[0] + same_store_edge_test[1], same_store_edge_test[1] + same_store_edge_test[0]]
    ).to(device).to(torch.int64)

    print('test edge data finished')

    # Building Heterogeneous graph
    class HeteroGNN(torch.nn.Module):
        def __init__(self, num_users, hidden_channels, item_features_dim):
            super(HeteroGNN, self).__init__()
            self.user_embedding = torch.nn.Embedding(num_users, item_features_dim)

            # HeteroConv for word2vec
            self.conv1 = HeteroConv({
                ('user', 'rates', 'item'): SAGEConv((item_features_dim, item_features_dim), hidden_channels),
                ('item', 'rated_by', 'user'): SAGEConv((item_features_dim, item_features_dim), hidden_channels),
                ('user', 'buys', 'item'): SAGEConv((item_features_dim, item_features_dim), hidden_channels),
                ('item', 'bought_by', 'user'): SAGEConv((item_features_dim, item_features_dim), hidden_channels),
                ('user', 'review', 'item'): SAGEConv((item_features_dim, item_features_dim), hidden_channels),
                ('item', 'reviewed_by', 'user'): SAGEConv((item_features_dim, item_features_dim), hidden_channels),
                ('item', 'related_to', 'item'): SAGEConv((item_features_dim, item_features_dim), hidden_channels),
                ('user', 'related_to', 'user'): SAGEConv((item_features_dim, item_features_dim), hidden_channels),
                ('item', 'same_store', 'item'): SAGEConv((item_features_dim, item_features_dim), hidden_channels)
            }, aggr='sum')

            self.lin = Linear(hidden_channels, 1)

        def forward(self, x_dict, edge_index_dict):
            # Assuming edge_index_dict is correctly formed and passed
            x_dict['user'] = self.user_embedding(x_dict['user'])  # Embed user features
            x_dict = self.conv1(x_dict, edge_index_dict)  # First layer of convolutions
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}  # Apply non-linearity
            return x_dict

    # Assuming data_train and data_test are defined properly with .x, .edge_index, etc.
    num_users_train = data_train['user'].num_nodes
    num_users_test = data_test['user'].num_nodes
    item_features_dim = data_train['item'].x.size(1)
    print(item_features_dim)
    hidden_channels = 16

    # Instantiate the model
    model = HeteroGNN(num_users_train, hidden_channels, item_features_dim).to(device)

    # Training process
    learning_rate = 0.001
    num_epochs = 900
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out_dict = model(
            {
                'user': torch.arange(num_users_train).to(device),
                'item': data_train['item'].x.to(device)
            },
            data_train.edge_index_dict
        )
        user_out = out_dict['user'].to(device)
        user_indices = data_train['user', 'rates', 'item'].edge_index[0]
        predicted_ratings = model.lin(user_out[user_indices]).squeeze()
        loss = criterion(predicted_ratings, data_train['user', 'rates', 'item'].edge_attr.squeeze())
        loss.backward()
        optimizer.step()
        if loss.item() < 0.05:
            break
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        out_dict = model(
            {
                'user': torch.arange(num_users_test).to(device),
                'item': data_test['item'].x.to(device)
            },
            data_test.edge_index_dict
        )
        user_out = out_dict['user']
        user_indices = data_test['user', 'rates', 'item'].edge_index[0]
        predicted_ratings = model.lin(user_out[user_indices]).squeeze().tolist()

    print(calculate_RMSE(predicted_ratings, testing_data['rating'].tolist()))
    print(calculate_MAE(predicted_ratings, testing_data['rating'].tolist()))

    return predicted_ratings


def calculate_RMSE(predicted_result, true_label):
    if len(predicted_result) != len(true_label):
        return 0

    total_error = 0
    # individual_diff = []
    length = len(predicted_result)
    i = 0

    while i < length:
        diff = predicted_result[i] - true_label[i]
        # individual_diff.append(diff)
        total_error += (diff * diff)
        i += 1

    return np.sqrt(total_error / length)


def calculate_MAE(predicted_result, true_label):
    if len(predicted_result) != len(true_label):
        return 0

    total_error = 0
    # individual_diff = []
    length = len(predicted_result)
    i = 0

    while i < length:
        diff = predicted_result[i] - true_label[i]
        # individual_diff.append(abs(diff))
        total_error += abs(diff)
        i += 1

    return np.sqrt(total_error / length)

def convert_rating(rating, threshold=3):
    if rating > threshold:
        return 1
    if rating < threshold:
        return -1
    return 0


# def random_walk(item_user_edge):
#     new_edge = [[], []]
#     item = item_user_edge[0]
#     user = item_user_edge[1]
#     for i in range(len(item_user_edge[0])):
#         # if i % 10000 == 0:
#         #     print(i / len(item))
#         start = item[i]
#         neighbours = user[item == start]
#         random_neighbour = random.choice(neighbours)
#         final_items = item[user == random_neighbour]
#         final_items = final_items[final_items != start]
#         if (len(final_items) > 0):
#             new_edge[0].append(start.tolist())
#             new_edge[1].append(random.choice(final_items).tolist())
#
#     return new_edge


if __name__ == '__main__':
    beauty_path = '../../Datasets/beauty.pkl'
    fashion_path = '../../Datasets/fashion.pkl'
    music_path = '../../Datasets/Musical_Instruments.pkl'
    movie_path = '../../Datasets/Movies_and_TV.pkl'
    toys_path = '../../Datasets/Toys_and_Games.pkl'

    meta_beauty_path = '../../Datasets/meta_beauty.pkl'
    meta_fashion_path = '../../Datasets/meta_fashion.pkl'
    meta_music_path = '../../Datasets/meta_Musical_Instruments.pkl'
    meta_movie_path = '../../Datasets/meta_Movies_and_TV.pkl'
    meta_toys_path = '../../Datasets/meta_Toys_and_Games.pkl'

    num_chunks = 5
    num_folds = num_chunks
    PERCENTAGE_SIZE = 1
    BATCH_SIZE = 250
    device = 'cuda:2'
    torch.cuda.manual_seed_all(42)  # If you're using GPU
    np.random.seed(42)
    hidden_channels = 16
    df_path_list = [beauty_path, fashion_path, music_path, movie_path, toys_path]
    meta_df_path_list = [meta_beauty_path, meta_fashion_path, meta_music_path, meta_movie_path, meta_toys_path]

    for df_path, meta_df_path in zip(df_path_list, meta_df_path_list):
        RMSE_list = []
        MAE_list = []

        df = load_dataset(df_path)
        meta_df = load_dataset(meta_df_path)

        if 'beauty' in df_path:
            identifier = "All_Beauty"
        elif 'fashion' in df_path:
            identifier = "Amazon_Fashion"
        elif 'Movies_and_TV' in df_path:
            identifier = "Movies_and_TV"
        elif 'Musical_Instruments' in df_path:
            identifier = "Musical_Instruments"
        elif 'Toys_and_Games' in df_path:
            identifier = "Toys_and_Games"

        # region pre-process
        # remove nan value from rating column
        df.dropna(subset=["rating"], inplace=True)

        '''
        this part is to remove empty title from interactions, 
        and remove empty title from item_attributes
        '''
        item_with_empty_title = meta_df[meta_df['title'].str.strip() == '']['parent_asin'].tolist()
        meta_df = meta_df[meta_df['title'].str.strip() != '']
        df = df[~df['parent_asin'].isin(item_with_empty_title)]

        '''
        this part is to remove nan value in the store column from interactions,
        and remove nan value in the store column from item_attributes
        '''
        meta_df['store'].replace({None: np.nan})
        removed_parent_asin = meta_df.loc[meta_df['store'].isna(), 'parent_asin']
        df = df[~df['parent_asin'].isin(removed_parent_asin)]
        meta_df.dropna(subset=['store'], inplace=True)

        meta_df = meta_df.reset_index(drop=True)
        df = df.reset_index(drop=True)
        df = df[0: 500000]

        if identifier == "Toys_and_Games":
            df = df[: 300000]
        # endregion

        # region get train, test dataset ready for word2vec
        shuffled_data = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split the data into 5(num_chunks) equal parts
        chunks = np.array_split(shuffled_data, num_chunks)
        # endregion

        items_dict = {}
        item_count = len(meta_df)
        counter = 0
        print_counter = 20000
        rate_count_train = len(meta_df)

        for index, row in meta_df.iterrows():
            if counter % print_counter == 0:
                print(str(round(counter / rate_count_train, 2) * 100) + '%')

            # 50 comes from A_llmrec hidden units
            items_dict[row['parent_asin']] = {
                "average_rating": row['average_rating'],
                "rating_number": row['rating_number'],
                "title": torch.nn.Embedding(1, 50),
                "store": row['store']
            }

            counter += 1

        stores = list(set(meta_df['store'].tolist()))

        del df, meta_df, item_with_empty_title, removed_parent_asin

        mae_list = []
        rmse_list = []
        i = 0
        # the train-evaluate process uses 5-Fold Cross-Validation
        while i < num_chunks:
            # Dynamically concatenate the chunks for training, excluding the one for validation
            train_chunks = []
            for j in range(num_chunks - 1):  # Select (num_folds - 1) chunks for training
                train_chunks.append(chunks[(i + j) % num_chunks])

            # Concatenate all the selected chunks for training
            result = train_and_evaluate(
                pd.concat(train_chunks),
                chunks[(i + num_chunks - 1) % num_chunks],  # Validation chunk
                items_dict
            )

            # Calculate RMSE and MAE for the validation chunk
            rmse = calculate_RMSE(result, chunks[(i + num_chunks - 1) % num_chunks]['rating'].tolist())
            mae = calculate_MAE(result, chunks[(i + num_chunks - 1) % num_chunks]['rating'].tolist())

            mae_list.append(mae)
            rmse_list.append(rmse)

            # Increment the loop counter
            i += 1

            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()

        print(
            'Dataset:', df_path,
            'RMSE:', sum(rmse_list) / len(rmse_list),
            "MAE:", sum(mae_list) / len(mae_list),
            "Hidden channels:", hidden_channels,
        )

        RMSE_list.append(round(sum(rmse_list) / len(rmse_list), 4))
        MAE_list.append(round(sum(mae_list) / len(mae_list), 4))

        print(rmse_list)
        print(mae_list)

        with open('mae.pkl', 'wb') as f:
            pickle.dump(mae_list, f)
        with open('rmse.pkl', 'wb') as f:
            pickle.dump(rmse_list, f)

        gc.collect()
        torch.cuda.empty_cache()

    temp = [
        ['RMSE'] + RMSE_list,
        ['MAE'] + MAE_list
    ]

    # Create DataFrame
    df = pd.DataFrame(temp)

    if 'beauty' in df_path:
        output_path = f'../../Datasets/Item_emb+HeteroGNN_hidden_channels={hidden_channels}_{identifier}.xlsx'
    else:
        output_path = f'../../Datasets/Item_emb+HeteroGNN_hidden_channels={hidden_channels}_{identifier}.xlsx'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df.to_excel(
        output_path,
        index=False, header=False
    )

'''
Dataset: ../../Datasets/beauty.pkl RMSE: 1.9263550178951043 MAE: 1.2352956496364105 Hidden channels: 16
[1.9180948528632953, 1.9242486808296126, 1.8940457957512085, 1.946122520530208, 1.9492632395011973]
[1.232875406533168, 1.2321887650878611, 1.22191764740238, 1.2435956132432748, 1.2459008159153684]

Dataset: ../../Datasets/fashion.pkl RMSE: 1.7889301206522579 MAE: 1.190354181355555 Hidden channels: 16
[1.809962036717198, 1.8289870679866695, 1.7583567945733083, 1.747928421618566, 1.7994162823655475]
[1.1959740937093768, 1.203758846700967, 1.1810364706239067, 1.1771488107213892, 1.1938526850221345]

Dataset: ../../Datasets/Musical_Instruments.pkl RMSE: 1.6136948244264055 MAE: 1.1107463260672383 Hidden channels: 16
[1.722322221463703, 1.5242161638376956, 1.60973418134841, 1.5974608169807285, 1.6147407385014911]
[1.1517962651215392, 1.0762407267840561, 1.107348919245471, 1.1059325168894967, 1.112413202295628]

Dataset: ../../Datasets/Movies_and_TV.pkl RMSE: 1.3792472008009433 MAE: 1.0114545239320847 Hidden channels: 16
[1.3000572743367862, 1.431774508752186, 1.3815417530045855, 1.4307614028520883, 1.3521010650590717]
[0.975643876142402, 1.0355084880350123, 1.0134108999030302, 1.0340887808345034, 0.9986205747454752

Dataset: ../../Datasets/Toys_and_Games.pkl RMSE: 1.467610003291657 MAE: 1.0513295344872002 Hidden channels: 16
[1.486180812944024, 1.4423096448912782, 1.4140931151669918, 1.4626627431019055, 1.5328037003540862]
[1.0630176861377028, 1.0403269184807769, 1.0259483428851335, 1.0462721285512746, 1.0810825963811133]
'''