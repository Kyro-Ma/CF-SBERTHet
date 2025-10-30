import gc
import sys
from collections import defaultdict
from models.a_llmrec_model import A_llmrec_model
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
import nltk
import argparse
from tqdm import tqdm
import os

# Set the CUDA_LAUNCH_BLOCKING environment variable to 1
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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
        # if counter % print_counter == 0:
        #     print(str(round(counter / rate_count_train, 2) * 100) + '%')

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
    print("standardise user id and item id for testing train_edge_data")
    for index, row in testing_data.iterrows():
        if counter % print_counter == 0:
            print(str(round(counter / rate_count_test, 2) * 100) + '%')

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
        # temp = [target['average_rating'], target['rating_number']] + target['embed'].tolist()
        temp = target['embed'].tolist()
        item_features_train.append(temp)
        counter += 1

    counter = 0
    print("Getting item features (testing)")
    for value in iid_test.keys():
        if counter % print_counter == 0:
            print(str(round(counter / len(iid_test.keys()), 2) * 100) + '%')

        target = items_dict[value]
        # temp = [target['average_rating'], target['rating_number']] + target['embed'].tolist()
        temp = target['embed'].tolist()
        item_features_test.append(temp)
        counter += 1

    # Adding item nodes with features
    data_train['item'].x = torch.tensor(item_features_train, dtype=torch.float).to(device)  # Item features (2D)
    data_test['item'].x = torch.tensor(item_features_test, dtype=torch.float).to(device)  # Item features (2D)

    # region training edges
    rating_edge_from_train, rating_edge_to_train = [], []
    rating_train = []
    verify_buy_from_train, verify_buy_to_train = [], []
    review_list_train = []
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
            review_list_train.append('Title: ' + row['title'] + 'Text: ' + row['text'])
            # review_train.append(get_word2vec_sentence_vector(row['title'] + row['text'], w2v_model, vector_size))

        if row['verified_purchase']:
            verify_buy_from_train.append(uid_train[row['user_id']])
            verify_buy_to_train.append(iid_train[row['parent_asin']])

        if len(review_list_train) == 64:
            embed_of_review = llmrec_model.generate_text_emb(review_list_train)

            for i in range(len(embed_of_review)):
                review_train.append(embed_of_review[i])

            review_list_train = []

        counter += 1

    # can't use the method below on 4080, cause memory isn't enough
    if len(review_list_train) > 0:
        embed_of_review = llmrec_model.generate_text_emb(review_list_train)
    # print('length of embed of review', len(embed_of_review))
    #
    for i in range(len(embed_of_review)):
        review_train.append(embed_of_review[i])

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
    # review_train = review_train.cpu().numpy().tolist()
    review_train = [t.cpu().numpy().tolist() for t in review_train]

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
    #      user_random_walk_train[1] + user_random_walk_train[0]]
    # ).to(device)
    # data_train['item', 'related_to', 'item'].edge_index = torch.tensor(
    #     [item_random_walk_train[0] + item_random_walk_train[1],
    #      item_random_walk_train[1] + item_random_walk_train[0]]
    # ).to(device)
    # build bidirectional edges for items within same store
    data_train['item', 'same_store', 'item'].edge_index = torch.tensor(
        [same_store_edge_train[0] + same_store_edge_train[1], same_store_edge_train[1] + same_store_edge_train[0]]
    ).to(device)

    print('train edge data finished')

    # region testing edges
    review_list_test = []
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
            review_list_test.append('Title: ' + row['title'] + 'Text: ' + row['text'])
            # review_test.append(get_word2vec_sentence_vector(row['title'] + row['text'], w2v_model, vector_size))

        if row['verified_purchase']:
            verify_buy_from_test.append(uid_test[row['user_id']])
            verify_buy_to_test.append(iid_test[row['parent_asin']])

        if len(review_list_test) == 64:
            embed_of_review = llmrec_model.generate_text_emb(review_list_test)

            for i in range(len(embed_of_review)):
                review_test.append(embed_of_review[i])

            review_list_test = []

        counter += 1

    # can't use the method below on 4080, cause memory isn't enough
    if len(review_list_test) > 0:
        embed_of_review = llmrec_model.generate_text_emb(review_list_test)

    for i in range(len(embed_of_review)):
        review_test.append(embed_of_review[i])

    for store in store_item_dict_test.keys():
        item_from_store = list(set(store_item_dict_test[store]))
        if len(item_from_store) < 2:
            pass
        for i in range(len(item_from_store)):
            for j in range(i, len(item_from_store)):
                same_store_edge_test[0].append(item_from_store[i])
                same_store_edge_test[1].append(item_from_store[j])

    # Convert List of NumPy Arrays to a Single NumPy Array
    # review_test = review_test.cpu().numpy().tolist()
    review_test = [t.cpu().numpy().tolist() for t in review_test]

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
            self.conv2 = HeteroConv({
                ('user', 'rates', 'item'): SAGEConv(hidden_channels, hidden_channels),
                ('item', 'rated_by', 'user'): SAGEConv(hidden_channels, hidden_channels),
                ('user', 'buys', 'item'): SAGEConv(hidden_channels, hidden_channels),
                ('item', 'bought_by', 'user'): SAGEConv(hidden_channels, hidden_channels),
                ('user', 'review', 'item'): SAGEConv(hidden_channels, hidden_channels),
                ('item', 'reviewed_by', 'user'): SAGEConv(hidden_channels, hidden_channels),
                ('item', 'same_store', 'item'): SAGEConv(hidden_channels, hidden_channels)
            }, aggr='sum')
            self.conv3 = HeteroConv({
                ('user', 'rates', 'item'): SAGEConv(hidden_channels, hidden_channels),
                ('item', 'rated_by', 'user'): SAGEConv(hidden_channels, hidden_channels),
                ('user', 'buys', 'item'): SAGEConv(hidden_channels, hidden_channels),
                ('item', 'bought_by', 'user'): SAGEConv(hidden_channels, hidden_channels),
                ('user', 'review', 'item'): SAGEConv(hidden_channels, hidden_channels),
                ('item', 'reviewed_by', 'user'): SAGEConv(hidden_channels, hidden_channels),
                ('item', 'same_store', 'item'): SAGEConv(hidden_channels, hidden_channels)
            }, aggr='sum')

            self.lin = Linear(hidden_channels, 1)

        def forward(self, x_dict, edge_index_dict):
            # Assuming edge_index_dict is correctly formed and passed
            x_dict['user'] = self.user_embedding(x_dict['user'])  # Embed user features
            x_dict = self.conv1(x_dict, edge_index_dict)  # First layer of convolutions
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}  # Apply non-linearity
            x_dict = self.conv2(x_dict, edge_index_dict)  # Second layer of convolutions
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}  # Apply non-linearity
            x_dict = self.conv3(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}  # Apply non-linearity
            return x_dict

    # Assuming data_train and data_test are defined properly with .x, .edge_index, etc.
    num_users_train = data_train['user'].num_nodes
    num_users_test = data_test['user'].num_nodes
    item_features_dim = data_train['item'].x.size(1)

    # Instantiate the model
    model = HeteroGNN(num_users_train, hidden_channels, item_features_dim).to(device)

    # Training process
    learning_rate = 0.001
    num_epochs = 250
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    # Training loop
    last_epoch_loss = 0
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

        if loss.item() < 0.05 or last_epoch_loss == loss.item():
            break

        last_epoch_loss = loss.item()
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


def get_text_embedding(item_id, item_embed_dict):
    # if item_id not in valid_item_ids:
    #     return None

    return item_embed_dict[item_id]


def convert_rating(rating, threshold=3):
    if rating > threshold:
        return 1
    if rating < threshold:
        return -1
    return 0


def get_word2vec_sentence_vector(sentence, model, vector_size):
    words = word_tokenize(sentence)
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) == 0:  # To handle cases where no words are in the model
        return np.zeros(vector_size)
    sentence_vector = np.mean(word_vectors, axis=0)
    return sentence_vector


def random_walk(item_user_edge):
    new_edge = [[], []]
    item = item_user_edge[0]
    user = item_user_edge[1]
    for i in range(len(item_user_edge[0])):
        # if i % 10000 == 0:
        #     print(i / len(item))
        start = item[i]
        neighbours = user[item == start]
        random_neighbour = random.choice(neighbours)
        final_items = item[user == random_neighbour]
        final_items = final_items[final_items != start]
        if (len(final_items) > 0):
            new_edge[0].append(start.tolist())
            new_edge[1].append(random.choice(final_items).tolist())

    return new_edge


if __name__ == '__main__':
    # region args
    parser = argparse.ArgumentParser()

    # GPU train options
    parser.add_argument("--multi_gpu", action='store_true', default=0)
    parser.add_argument('--gpu_num', type=int, default=1)

    # model setting
    parser.add_argument("--llm", type=str, default='llama', help='flan_t5, opt, vicuna')
    parser.add_argument("--recsys", type=str, default='sasrec')

    # dataset setting
    parser.add_argument("--rec_pre_trained_data", type=str, default='All_Beauty')

    # train phase setting
    parser.add_argument("--pretrain_stage1", action='store_true')
    parser.add_argument("--pretrain_stage2", action='store_true')
    '''
    action='store_true': This means that when the --inference argument is present in the command line, 
    the value of args.inference will be set to True. If --inference is not present, args.inference will 
    be set to False.
    '''
    parser.add_argument("--inference", action='store_true')

    # hyperparameters options
    parser.add_argument('--batch_size1', default=32, type=int)
    parser.add_argument('--batch_size2', default=2, type=int)
    parser.add_argument('--batch_size_infer', default=2, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument("--stage1_lr", type=float, default=0.0001)
    parser.add_argument("--stage2_lr", type=float, default=0.0001)

    args = parser.parse_args()
    # endregion

    beauty_path = 'Datasets/beauty.pkl'
    fashion_path = 'Datasets/fashion.pkl'
    movie_path = 'Datasets/Movies_and_TV.pkl'
    music_path = 'Datasets/Musical_Instruments.pkl'
    meta_beauty_path = 'Datasets/meta_beauty.pkl'
    meta_fashion_path = 'Datasets/meta_fashion.pkl'
    meta_movie_path = 'Datasets/meta_Movies_and_TV.pkl'
    meta_music_path = 'Datasets/meta_Musical_Instruments.pkl'

    threshold_for_fashion = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    threshold_for_beauty = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    threshold_for_movie = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    threshold_for_music = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    num_chunks = 5
    gnn_layers = 3
    num_folds = num_chunks
    PERCENTAGE_SIZE = 1
    BATCH_SIZE = 250
    hidden_channels = 16
    device = 'cuda:4'
    torch.cuda.manual_seed_all(42)  # If you're using GPU
    np.random.seed(42)
    args.device = device
    args.pretrain_stage2 = 1  # !!! this is important, don't change it or can't use its item_emb_proj
    args.pretrain_stage1 = 1  # !!! this is important, don't change it or can't use its self.sbert

    df_path_list = [beauty_path, fashion_path, music_path, movie_path]
    meta_df_path_list = [meta_beauty_path, meta_fashion_path, meta_music_path, meta_movie_path]
    threshold_list = [threshold_for_beauty, threshold_for_fashion, threshold_for_music, threshold_for_movie]

    count = 0

    for df_path, meta_df_path, threshold in zip(df_path_list, meta_df_path_list, threshold_list):
        # if count == 0:
        #     count += 1
        #     continue

        RMSE_list = []
        MAE_list = []

        if 'beauty' in df_path:
            args.rec_pre_trained_data = "All_Beauty"
        elif 'fashion' in df_path:
            args.rec_pre_trained_data = "Amazon_Fashion"
        elif 'Movies_and_TV' in df_path:
            args.rec_pre_trained_data = "Movies_and_TV"
        elif 'Musical_Instruments' in df_path:
            args.rec_pre_trained_data = "Musical_Instruments"
        elif 'Toys_and_Games' in df_path:
            args.rec_pre_trained_data = "Toys_and_Games"

        llmrec_model = A_llmrec_model(args).to(args.device)

        phase1_epoch = 10  # default is 10, its meaning is the epoch num used to train phase1 of A_LLMRec
        llmrec_model.load_model(args, phase1_epoch=phase1_epoch)

        for local_threshold in threshold:
            df = load_dataset(df_path)
            meta_df = load_dataset(meta_df_path)
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

            # region pre-process
            countU = defaultdict(int)
            countP = defaultdict(int)
            df_rows_to_drop = []
            meta_df_rows_to_drop = []
            itemmap = {}
            itemnum = 0
            usernum = 0

            countU_path = 'countU_' + args.rec_pre_trained_data + '_' + '.pkl'
            countP_path = 'countP_' + args.rec_pre_trained_data + '_' + '.pkl'
            if os.path.exists(countU_path) and os.path.exists(countP_path):
                countU = load_dataset(countU_path)
                countP = load_dataset(countP_path)
            else:
                for _, l in df.iterrows():
                    asin = l['parent_asin']
                    user = l['user_id']
                    # time = l['timestamp']
                    countU[user] += 1
                    countP[asin] += 1

                with open(countU_path, 'wb') as f:
                    pickle.dump(countU, f)
                with open(countP_path, 'wb') as f:
                    pickle.dump(countP, f)

            df_rows_to_drop_path = 'df_rows_to_drop_' + args.rec_pre_trained_data + '_' + str(local_threshold) + '.pkl'
            itemmap_path = 'itemmap_' + args.rec_pre_trained_data + '_' + str(local_threshold) + '.pkl'

            if os.path.exists(df_rows_to_drop_path) and os.path.exists(itemmap_path):
                df_rows_to_drop = load_dataset(df_rows_to_drop_path)
                itemmap = load_dataset(itemmap_path)
            else:
                for index, l in tqdm(df.iterrows(), desc='df preprocessing'):
                    asin = l['parent_asin']
                    user = l['user_id']
                    time = l['timestamp']

                    if countU[user] < local_threshold or countP[asin] < local_threshold:
                        df_rows_to_drop.append(index)
                        continue

                    # if user in usermap:
                    #     userid = usermap[user]
                    # else:
                    #     userid = usernum
                    #     usermap[user] = userid
                    #     User[userid] = []

                    if asin in itemmap:
                        # itemid = itemmap[asin]
                        pass
                    else:
                        itemnum += 1
                        itemid = itemnum
                        itemmap[asin] = itemid

                with open(itemmap_path, 'wb') as f:
                    pickle.dump(itemmap, f)
                with open(df_rows_to_drop_path, 'wb') as f:
                    pickle.dump(df_rows_to_drop, f)

            # drop the rows
            df.drop(df_rows_to_drop, inplace=True)
            df.reset_index(drop=True, inplace=True)

            valid_item_ids = list(itemmap.keys())
            for index, l in tqdm(meta_df.iterrows(), desc='meta df preprocessing'):
                if l['parent_asin'] not in valid_item_ids:
                    meta_df_rows_to_drop.append(index)

            meta_df.drop(meta_df_rows_to_drop, inplace=True)
            meta_df.reset_index(drop=True, inplace=True)

            # countU = df['user_id'].value_counts().to_dict()
            # countP = df['parent_asin'].value_counts().to_dict()
            # threshold = local_threshold
            # df_mask = df['user_id'].map(countU) >= threshold
            # df_mask &= df['parent_asin'].map(countP) >= threshold
            # df = df[df_mask].reset_index(drop=True)
            #
            # # Create itemmap using unique parent_asins from filtered df
            # valid_asins = df['parent_asin'].unique()
            # itemmap = {asin: i + 1 for i, asin in enumerate(valid_asins)}  # itemid starts from 1
            # itemnum = len(itemmap)
            #
            # # Filter meta_df in one line
            # meta_df = meta_df[meta_df['parent_asin'].isin(itemmap)].reset_index(drop=True)
            # endregion

            # item_list = list(set(df['parent_asin'].tolist() + meta_df['parent_asin'].tolist()))

            item_list = df['parent_asin'].tolist()
            itemmaps = []
            for item in item_list:
                item_map = itemmap[item]
                itemmaps.append(item_map)

            # item_embeddings = llmrec_model.item_emb_proj(llmrec_model.get_item_emb(itemmaps))[0]
            item_embeddings = llmrec_model.generate_item_emb(itemmaps)
            item_embed_dict = dict()
            for i in range(len(item_embeddings)):
                item_embed_dict[item_list[i]] = item_embeddings[i]

            if (args.rec_pre_trained_data == "Musical_Instruments" or
                args.rec_pre_trained_data == "Movies_and_TV"):
                df = df[0: 500000]

            if args.rec_pre_trained_data == "Toys_and_Games":
                df = df[0: 300000]

            # region get train, test dataset
            shuffled_data = df.sample(frac=1, random_state=42).reset_index(drop=True)
            chunks = np.array_split(shuffled_data, num_chunks)
            # endregion

            items_dict = {}
            item_count = len(meta_df)
            counter = 0
            print_counter = 20000
            rate_count_train = len(meta_df)
            valid_item_ids = list(item_embed_dict.keys())
            for index, row in meta_df.iterrows():
                if counter % print_counter == 0:
                    print(str(round(counter / rate_count_train, 2) * 100) + '%')

                items_dict[row['parent_asin']] = {
                    "average_rating": row['average_rating'],
                    "rating_number": row['rating_number'],
                    "embed": get_text_embedding(row['parent_asin'], item_embed_dict),
                    "store": row['store']
                }

                counter += 1

            stores = list(set(meta_df['store'].tolist()))

            del df, meta_df, item_with_empty_title, removed_parent_asin

            mae_list = []
            rmse_list = []

            i = 0
            while i < num_folds:
                # Dynamically concatenate the chunks for training, excluding the one for validation
                train_chunks = []
                for j in range(num_folds - 1):  # Select (num_folds - 1) chunks for training
                    train_chunks.append(chunks[(i + j) % num_folds])

                # Concatenate all the selected chunks for training
                result = train_and_evaluate(
                    pd.concat(train_chunks),
                    chunks[(i + num_folds - 1) % num_folds],  # Validation chunk
                    items_dict
                )

                # Calculate RMSE and MAE for the validation chunk
                rmse = calculate_RMSE(result, chunks[(i + num_folds - 1) % num_folds]['rating'].tolist())
                mae = calculate_MAE(result, chunks[(i + num_folds - 1) % num_folds]['rating'].tolist())

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
                'threshold:', local_threshold
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
            # count += 1

        temp = [
            ['RMSE'] + RMSE_list,
            ['MAE'] + MAE_list
        ]

        # Create DataFrame
        df = pd.DataFrame(temp)

        if 'beauty' in df_path:
            output_path = f'../../Datasets/SBERT+HeteroGNN+InteractionFilter_hidden_channels={hidden_channels}_gnn_layers={gnn_layers}_{args.rec_pre_trained_data}.xlsx'
        else:
            output_path = f'../../Datasets/SBERT+HeteroGNN+InteractionFilter_hidden_channels={hidden_channels}_gnn_layers={gnn_layers}_{args.rec_pre_trained_data}.xlsx'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_excel(
            output_path,
            index=False, header=False
        )
    '''
    sbert without random walk on beauty (RMSE:1.45, MAE: 1.03, 5-fold) threshold=4
    [1.4211206596438375, 1.4564419507837019, 1.4403185993384024, 1.4396286355165038, 1.4992591365657673]
    [1.019817613512718, 1.033346646741257, 1.0247676984620433, 1.0284196827884982, 1.041174569143495]

    sbert with random walk on beauty (RMSE:1.34, MAE: 1.00, 5-fold) threshold=2
    [1.35742859038113, 1.3790553700325663, 1.3590028477230296, 1.2783343610657774, 1.3378308190276773]
    [1.010551591900119, 1.0072547670979868, 1.0136412129049364, 0.9819472584955433, 1.0107982161338098]

    sbert with random walk on fashion (RMSE:1.49, MAE: 1.07, 5-fold) 
    [1.4867074391825088, 1.495040166246865, 1.4789086850252913, 1.4911871208532064, 1.511470367112716]
    [1.069340424426535, 1.0719301281501736, 1.06497975797665, 1.06840386004193, 1.0739476402902426] 

    sbert with random walk on fashion (RMSE:1.46, MAE: 1.06, 5-fold)
    '''
