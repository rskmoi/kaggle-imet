#! /usr/bin/python3
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import pandas as pd

def make_label(dense_label):
    label = np.zeros((1103,), dtype=np.uint8)
    dense_label_split = dense_label.split(" ")
    for cur_label in dense_label_split:
        label[int(cur_label)] = 1.

    assert label.shape == (1103,)
    return label


def stratified_split():
    print("Splitting train data...")
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=92)

    train_df = pd.read_csv("../input/train.csv")
    train_df_orig = train_df.copy()
    X = train_df["id"].tolist()
    y = train_df['attribute_ids'].tolist()
    y = [make_label(cur_y) for cur_y in y]
    for train_index, test_index in msss.split(X, y):
        new_train_df = train_df_orig.loc[train_df_orig.index.intersection(train_index)].copy()
        new_valid_df = train_df_orig.loc[train_df_orig.index.intersection(test_index)].copy()
        new_train_df.to_csv("./data/train_split_90pc.csv", index=False)
        new_valid_df.to_csv("./data/valid_split_10pc.csv", index=False)
    print("Successfully finished!")

if __name__ == '__main__':
    stratified_split()
