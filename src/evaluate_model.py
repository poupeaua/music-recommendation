import os
import pandas as pd
import numpy as np
import h5py
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
import progress
from progress.bar import Bar


# absolute path to music-recommendation directory
PROJECT_ABSPATH = "/home/osboxes/Documents/python/machlrn/music-recommendation"



def dim_red(dataframe, min_variance_explained):
    """
        Reduce the dimension of features

        Arguments:
            dataframe (pandas df) : first column song_id then N1 features
            min_variance_explained (float) : in [0, 1], variance explained sum
                in order to choose the number of best components.

        Return:
            df (pandas df) : same sort of dataframe but with less features
    """
    array_features = dataframe.iloc[:, 1:].to_numpy()
    n_features = array_features.shape[1]
    pca = PCA(n_components=min_variance_explained, svd_solver='full')
    new_array_features = pca.fit_transform(array_features)
    df = pd.DataFrame(data=new_array_features)
    df.insert(loc=0, value=dataframe.iloc[:, 0], column=dataframe.columns.values[0])
    return df



def calc_average_precision_score(y_pred, y_true, method):
    """
        Arguments:
            y_pred (array) : song id predicted as good recommendation by
                distance (similarity search).
            y_true (array) : the song id expected to be found in y_pred
            method (str) : method to calculate the metric in [basic, one_if_any]
                Basic is the real average precision calculation technique.

        Return:
            ap_score (float) : score that tell how good are the recommendation
    """
    similarity = [True if song_id in y_true else False for song_id in y_pred  ]

    if method == "basic":
        # basic average precision score metric
        tmp = 0
        ap_score = 0
        for i, val in enumerate(similarity):
            if val is True:
                tmp += 1
                ap_score += tmp/(i+1)
        ap_score /= len(similarity)
    elif method == "one_if_any":
        # other metric
        ap_score = int(np.any(similarity))
    return ap_score


def compute_avg_prec(clt_centers, df_rest_features, test_id, n, method, avg_on_cluster):
    """
        Arguments:
            clt_centers (array) : cluster centers of size nb_features
            df_rest_features (pd df) : shape (n_songs__not_in_train, nb_features+1)
                (+1 for song_id)
            test_id (array) : song id used to evaluate the avg prec.
            n (int) : number of closest songs chosen to compute each avg prec
            method (str) : method to calculate the metric in [basic, one_if_any]
                Basic is the real average precision calculation technique.
            avg_on_cluster (bool) : whether to average the metric based on every cluster
                or take the best value (closest to 1).

        Return:
            average_precision (float) : value in [0, 1]
    """
    # get some information
    n_clusters = clt_centers.shape[0]
    array_features = df_rest_features.iloc[:, 1:].to_numpy()
    dists = pairwise_distances(X=array_features, Y=clt_centers)
    df = pd.DataFrame(data=dists)
    song_id_column_name = df_rest_features.columns.values[0]
    df.insert(loc=0, value=df_rest_features.iloc[:, 0], column=song_id_column_name)

    # AP for each cluster
    ap_total = np.zeros(shape=n_clusters)
    for i in range(n_clusters):
        df_tmp = df[[song_id_column_name, i]]
        df_sorted = df_tmp.sort_values(by=i)
        first_id = df_sorted.get_song_id_decoded.values[:n]
        ap_total[i] =  calc_average_precision_score(y_true=test_id,
                                                    y_pred=first_id,
                                                    method=method)

    if avg_on_cluster:
        # taking the "best" value from all the clusters
        ap_final = np.mean(ap_total)
    else:
        # average value on all clusters
        ap_final = np.max(ap_total)

    return ap_final



def evaluate_model(m, k, t, n,
                   dim_reduction=True,
                   min_variance_explained=0.999,
                   method="basic",
                   avg_on_cluster=False):
    """
        Evaluate the model of music recommendation calculating the
        mean average precision.

        Arguments:
            m (int) : minimum number of users
            k (int) : number of clusters
            t (int) : number of songs in each test set
                - constraint t < m/2
            n (int) : number of closest songs chosen to compute each avg prec
            dim_reduction (bool) : reduce feature dimension or not
            min_variance_explained (float) : in [0, 1], variance explained sum
                in order to choose the number of best components.
            method (str) : method to calculate the metric in [basic, one_if_any]
                Basic is the real average precision calculation technique.
            avg_on_cluster (bool) : whether to average the metric based on every cluster
                or take the best value (closest to 1).

        Return:
            mean_average_precision (float) : in [0, 1] and evaluate how good
                the model of music recommendation is
    """
    if t >= m/2:
        raise RuntimeError("t >= m/2")

    # get paths
    user_df_path = os.path.join(PROJECT_ABSPATH, "data/UserTasteFiltered/min_users_" + str(m) +"_taste_triplets.txt")
    song_features_df_path = os.path.join(PROJECT_ABSPATH, "data/SongsFeatures/df_cleaned_features.csv")

    # access data
    df_users = pd.read_csv(filepath_or_buffer=user_df_path)
    df_song_features = pd.read_csv(filepath_or_buffer=song_features_df_path)
    df_song_features.dropna(inplace=True)

    # if dimension reduction
    if dim_reduction:
        print("Processing dimension reduction...")
        nb_features_before = len(df_song_features.columns)-1
        df_song_features = dim_red(dataframe=df_song_features,
                                   min_variance_explained=min_variance_explained)
        print("Number of features reduced from", nb_features_before, "to", len(df_song_features.columns)-1)

    # useful data information
    unique_user_id = df_users.user_id.unique()
    nb_users = len(unique_user_id)

    # calculating mAP
    stock = np.zeros(shape=nb_users)
    for i, user_id in Bar('Calculating mAP', max=nb_users).iter(enumerate(unique_user_id)):
        # extract all the information needed for the current user
        cur_user_taste = df_users[df_users.user_id == user_id]
        cur_user_song_id = cur_user_taste.song_id.values
        test_id, train_id = cur_user_song_id[:t], cur_user_song_id[t:]
        df_rest_features, df_train_features = df_song_features[~df_song_features.get_song_id_decoded.isin(train_id)], df_song_features[df_song_features.get_song_id_decoded.isin(train_id)]
        array_train_features = df_train_features.iloc[:, 1:].to_numpy()
        # create model
        clt = KMeans(n_clusters=k, random_state=0, n_jobs=-1)
        clt.fit(X=array_train_features)
        # calculating avg precision
        average_precision = compute_avg_prec(clt_centers=clt.cluster_centers_,
                                             df_rest_features=df_rest_features,
                                             test_id=test_id,
                                             n=n,
                                             method=method,
                                             avg_on_cluster=avg_on_cluster)
        stock[i] = average_precision
    mean_average_precision = np.mean(stock)
    return mean_average_precision


if __name__ == "__main__":
    # evaluate the model
    # m (min_nb_songs_per_users) [5, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    m = 20
    t = int(m/2)-1
    mean_average_precision = evaluate_model(m=m,
                                            k=6,
                                            n=25,
                                            t=t,
                                            dim_reduction=True,
                                            min_variance_explained=0.999,
                                            method="one_if_any",
                                            avg_on_cluster=False)
    print("mAP = " + str(round(mean_average_precision*100, 2)) + "%")
