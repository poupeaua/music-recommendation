import os
import pandas as pd
import numpy as np
import h5py
import progress
from progress.bar import Bar

# import tools functions
import h5df_getters
import h5df_spe_getters



# absolute path to music-recommendation directory
PROJECT_ABSPATH = "/home/osboxes/Documents/python/machlrn/music-recommendation"

# features to extract
FEATURES_CONSIDERED = [h5df_spe_getters.get_song_id_decoded,
                       h5df_getters.get_duration,
                       h5df_getters.get_end_of_fade_in,
                       h5df_getters.get_start_of_fade_out,
                       h5df_getters.get_key, # exists confidence
                       h5df_getters.get_loudness,
                       h5df_getters.get_mode, # exists confidence
                       h5df_getters.get_tempo,
                       h5df_getters.get_time_signature, # exists confidence
                       h5df_spe_getters.get_tatums_start_size,
                       h5df_spe_getters.get_tatums_start_min,
                       h5df_spe_getters.get_tatums_start_max,
                       h5df_spe_getters.get_tatums_start_mean,
                       h5df_spe_getters.get_tatums_start_var,
                       h5df_spe_getters.get_segments_start_size,
                       h5df_spe_getters.get_segments_timbre_min,
                       h5df_spe_getters.get_segments_timbre_max,
                       h5df_spe_getters.get_segments_timbre_mean,
                       h5df_spe_getters.get_segments_timbre_var,
                       h5df_spe_getters.get_segments_pitches_min,
                       h5df_spe_getters.get_segments_pitches_max,
                       h5df_spe_getters.get_segments_pitches_mean,
                       h5df_spe_getters.get_segments_pitches_var,
                       h5df_spe_getters.get_sections_start_size,
                       h5df_spe_getters.get_sections_start_var,
                       h5df_spe_getters.get_beats_start_size,
                       h5df_spe_getters.get_beats_start_var,
                       h5df_spe_getters.get_bars_start_size,
                       h5df_spe_getters.get_bars_start_var,
                       h5df_getters.get_year, # lacks => 0
                       h5df_getters.get_song_hotttnesss # lacks => nan
                      ]


# /home/osboxes/Documents/python/machlrn/music-recommendation/data/MillionSongSubset/data/A/T/C/TRATCDG128F932B4D2.h5
# /home/osboxes/Documents/python/machlrn/music-recommendation/data/MillionSongSubset/data/A/H/I/TRAHIRO128F93093A1.h5
# /home/osboxes/Documents/python/machlrn/music-recommendation/data/MillionSongSubset/data/A/Y/Y/TRAYYCA128F422B2A1.h5
# /home/osboxes/Documents/python/machlrn/music-recommendation/data/MillionSongSubset/data/A/Y/G/TRAYGHE12903CD213E.h5
def features_extractor(save, nb_elements=2939):
    """
        Create a pandas dataframe and save it
    """
    # nb .h5 paths = 2939
    df_paths_to_songs = pd.read_csv(os.path.join(PROJECT_ABSPATH, "data/SongsFeatures/df_abspath_to_features.txt"))
    paths_to_h5 = df_paths_to_songs.values[:, 0][:nb_elements]

    # initialize the pandas dataframe
    stock_features_name = []
    for fun in FEATURES_CONSIDERED:
        stock_features_name.append(str(fun).split(" ")[1])

    # extract all the features - stock (2939, 30)
    stock = []
    for i, file in Bar('Processing', max=nb_elements).iter(enumerate(paths_to_h5)):
        h5 = h5df_getters.open_h5_file_read(file)
        tmp = []
        for j, fun in enumerate(FEATURES_CONSIDERED):
            tmp.append(fun(h5))
        stock.append(tmp)
        h5.close()

    # create and save dataframe
    df = pd.DataFrame(data=stock, columns=stock_features_name)
    # df.to_csv(path_or_buf=save, index=False)




if __name__ == "__main__":
    save = os.path.join(PROJECT_ABSPATH, "data/SongsFeatures/df_uncleaned_features.csv")
    df = features_extractor(save=save, nb_elements=2939)
