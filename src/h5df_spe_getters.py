import os
import pandas as pd
import numpy as np
import h5py

# import tools functions
import h5df_getters


# all the size depend on each song


def get_song_id_decoded(h5):
    return h5df_getters.get_song_id(h5).decode("utf-8")

# TATUMS
def get_tatums_start_size(h5):
    return len(np.array(h5df_getters.get_tatums_start(h5)))

def get_tatums_start_min(h5):
    array = np.array(h5df_getters.get_tatums_start(h5))
    if len(array) != 0:
        return np.min(array)
    else:
        return 0

def get_tatums_start_max(h5):
    array = np.array(h5df_getters.get_tatums_start(h5))
    if len(array) != 0:
        return np.max(array)
    else:
        return 0

def get_tatums_start_mean(h5):
    array = np.array(h5df_getters.get_tatums_start(h5))
    if len(array) != 0:
        return np.mean(array)
    else:
        return 0

def get_tatums_start_var(h5):
    array = np.array(h5df_getters.get_tatums_start(h5))
    if len(array) != 0:
        return np.min(array)
    else:
        return 0


# SEGMENTS
def get_segments_start_size(h5):
    return len(np.array(h5df_getters.get_segments_start(h5)))

# --- timbre
def get_segments_timbre_min(h5):
    return np.min(np.array(h5df_getters.get_segments_timbre(h5)))

def get_segments_timbre_max(h5):
    return np.max(np.array(h5df_getters.get_segments_timbre(h5)))

def get_segments_timbre_mean(h5):
    return np.mean(np.array(h5df_getters.get_segments_timbre(h5)))

def get_segments_timbre_var(h5):
    return np.var(np.array(h5df_getters.get_segments_timbre(h5)))

# --- pitches
def get_segments_pitches_min(h5):
    return np.min(np.array(h5df_getters.get_segments_pitches(h5)))

def get_segments_pitches_max(h5):
    return np.max(np.array(h5df_getters.get_segments_pitches(h5)))

def get_segments_pitches_mean(h5):
    return np.mean(np.array(h5df_getters.get_segments_pitches(h5)))

def get_segments_pitches_var(h5):
    return np.var(np.array(h5df_getters.get_segments_pitches(h5)))


# SECTION
def get_sections_start_size(h5):
    return len(np.array(h5df_getters.get_sections_start(h5)))

def get_sections_start_var(h5):
    return np.var(np.array(h5df_getters.get_sections_start(h5)))


# BEATS
def get_beats_start_size(h5):
    return len(np.array(h5df_getters.get_beats_start(h5)))

def get_beats_start_var(h5):
    return np.var(np.array(h5df_getters.get_beats_start(h5)))


# BARS
def get_bars_start_size(h5):
    return len(np.array(h5df_getters.get_bars_start(h5)))

def get_bars_start_var(h5):
    return np.var(np.array(h5df_getters.get_bars_start(h5)))
