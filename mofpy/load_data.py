import pandas as pd
import os
import pickle  # For python 3.xx
import numpy as np


def load_input_data():
    histogram_pickles_dir = os.path.join('data',
                                         'SA_interpolated_histogram_pickles_100')

    output = pd.read_csv(os.path.join('data', 'data_file.csv'))
    output.columns = output.columns.str.strip()

    histogram_pickles_files = \
        output.apply(lambda x:
                     os.path.join(histogram_pickles_dir,
                                  name_maker(x.formula, x.ID)), axis=1)
    names = []
    arr = []
    for f in histogram_pickles_files:
        if f.endswith('pkl'):
            names.append(f.split('/')[-1].split('.')[0])
            arr.append(pickle.load(open(f, 'rb'), encoding='latin1'))

    X = np.asarray(arr)

    y_act = output['relaxed']

    ind = y_act < 5
    y = y_act[ind].values
    X = X[ind]

    image_shape = X.shape

    X = X.reshape(image_shape[0], image_shape[1], image_shape[2], 1)

    return X, y


def name_maker(formula, ID):
    return 'converted_cifs'+formula+'_'+str(ID)+'_input.pkl'