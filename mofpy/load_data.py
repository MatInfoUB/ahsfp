import pandas as pd
import os
import pickle  # For python 3.xx
import numpy as np


def load_input_data(filtering_type=1):

    six_seven_row_elements = ['Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
                              'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os',
                              'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr',
                              'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf',
                              'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt',
                              'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']

    histogram_pickles_dir = os.path.join('data',
                                         'SA_interpolated_histogram_pickles_100')

    output = pd.read_csv(os.path.join('data', 'data_file.csv'))
    output.columns = output.columns.str.strip()

    output['A_atom'] = output['A_atom'].str.strip()
    output['B_atom'] = output['B_atom'].str.strip()

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

    output['X'] = arr

    if filtering_type:
        ind = output.apply(lambda x: x.A_atom in six_seven_row_elements or
                                     x.B_atom in six_seven_row_elements or
                                     x.relaxed > 1.05 * x.unrelaxed, axis=1)
    else:
        ind = output.apply(lambda x: x['relaxed'] > 5, axis=1)
        #  or
        #                                      x.relaxed > 1.02 * x.unrelaxed or
        #                                      x.relaxed == x.unrelaxed

    output = output[~ind]
    output = output.sort_values(by='ID')

    y = output['relaxed'].values
    X = output['X']

    X = np.asarray([x.tolist() for x in X])
    image_shape = X.shape

    X = X.reshape(image_shape[0], image_shape[1], image_shape[2], 1)

    return X, y, output


def name_maker(formula, ID):
    return 'converted_cifs'+formula+'_'+str(ID)+'_input.pkl'


def load_new_data():

    histogram_pickles_dir = os.path.join('data',
                                         'SA_interpolated_histogram_pickles_100_new')
    output = pd.read_csv(os.path.join('data', 'data_file.csv'))
    output.columns = output.columns.str.strip()
    newfilelist = os.listdir(histogram_pickles_dir)

    histogram_pickles_files = []
    inds = []
    for ind, row in output.iterrows():
        filename = name_maker(row.formula, row.ID)
        if filename in newfilelist:
            histogram_pickles_files.append(os.path.join(histogram_pickles_dir, filename))
            inds.append(ind)

    names = []
    arr = []
    for f in histogram_pickles_files:
        if f.endswith('pkl'):
            names.append(f.split('/')[-1].split('.')[0])
            arr.append(pickle.load(open(f, 'rb'), encoding='latin1'))

    output = output.iloc[inds]
    output['X'] = arr

    output = output.sort_values(by='ID')

    y = output['relaxed'].values
    X = output['X']

    X = np.asarray([x.tolist() for x in X])
    image_shape = X.shape

    X = X.reshape(image_shape[0], image_shape[1], image_shape[2], 1)

    return X, y, output

