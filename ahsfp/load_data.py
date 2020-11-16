import pandas as pd
import os
import pickle  # For python 3.xx
import numpy as np
from pymatgen import Composition


def load_input_data(filtering_type=0):

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


def name_maker_new(formula, ID):
    return 'converted_cifs'+formula+'_'+str(ID)+'.pkl'


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


def name_to_split(name):

    atoms = Composition(name).formula.split(' ')[:2]
    a_atom = atoms[0][:-1]
    b_atom = atoms[1][:-1]

    return a_atom, b_atom


def load_kuzmanovski():

    pickle_dir = os.path.join('data', 'kuzmanovski_dataset_fp_pickles')
    filelist = os.listdir(pickle_dir)

    names = []
    arr = []
    a_atoms = []
    b_atoms = []

    for f in filelist:
        if f.endswith('pkl'):
            name = f.split('.')[0]
            a_atom, b_atom = name_to_split(name)
            a_atoms.append(a_atom)
            b_atoms.append(b_atom)
            names.append(name)
            arr.append(pickle.load(open(os.path.join(pickle_dir, f), 'rb'), encoding='latin1'))

    ann_table = pd.read_csv(os.path.join('data', 'kuzmanovski.csv'))


def load_formation_data():

    histogram_pickles_dir = os.path.join('data',
                                         'SA_interpolated_histogram_pickles_100_FE')

    output = pd.read_csv(os.path.join('data', 'output_table.csv'))

    histogram_pickles_files = \
        output.apply(lambda x:
                     os.path.join(histogram_pickles_dir,
                                  name_maker_new(x.Names, x.Entry_Id)), axis=1)

    names = []
    arr = []
    for f in histogram_pickles_files:
        if f.endswith('pkl'):
            names.append(f.split('/')[-1].split('.')[0])
            arr.append(pickle.load(open(f, 'rb'), encoding='latin1'))

    output['X'] = arr

    # y = output.Energy.values
    X = output['X']

    X = np.asarray([x.tolist() for x in X])
    image_shape = X.shape

    X = X.reshape(image_shape[0], image_shape[1], image_shape[2], 1)

    return X, output


def load_mech_prop_data():

    id_file = 'mech_prop_corrected.csv'
    output = pd.read_csv(id_file)

    histogram_pickles_dir = os.path.join('data',
                                         'SA_interpolated_histogram_pickles_100_new')

    histogram_pickles_files = \
        output.apply(lambda x:
                     os.path.join(histogram_pickles_dir,
                                  name_maker(x.OQMD_formula, x.OQMD_Id)), axis=1)

    names = []
    arr = []
    for f in histogram_pickles_files:
        if f.endswith('pkl'):
            names.append(f.split('/')[-1].split('.')[0])
            arr.append(pickle.load(open(f, 'rb'), encoding='latin1'))

    output['X'] = arr

    #Filtering Step
    y = output['c11']
    y -= y.mean()
    ind = y.abs() < 300

    # Applying the filtering
    output = output[ind]
    output = output.sort_values(by='OQMD_Id') #Sorting by IDs

    y = output['c11'].values
    X = output['X']

    X = np.asarray([x.tolist() for x in X])
    image_shape = X.shape

    X = X.reshape(image_shape[0], image_shape[1], image_shape[2], 1)

    return X, y, output


