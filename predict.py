from ahsfp.load_data import *
from ahsfp.utils import *
import ahsfp
from keras.models import load_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--parameter', choices=['old_lattice', 'new_lattice'],
                    default='new_lattice', help='Choice of model')
parser.add_argument('--model_dir', default='pretrained', help='Directory of pretrained model')
parser.add_argument('--model_file', default='new_lattice_model.h5', help='Name of the pretrained model')
parser.add_argument('--data_dir', default='None', help='Directory of custom data')
parser.add_argument('--vis', default=True, choices=[True, False], type=bool, help='Choose if observed vs predicted plot is shown')

FLAGS = parser.parse_args()

parameter = FLAGS.parameter
model_dir = FLAGS.model_dir
model_file = FLAGS.model_file
data_dir = FLAGS.data_dir
vis = FLAGS.vis

sns.set_context('talk')


def predict():

    X, y, X_val, y_val = None, None, None, None
    if parameter == 'old_lattice':
        model = ahsfp.RegressorModel(model=load_model('pretrained/lattice_parameter.h5'))
        X, y, output = load_input_data(filtering_type=0)
    elif parameter == 'new_lattice':
        model = ahsfp.RegressorModel(model=load_model('pretrained/new_lattice_model.h5'))
        X, y, output = load_new_data()

    X = X[:, 20:70, 20:70, :]

    if parameter == 'old_lattice':
        pred_table = pd.read_csv('test_results.csv')
    elif parameter == 'new_lattice':
        pred_table = pd.read_csv('test_results_lattice.csv')

    val_id = [id in pred_table['Id'].to_list() for id in output['ID'].to_list()]

    X_val = X[val_id]
    y_val = y[val_id]

    if vis:
        plot_result(X=X_val, y=y_val, model=model, filename='figs/' + parameter)
    print('The R2 score for the whole dataset is: ', model.r_2_score(X_val, y_val))


if __name__ == '__main__':
    predict()