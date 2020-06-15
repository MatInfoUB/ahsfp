from ahsfp.load_data import *
import ahsfp
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--parameter', choices=['old_lattice', 'new_lattice'],
                    default='new_lattice', help='Choice of model')
parser.add_argument('--model_dir', default='pretrained', help='Directory of pretrained model')
parser.add_argument('--model_file', default='new_lattice_model.h5', help='Name of the pretrained model')
parser.add_argument('--data_dir', default='None', help='Directory of custom data')

FLAGS = parser.parse_args()

parameter = FLAGS.parameter
model_dir = FLAGS.model_dir
model_file = FLAGS.model_file
data_dir = FLAGS.data_dir


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

    y_pred = model.predict(X_val).reshape(y_val.size)
    prediction = pd.DataFrame({'Observed': y_val, 'Predicted': y_pred})

    sns.jointplot(x='Observed', y='Predicted', data=prediction)
    plt.savefig('figs/' + parameter, bbox_inches='tight', dpi=300)
    plt.show()
    print('The R2 score for the whole dataset is: ', model.r_2_score(X_val, y_val))


if __name__ == '__main__':
    predict()