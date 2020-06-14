from mofpy.load_data import *
import mofpy
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_context('talk')


def main():

    model = mofpy.RegressorModel(model=load_model('pretrained/lattice_parameter.h5'))
    X, y, output = load_input_data(filtering_type=0)
    X = X[:, 20:70, 20:70, :]

    pred_table = pd.read_csv('test_results.csv')
    val_id = [id in pred_table['Id'].to_list() for
              id in output['ID'].to_list()]

    X_val = X[val_id]
    y_val = y[val_id]

    pred_table['Predicted_CNN'] = model.predict(X_val)

    sns.jointplot(x='Observed', y='Predicted_CNN', data=pred_table)
    plt.show()

    print('The R2 score for the whole dataset is: ', model.r_2_score(X_val, y_val))



if __name__ == '__main__':
    main()