import pandas as pd
import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt


def plot_result(X, y, model, savefig=True, filename=None):

    y_pred = model.predict(X).reshape(y.size)
    prediction = pd.DataFrame({'Observed': y, 'Predicted': y_pred})

    sns.jointplot(x='Observed', y='Predicted', data=prediction)

    if savefig:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()


def model_to_csv(model, filename='model_summary.csv'):
    model = model.model
    names = [layer.__class__.__name__ for layer in model.layers]
    output_shapes = [str(layer.output_shape) for layer in model.layers]
    parameters = [layer.count_params() for layer in model.layers]

    model_table = pd.DataFrame({'Name': names, 'Output Shapes': output_shapes, 'Parameters': parameters})
    model_table.to_csv(filename)


