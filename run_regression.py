import mofpy
import seaborn as sns
from mofpy.load_data import *

pred_table = pd.read_csv('test_results.csv')
pred_table = pred_table.sort_values(by='Id')
X, y, output = load_input_data(filtering_type=0)


val_id = [id in pred_table['Id'].to_list() for
          id in output['ID'].to_list()]


input_shape = X.shape[1:]
model = mofpy.Regressor(input_shape=input_shape, epochs=100, batch_size=300)

model.build_model()
model.compile()

model.fit(X, y, kfold=False, verbose=1)


X_val = X[val_id]
y_val = y[val_id]
pred_table['Predicted_CNN'] = model.predict(X_val)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
y_train_pred = model.predict(X_train).reshape(len(y_train))
y_test_pred = model.predict(X_test).reshape(len(y_test))
import pandas as pd
Pred_table = pd.DataFrame({'Observed': y_test, 'Predicted':
    y_test_pred})
Train_table = pd.DataFrame({'Observed': y_train, 'Predicted':
    y_train_pred})
print('Testing R-squared is: ', model.r_2_score(X_test, y_test))
sns.jointplot(x='Observed', y='Predicted', data=Pred_table)

print('Training R-squared is: ', model.r_2_score(X_train, y_train))
pred_table.to_csv('test_results.csv')


