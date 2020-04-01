import mofpy
import seaborn as sns
from mofpy.load_data import *

X, y = load_input_data()
input_shape = X.shape[1:]
model = mofpy.Regressor(input_shape=input_shape, epochs=100)

model.build_model()
model.compile()

model.fit(X, y, kfold=False, verbose=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
y_test_pred = model.predict(X_test).reshape(len(y_test))
import pandas as pd
Pred_table = pd.DataFrame({'Observed': y_test, 'Predicted':
    y_test_pred})
sns.jointplot(x='Observed', y='Predicted', data=Pred_table)
print('Testing R-squared is: ', model.r_2_score(X_test, y_test))
