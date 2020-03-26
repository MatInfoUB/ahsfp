import mofpy
import seaborn as sns

X, y = mofpy.load_input_data()
input_shape = X.shape[1:]
model = mofpy.Regressor(input_shape=input_shape)

model.build_model()
model.compile()

model.fit(X, y, verbose=1)
