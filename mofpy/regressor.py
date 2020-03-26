from keras.models import Input, Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.layers import Flatten, Dense
import keras
from sklearn.model_selection import train_test_split, KFold


class Regressor:

    def __init__(self, random_state=None, optimizer=None, batch_size=100,
                 input_shape=None, learning_rate=0.001, epochs=50):

        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_shape = input_shape

        if optimizer is None:
            self.optimizer = 'Adam'
        else:
            self.optimizer = optimizer

        self.learning_rate = learning_rate
        self.model = None

    def build_model(self):

        inp = Input(shape=self.input_shape)
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                   kernel_regularizer='l2')(inp)
        x = MaxPooling2D(2, 2)(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(filters=32, kernel_size=(3,3),activation='relu')(x)
        x = Conv2D(filters=32, kernel_size=(3,3),activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(filters=32, kernel_size=(3,3),activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = BatchNormalization(axis=-1)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=-1)(x)

        output = Dense(1)(x)
        model = Model(inp, output)
        self.model = model

    def compile(self):

        try:
            self.optimizer = getattr(keras.optimizers, self.optimizer)
        except:
            raise NotImplementedError('optimizer not implemented in keras')
        opt = self.optimizer(lr=self.learning_rate)
        self.model.compile(optimizer=opt, loss='mse')

    def fit(self, X, y, kfold=False, n_fold=10, verbose=1):

        if self.model is None:
            raise NotImplementedError('Model not initialized')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        if not kfold:
            self.training = self.model.fit(X_train, y_train, verbose=verbose, batch_size=self.batch_size,
                                           epochs=self.epochs, validation_data=(X_test, y_test))
        else:
            folds = KFold(n_splits=n_fold, shuffle=True, random_state=0)
            self.cv_scores = []
            for train, val in folds.split(X_train, y_train):
                self.compile()
                self.model.fit(X_train[train], y_train[train], batch_size=self.batch_size, verbose=verbose,
                               epochs=self.epochs, validation_data=[X_train[val], y_train[val]])
                self.cv_scores.append(self.model.evaluate(X_train[val], y_train[val]))

    def predict(self, X):

        return self.model.predict(X)

