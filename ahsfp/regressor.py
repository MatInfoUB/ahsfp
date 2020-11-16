from keras.models import Input, Model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout
from keras.layers import Flatten, Dense
import keras
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score


class ConvLayer(Conv2D):

    def __init__(self, filters=None, padding='same', strides=(1, 1)):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        filters = number of convolution filters
        """
        super(ConvLayer, self).__init__(filters=filters,
                                        kernel_size=(3, 3),
                                        activation='relu',
                                        padding=padding,
                                        strides=strides)


class RegressorModel:

    def __init__(self, random_state=None, optimizer='Adam', batch_size=100,
                 input_shape=None, learning_rate=0.001, epochs=50, model=None):

        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.model = model

    def build_model(self):

        inp = Input(shape=self.input_shape)
        x = ConvLayer(filters=32)(inp)
        x = ConvLayer(filters=32)(x)
        x = MaxPooling2D(2, 2)(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=-1)(x)
        for i in range(3):
            x = ConvLayer(filters=64)(x)
        x = MaxPooling2D(2, 2)(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization(axis=-1)(x)
        for i in range(4):
            x = ConvLayer(filters=128)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = BatchNormalization(axis=-1)(x)
        x = Flatten()(x)

        for i in range(2):

            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
            # x = BatchNormalization(axis=-1)(x)

        output = Dense(1)(x)
        model = Model(inp, output)
        self.model = model

    def compile(self):

        try:
            self.optimizer = getattr(keras.optimizers, self.optimizer_name)
        except:
            raise NotImplementedError('optimizer not implemented in keras')
        opt = self.optimizer(lr=self.learning_rate)
        self.model.compile(optimizer=opt, loss='mse')

    def fit(self, X, y, kfold=False, n_fold=10, verbose=1):

        if self.model is None:
            raise NotImplementedError('Model not initialized')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
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
                self.cv_scores.append(self.r_2_score(X_train[val], y_train[val]))

    def predict(self, X):

        return self.model.predict(X)

    def r_2_score(self, X, y):

        y_predict = self.model.predict(X)
        return r2_score(y_true=y, y_pred=y_predict)

    def residuals(self, X, y):

        return y - self.model.predict(X).reshape(len(y))


class smallRegressorModel:

    def __init__(self, random_state=None, optimizer='Adam', batch_size=100,
                 input_shape=None, learning_rate=0.001, epochs=50, model=None):

        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.model = model

    def build_model(self):

        inp = Input(shape=self.input_shape)
        x = ConvLayer(filters=32)(inp)
        # x = ConvLayer(filters=32)(x)
        x = MaxPooling2D(2, 2)(x)
        x = Dropout(0.5)(x)
        # x = BatchNormalization(axis=-1)(x)
        # for i in range(3):
        #     x = ConvLayer(filters=64)(x)
        x = ConvLayer(filters=32, strides=(2, 2))(x)
        x = MaxPooling2D(2, 2)(x)
        x = Dropout(0.5)(x)
        # x = BatchNormalization(axis=-1)(x)
        x = ConvLayer(filters=32, strides=(2, 2))(x)
        # for i in range(4):
        #     x = ConvLayer(filters=128)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        x = BatchNormalization(axis=-1)(x)
        x = Flatten()(x)

        for i in range(2):

            x = Dense(128, activation='relu', kernel_regularizer='l2')(x)
            x = Dropout(0.5)(x)
            # x = BatchNormalization(axis=-1)(x)

        output = Dense(1)(x)
        model = Model(inp, output)
        self.model = model

    def compile(self):

        try:
            self.optimizer = getattr(keras.optimizers, self.optimizer_name)
        except:
            raise NotImplementedError('optimizer not implemented in keras')
        opt = self.optimizer(lr=self.learning_rate)
        self.model.compile(optimizer=opt, loss='mse')

    def fit(self, X, y, kfold=False, n_fold=10, verbose=1):

        if self.model is None:
            raise NotImplementedError('Model not initialized')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
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
                self.cv_scores.append(self.r_2_score(X_train[val], y_train[val]))

    def predict(self, X):

        return self.model.predict(X)

    def r_2_score(self, X, y):

        y_predict = self.model.predict(X)
        return r2_score(y_true=y, y_pred=y_predict)

    def residuals(self, X, y):

        return y - self.model.predict(X).reshape(len(y))


class ClassifierModel:

    def __init__(self, random_state=None, optimizer='Adam', batch_size=100,
                 input_shape=None, learning_rate=0.001, epochs=50, model=None):

        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.optimizer_name = optimizer
        self.learning_rate = learning_rate
        self.model = model

    def build_model(self):

        inp = Input(shape=self.input_shape)
        x = ConvLayer(filters=32, padding='same')(inp)
        x = Dropout(0.5)(x)
        # x = ConvLayer(filters=32)(x)
        x = MaxPooling2D(2, 2)(x)

        # x = BatchNormalization(axis=-1)(x)
        for i in range(1):
            x = ConvLayer(filters=32, padding='same')(x)
        x = Dropout(0.5)(x)
        x = MaxPooling2D(2, 2)(x)

        # x = BatchNormalization(axis=-1)(x)
        for i in range(1):
            x = ConvLayer(filters=32, padding='same')(x)
        x = Dropout(0.5)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        for i in range(1):
            x = ConvLayer(filters=32, padding='same')(x)
        x = Dropout(0.5)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # x = BatchNormalization(axis=-1)(x)
        x = Flatten()(x)

        for i in range(2):

            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
            # x = BatchNormalization(axis=-1)(x)

        output = Dense(1, activation='sigmoid')(x)
        model = Model(inp, output)
        self.model = model