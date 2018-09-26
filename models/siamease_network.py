import logging
from keras.layers import Input, Dense, Dot, Dropout, BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from utils.timer import timed
from config import MODELS_PATH


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SiameaseNetwork():
    """
    Siamease Network model.

    """
    def __init__(self, input_shape, dropout=0.1):
        """

        :param input_shape: int, number of features f
        :param dropout: float, dropout parameter in the dense block
        """
        self.input_shape = (input_shape, )
        self.dropout = dropout
        self.model = self.gen_model()
        self.batch_size = 32
        self.max_epochs = 200

    def dense_block(self, model, size):
        """
        Adds dense operations to the input model

        :param model: keras model, input model to be extended
        :param size: number of neurons
        :return: keras model, extended model
        """

        model.add(Dense(size, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout))

        return model

    def gen_model(self):
        """
        Generates Siamease network.

        :return:
        """

        left_input = Input(self.input_shape)
        right_input = Input(self.input_shape)

        encoder = Sequential()
        encoder = self.dense_block(encoder, 128)
        encoder = self.dense_block(encoder, 64)
        encoder.add(Dense(16))

        encoded_l = encoder(left_input)
        encoded_r = encoder(right_input)

        cosine = Dot(1, normalize=True)([encoded_l, encoded_r])
        prediction = Dense(1, activation='sigmoid')(cosine)

        siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
        optimizer = Adam(lr=0.0001)
        siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        return siamese_net

    @timed(logger)
    def fit(self, left, right, label):

        """
        Fits the created model with the provided data. The built-in validation loss is applied to avoid overfitting.

        :param left: array, input data
        :param right: array, input data
        :param label: labels
        :return:
        """

        assert (left.shape[1] == right.shape[1] == self.input_shape[0]), "Inputs are not compatible with model"

        assert (left.shape[0] == right.shape[0] == label.shape[0]), "Inputs of various shapes"

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto',
                                       baseline=None, restore_best_weights=True)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

        csv_logger = CSVLogger(MODELS_PATH + 'training.log')

        self.model.fit(x=[left, right], y=label, batch_size=self.batch_size)
        optimizer = Adam(lr=0.01)
        self.model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
        self.model.fit(x=[left, right], y=label, batch_size=self.batch_size, validation_split=0.1, shuffle=True,
                       epochs=self.max_epochs, callbacks=[early_stopping, reduce_lr, csv_logger])

    @timed(logger)
    def save(self, name='model_v100'):
        """
        Saves the model.

        :param name: str, name
        :return:
        """

        model_name = MODELS_PATH + name + '.h5'
        self.model.save(model_name)

