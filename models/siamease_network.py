import logging
import tensorflow as tf
from keras.layers import Input, Dense, Dot, Dropout, BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger
from utils.timer import timed
from config import MODELS_PATH


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def recall(y_true, y_pred):
    """
    Wraper for tf streaming recall in keras.

    :param y_true: array, true labels
    :param y_pred: array, predicted labels
    :return: float, recall metric
    """
    # any tensorflow metric
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'recall' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value


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
        self.metrics = [recall]
        self.batch_size = 32
        self.max_epochs = 200
        self.model = self.gen_model()

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
        encoder = self.dense_block(encoder, 32)
        encoder = self.dense_block(encoder, 16)
        encoder.add(Dense(8))

        encoded_l = encoder(left_input)
        encoded_r = encoder(right_input)

        cosine = Dot(1, normalize=True)([encoded_l, encoded_r])
        prediction = Dense(1, activation='sigmoid')(cosine)

        siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)
        optimizer = Adam(lr=0.0001)
        siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=self.metrics)

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

        # assert (left.shape[1] == right.shape[1] == self.input_shape[0]), "Inputs are not compatible with model"
        #
        # assert (left.shape[0] == right.shape[0] == label.shape[0]), "Inputs of various shapes"

        early_stopping = EarlyStopping(monitor='recall', min_delta=0.005, patience=10, verbose=0, mode='max')

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001, min_delta=0.005)

        csv_logger = CSVLogger(MODELS_PATH + 'training.log')

        self.model.fit(x=[left, right], y=label, batch_size=self.batch_size)
        optimizer = Adam(lr=0.01)
        self.model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=self.metrics)
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

    def evaluate(self, left, right, label):
        """
        Evaluates the model.

        :param left: array, input data
        :param right: array, input data
        :param label: labels
        :return:
        """

        evaluation = self.model.evaluate(x=[left, right], y=label, batch_size=self.batch_size)
        return evaluation


