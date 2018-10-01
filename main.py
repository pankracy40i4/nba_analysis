import logging
import warnings
import pandas as pd
from utils.timer import timed
from utils.data_transformations import load_data, data_scores_analysis, scale_and_split, data_transofrmation_for_model
from models.siamease_network import SiameaseNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@timed(logger)
def main():

    warnings.filterwarnings(module='tensorflow*', action='ignore', category=RuntimeWarning)

    data = load_data()
    data_scores_analysis(data)

    train, test, sc, cols = scale_and_split(data)

    left, right, label, df_info = data_transofrmation_for_model(train, cols)

    test_left, test_right, test_label, df_info_test = data_transofrmation_for_model(test, cols)

    model = SiameaseNetwork(len(cols), dropout=0.1)
    model.fit(left, right, label=label)

    model_eval = model.evaluate(test_left, test_right, test_label)
    logger.info('On the test season the model has %s matrices' % model_eval)


if __name__ == "__main__":

    main()
