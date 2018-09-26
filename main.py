import logging
import warnings
import pandas as pd
from utils.timer import timed
from utils.data_transformations import load_data, data_scores_analysis
from models.siamease_network import SiameaseNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@timed(logger)
def main():

    data = load_data()
    data_scores_analysis(data)


if __name__ == "__main__":

    main()
