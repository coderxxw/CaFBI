import os
import json
import logging
import pandas as pd
from numpy import int64

# Log format
_LOG_FORMAT = '%(asctime)s:%(levelname)s:%(name)s:%(message)s'


class SingletonLogger:
    """
    Singleton logger class to avoid creating multiple logger instances.
    """
    _instance = None

    @staticmethod
    def get_logger():
        """
        Get the singleton logger instance.
        If the instance doesn't exist, create a new logger instance and initialize it.
        """
        if SingletonLogger._instance is None:
            # Create a log formatter
            formatter = logging.Formatter(_LOG_FORMAT)
            # Create a log handler
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            # Create a logger
            logger = logging.getLogger('repeatflow')
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            SingletonLogger._instance = logger
        return SingletonLogger._instance


def load_configuration(file_path):
    """
    Load JSON configuration from the specified file.
    If the file does not exist, raise a FileNotFoundError.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file {file_path} not found")
    with open(file_path, 'r') as file:
        return json.load(file)


def apply_k_core_filter(data, min_user_interactions, min_item_interactions):
    """
    Apply k-core filtering to ensure that each user has at least min_user_interactions interactions
    and each item has at least min_item_interactions interactions.
    """
    if min_user_interactions <= 1 and min_item_interactions <= 1:
        return data

    def filter_users(dataframe):
        """
        Filter out users whose interaction count is greater than or equal to min_user_interactions.
        """
        user_interaction_counts = dataframe['org_user'].value_counts()
        valid_users = user_interaction_counts[user_interaction_counts >= min_user_interactions].index
        return dataframe[dataframe['org_user'].isin(valid_users)]

    def filter_items(dataframe):
        """
        Filter out items whose interaction count is greater than or equal to min_item_interactions.
        """
        item_interaction_counts = dataframe['org_item'].value_counts()
        valid_items = item_interaction_counts[item_interaction_counts >= min_item_interactions].index
        return dataframe[dataframe['org_item'].isin(valid_items)]

    unique_interactions = data[['org_user', 'org_item']].drop_duplicates()
    while True:
        previous_size = len(unique_interactions)
        unique_interactions = filter_users(unique_interactions)
        unique_interactions = filter_items(unique_interactions)
        if len(unique_interactions) == previous_size:
            break

    return data.merge(unique_interactions, on=['org_user', 'org_item'], how='inner')


def equalize_user_timestamps(data):
    """
    Adjust the timestamps for each user to be evenly distributed.
    """

    def adjust_timestamps(user_data):
        if len(user_data) <= 2:
            return user_data
        user_data = user_data.sort_values('timestamp')
        first_timestamp = int(user_data['timestamp'].iloc[0])
        last_timestamp = int(user_data['timestamp'].iloc[-1])
        new_timestamps = [first_timestamp + (i * (last_timestamp - first_timestamp) // (len(user_data) - 1))
                          for i in range(len(user_data))]
        user_data['timestamp'] = new_timestamps
        return user_data

    return data.groupby('org_user').apply(adjust_timestamps).reset_index(drop=True)


def filter_low_interaction_users(data):
    """
    Filter out users whose interaction count is below the 25th percentile.
    """
    user_interaction_counts = data.groupby('org_user')['org_item'].count()
    lower_quantile = user_interaction_counts.quantile(0.25)
    filtered_data = data[(data.groupby('org_user')['org_item'].transform('count') >= lower_quantile)]
    return filtered_data


def preprocess_dataset(config_path):
    """
    Main function to preprocess the dataset based on the configuration file.
    """
    logger = SingletonLogger.get_logger()
    logger.info(f"Loading configuration from {config_path}")

    # Load configuration and dataset parameters
    config_params = load_configuration(config_path)
    dataset_config = config_params['dataset']
    min_user_core = dataset_config.get('u_ncore', 1)
    min_item_core = dataset_config.get('i_ncore', 1)

    # Input and output file paths
    input_data_path = os.path.join(dataset_config['path'],
                                   f"{dataset_config['interactions']}.{dataset_config['file_format']}")
    output_data_path = os.path.join(dataset_config['path'],
                                    f"{dataset_config['name']}.csv")

    # Check if the preprocessed file already exists
    if os.path.exists(output_data_path):
        logger.info(f"Loading preprocessed dataset from {output_data_path}")
        data = pd.read_csv(output_data_path)
    else:
        logger.info(f"Reading raw data from {input_data_path}")
        data = pd.read_csv(input_data_path, sep=dataset_config['sep'], names=dataset_config['col_names'])

        # Apply filtering operations
        logger.info(
            f"Applying k-core filtering: user interactions ≥ {min_user_core}, item interactions ≥ {min_item_core}")
        data = apply_k_core_filter(data, min_user_core, min_item_core)
        data = filter_low_interaction_users(data)
        # Adjust timestamps
        data = equalize_user_timestamps(data)

        # Process timestamp data type
        data['timestamp'] = data['timestamp'].astype(str)
        data['timestamp'] = data['timestamp'].str[:-3]
        data['timestamp'] = data['timestamp'].astype(int64)

        logger.info(f"Saving preprocessed dataset to {output_data_path}")
        data.to_csv(output_data_path, index=False)

    # Log dataset statistics
    logger.info(f"Number of users: {data['org_user'].nunique()}")
    logger.info(f"Number of items: {data['org_item'].nunique()}")
    logger.info(f"Number of interactions: {len(data)}")

    # Create the log directory
    log_directory = 'exp/logs/LastFM'
    os.makedirs(log_directory, exist_ok=True)


if __name__ == "__main__":
    preprocess_dataset('configs/lastfm.json')
