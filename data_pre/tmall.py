import os
import json
import logging
from datetime import datetime

import pandas as pd

_FORMAT = '%(asctime)s:%(levelname)s:%(name)s:%(message)s'

class Logger:
    """Singleton logger class to avoid multiple logger instances."""
    _instance = None

    @staticmethod
    def get_logger():
        if Logger._instance is None:
            formatter = logging.Formatter(_FORMAT)
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger = logging.getLogger('repeatflow')
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            Logger._instance = logger
        return Logger._instance


def load_configuration(file_path):
    """Load JSON configuration from the given file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file {file_path} not found")
    with open(file_path, 'r') as file:
        return json.load(file)


def filter_kcore(data, u_ncore, i_ncore):
    """
    Apply k-core filtering to ensure users have at least `u_ncore` interactions
    and items have at least `i_ncore` interactions.
    """
    if u_ncore <= 1 and i_ncore <= 1:
        return data

    def filter_users(df):
        """Filter users with interactions >= u_ncore."""
        user_counts = df['org_user'].value_counts()
        valid_users = user_counts[user_counts >= u_ncore].index
        return df[df['org_user'].isin(valid_users)]

    def filter_items(df):
        """Filter items with interactions >= i_ncore."""
        item_counts = df['org_item'].value_counts()
        valid_items = item_counts[item_counts >= i_ncore].index
        return df[df['org_item'].isin(valid_items)]

    unique_data = data[['org_user', 'org_item']].drop_duplicates()
    while True:
        prev_size = len(unique_data)
        unique_data = filter_users(unique_data)
        unique_data = filter_items(unique_data)
        if len(unique_data) == prev_size:
            break

    return data.merge(unique_data, on=['org_user', 'org_item'], how='inner')

def equalize_timestamp(data):
    """Adjust timestamps to be evenly distributed for each user."""
    def adjust_timestamps(user_data):
        if len(user_data) <= 2:
            return user_data
        user_data = user_data.sort_values('timestamp')
        first, last = user_data['timestamp'].iloc[0], user_data['timestamp'].iloc[-1]
        new_timestamps = [first + (i * (last - first) // (len(user_data) - 1))
                          for i in range(len(user_data))]
        user_data['timestamp'] = new_timestamps
        return user_data

    return data.groupby('org_user').apply(adjust_timestamps).reset_index(drop=True)

def remove_behavior_type_one(data):
    """
    Remove rows where behavior_type is 1.
    """
    return data[data['behavior_type'] != 1]

def convert_time_to_timestamp(data):
    """
    Convert the 'time' column to timestamp in seconds.
    """
    def time_to_timestamp(time_str):
        dt = datetime.strptime(time_str, '%Y-%m-%d %H')
        return int(dt.timestamp())
    data['timestamp'] = data['timestamp'].apply(time_to_timestamp)
    return data


def filter_interaction(data):
    user_interaction_counts = data.groupby('org_user')['org_item'].count()
    lower_quantile = user_interaction_counts.quantile(0.5)
    upper_quantile = user_interaction_counts.quantile(1)
    filtered_data = data[(data.groupby('org_user')['org_item'].transform('count') >= lower_quantile) &
                      (data.groupby('org_user')['org_item'].transform('count') <= upper_quantile)]
    return filtered_data


def preprocess_dataset(config_path):
    """Main function to preprocess dataset based on configuration."""
    logger = Logger.get_logger()
    logger.info(f"Loading configuration from {config_path}")

    # Load config and dataset parameters
    params = load_configuration(config_path)
    dataset_params = params['dataset']
    u_ncore, i_ncore = dataset_params.get('u_ncore', 1), dataset_params.get('i_ncore', 1)

    # Paths for input and output files
    data_path = os.path.join(dataset_params['path'],
                             f"{dataset_params['interactions']}.{dataset_params['file_format']}")
    output_path = os.path.join(dataset_params['path'],
                               f"{dataset_params['name']}.csv")

    # Check if processed file already exists
    if os.path.exists(output_path):
        logger.info(f"Loading preprocessed dataset from {output_path}")
        data = pd.read_csv(output_path)
    else:
        logger.info(f"Reading raw data from {data_path}")
        data = pd.read_csv(data_path, sep=dataset_params['sep'], names=dataset_params['col_names'], header=None, skiprows=1)
        data = remove_behavior_type_one(data)
        data = convert_time_to_timestamp(data)

        # Apply filtering
        logger.info(f"Applying k-core filter: users ≥ {u_ncore}, items ≥ {i_ncore}")
        data = filter_kcore(data, u_ncore, i_ncore)
        # logger.info("Equalizing timestamps")
        data = filter_interaction(data)
        data = equalize_timestamp(data)
        data = data[['org_user', 'org_item', 'timestamp']]
        data = data.drop_duplicates()

        logger.info(f"Saving preprocessed dataset to {output_path}")
        data.to_csv(output_path, index=False)

    # Log dataset statistics
    logger.info(f"Number of users: {data['org_user'].nunique()}")
    logger.info(f"Number of items: {data['org_item'].nunique()}")
    logger.info(f"Number of interactions: {len(data)}")

    log_dir = 'exp/logs/Tmall'
    os.makedirs(log_dir, exist_ok=True)


if __name__ == "__main__":
    preprocess_dataset('configs/tmall.json')
