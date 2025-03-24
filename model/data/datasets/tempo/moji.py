import os
import pickle
import numpy as np
from tqdm import tqdm

from model.data.datasets.tempo import TempoDataset
from model.utils.tempo import ts_to_high_level_tempo


class TempDataset(TempoDataset):
    def __init__(self, params):
        super().__init__(params)
        self._generate_fism()

    def _fetch_tempo_data(self):
        """Fetches additional temporal data and caches it."""
        self.logger.debug('Fetching additional temporal data')
        tempo_data_path = os.path.join(self.cache_path, self.model_name)
        os.makedirs(tempo_data_path, exist_ok=True)
        timedict_path = os.path.join(tempo_data_path, 'timedict.pkl')

        if os.path.exists(timedict_path):
            time_dict = pickle.load(open(timedict_path, 'rb'))
        else:
            time_set = self._aggregate_time_sets()
            time_dict = ts_to_high_level_tempo(time_set=time_set)
            pickle.dump(time_dict, open(timedict_path, 'wb'))

        self.data['time_dict'] = time_dict
        self.logger.debug('Finished fetching additional temporal data')

    def _aggregate_time_sets(self):
        """Aggregates unique timestamps from train, validation, and test sets."""
        train_set, valid_set, test_set = self.data['train_set'], self.data['valid_set'], self.data['test_set']
        time_set = self._time_set(train_set, indexes=self.data['train_interaction_indexes'])
        time_set.update(self._time_set(valid_set), self._time_set(test_set))
        return time_set

    def _generate_fism(self, n_epochs=100):
        """Generates FISM data if required by model parameters."""
        n_items = self.model_params['fism'].get('n_items', 0)
        if n_items > 0:
            self.logger.debug('Generating representative items for FISM')
            self._generate_fism_type('item', n_epochs, n_items)

    def _generate_fism_type(self, fism_type, n_epochs, n_items):
        """Generates FISM item or user data based on specified type."""
        fism_params = self.model_params['fism']
        cache_params = self.cache_params
        fism_sampling, fism_beta = fism_params['sampling'], fism_params['beta']

        tempo_data_path = os.path.join(
            self.cache_path, 'fism', fism_type, f'nitems{n_items}', f'{fism_sampling}-sampling')
        os.makedirs(tempo_data_path, exist_ok=True)

        elem_freq_dict = self._load_elem_freq_dict(fism_type, cache_params)

        for ep in range(n_epochs):
            fism_path = os.path.join(tempo_data_path, self._get_fism_filename(fism_type, fism_sampling, fism_beta, ep, n_items))
            if not os.path.exists(fism_path):
                fism_dict = self._generate_fism_dict(fism_type, elem_freq_dict, fism_beta, fism_sampling, n_items)
                pickle.dump(fism_dict, open(fism_path, 'wb'))

    def _load_elem_freq_dict(self, fism_type, cache_params):
        """Loads the element frequency dictionary from cache."""
        prefix = 'item_then_user' if fism_type == 'user' else 'user_then_item'
        elem_freq_path = os.path.join(
            self.cache_path,
            f'samples-step{self.dataset_params["samples_step"]}_'
            f'{cache_params["train_interactions"]}_aggregated_{prefix}_dict_{self.common}_'
            f'seqlen{self.seqlen}.pkl'
        )
        return pickle.load(open(elem_freq_path, 'rb'))

    def _get_fism_filename(self, fism_type, fism_sampling, fism_beta, ep, n_items):
        """Constructs the filename for FISM cache files."""
        filename = f'fism_nitems{n_items}_seqlen{self.seqlen}_{fism_sampling}-sampling'
        if fism_sampling != 'uniform':
            filename += f'_beta{fism_beta}'
        return f'{filename}_ep{ep}.pkl'

    def _generate_fism_dict(self, fism_type, elem_freq_dict, fism_beta, fism_sampling, n_items):
        """Generates FISM dictionary by sampling representative items."""
        fism_dict = {}
        n_elems = self.data['n_items'] if fism_type == 'user' else self.data['n_users']
        
        for eid in tqdm(range(1, n_elems + 1), desc='Generating FISM data'):
            elem_dict = elem_freq_dict[eid]
            iids, freqs = zip(*elem_dict.items())
            freqs = np.array(freqs, dtype=np.float32)
            if fism_beta != 1.0:
                freqs **= fism_beta
            freqs /= np.sum(freqs)

            chosen_items = self._sample_fism_items(iids, freqs, fism_sampling, n_items)
            fism_dict[eid] = chosen_items
        
        return fism_dict

    def _sample_fism_items(self, iids, freqs, fism_sampling, n_items):
        """Samples items for FISM based on the specified sampling method."""
        if n_items >= len(iids):
            return list(iids)
        if fism_sampling == 'uniform':
            return list(np.random.permutation(iids)[:n_items])
        return list(np.random.choice(iids, size=n_items, p=freqs, replace=False))
