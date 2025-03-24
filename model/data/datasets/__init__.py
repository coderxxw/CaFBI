from model.data.datasets.dataset import Dataset
from model.data.datasets.tempo.moji import TempDataset


_SUPPORTED_DATASETS = {
    'pos': Dataset,
    'tempo': TempDataset,
}


def dataset_factory(params):
    """
    Factory that generate dataset
    :param params:
    :return:
    """
    print(params)
    dataloader_type = params['dataset'].get('dataloader', 'pos')
    try:
        return _SUPPORTED_DATASETS[dataloader_type](params).data
    except KeyError:
        raise KeyError(f'Not support {dataloader_type} dataset')
