import numpy as np
from tqdm import tqdm

from model.logging import get_logger


class Evaluator:
    @classmethod
    def eval(cls, dataloader, model, item_pops=None):
        """
        Get score on valid/test dataset
        :param dataloader:
        :param model:
        :return:
        """
        top_k = [5, 10, 20]
        ndcg = np.zeros(len(top_k))
        hr = np.zeros(len(top_k))
        mrr = np.zeros(len(top_k))

        n_users = 0
        n_batches = dataloader.get_num_batches()
        # for each batch
        for _ in tqdm(range(1, n_batches),
                      desc='Evaluating...'):
            # get batch data
            batch_data = dataloader.next_batch()
            feed_dict = model.build_feedict(batch_data,
                                            is_training=False)
            # get prediction from model
            predictions = -model.predict(feed_dict)
            # calculate evaluation metrics
            for _, pred in enumerate(predictions):
                n_users += 1
                rank = pred.argsort().argsort()[0]
                for j, k in enumerate(top_k):
                    if rank < k:
                        ndcg[j] += 1 / np.log2(rank + 2)
                        hr[j] += 1
                        mrr[j] += 1 / (rank + 1)

        out = (ndcg / n_users, hr / n_users, mrr / n_users)
        return out
