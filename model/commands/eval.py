import os
import numpy as np
import tensorflow as tf

from model.logging import get_logger
from model.utils.params import process_params
from model.data.datasets import dataset_factory
from model.models import ModelFactory
from model.data.loaders import dataloader_factory
from model.eval.evaluator import Evaluator


def entrypoint(params):
    """ Command entrypoint
    :param params: Deserialized JSON configuration file
                   provided in CLI args.
    """
    logger = get_logger()
    tf.compat.v1.disable_eager_execution()
    # process params
    training_params, model_params = process_params(params)
    model_name = model_params['name']
    dataset_params = params['dataset']
    cache_path = params['cache']['path']
    if model_params['type'] == 'tempo' and \
            model_params['name'] == 'tisasrec':
        params['cache']['path'] = os.path.join(
            params['cache']['path'],
            f'seqlen{model_params["params"]["seqlen"]}')
    # create model directory if not exist
    if not os.path.exists(training_params['model_dir']):
        os.makedirs(training_params['model_dir'], exist_ok=True)
    logger.info(training_params['model_dir'])
    timespan = model_params['params'].get(
        'timespan', 256)
    # load dataset
    data = dataset_factory(params=params)

    # start model training
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    with tf.compat.v1.Session(config=sess_config) as sess:
        model = ModelFactory.generate_model(sess=sess,
                                            params=training_params,
                                            n_users=data['n_users'],
                                            n_items=data['n_items'],
                                            command='eval')
        # generate users for test
        scores = []
        batch_size = training_params['batch_size']
        num_scored_users = params['eval'].get('n_users')
        random_seeds = params['eval'].get('random_seeds')
        best_epoch = params['best_epoch']
        logger.info(f'Make sure the best VALIDATION on EPOCH #{best_epoch}')
        seqlen = model_params['params'].get('seqlen', 50)
        n_fism_items = -1
        fism_sampling = 'uniform'
        fism_type = 'item'
        fism_beta = 1.0
        if 'fism' in model_params['params']:
            n_fism_items = model_params['params']['fism']['n_items']
            fism_sampling = model_params['params']['fism']['sampling']
            fism_type = model_params['params']['fism']['type']
            fism_beta = model_params['params']['fism']['beta']
        neg_sampling = 'uniform'
        if 'negative_sampling' in params['eval']:
            neg_sampling = params['eval']['negative_sampling']['type']
        for step, seed in enumerate(random_seeds):
            logger.info(f'EVALUATION for #{step + 1} COHORT')
            test_dataloader = dataloader_factory(
                data=data,
                batch_size=batch_size,
                seqlen=seqlen,
                mode='test',
                random_seed=seed,
                num_scored_users=num_scored_users,
                model_name=model_name,
                timespan=timespan,
                cache_path=cache_path,
                epoch=best_epoch,
                n_fism_items=n_fism_items,
                fism_sampling=fism_sampling,
                fism_type=fism_type,
                fism_beta=fism_beta,
                neg_sampling=neg_sampling)

            score = Evaluator.eval(test_dataloader, model,
                                   item_pops=None)
            ndcg, hr, mrr = score
            message = [f'Step #{step + 1}',
                       f'Val NDCG@5 {ndcg[0]:7.4f}',
                       f'Val NDCG@10 {ndcg[1]:7.4f}',
                       f'Val NDCG@20 {ndcg[2]:7.4f}',

                        f'Val HR@5 {hr[0]:7.4f}',
                        f'Val HR@10 {hr[1]:7.4f}',
                        f'Val HR@20 {hr[2]:7.4f}',

                        f'Val mrr@5 {mrr[0]:7.4f}',
                        f'Val mrr@10 {mrr[1]:7.4f}',
                        f'Val mrr@20 {mrr[2]:7.4f}']
            logger.info(','.join(message))
            scores_10 = [ndcg[1], hr[1], mrr[1]]
            scores.append(scores_10)
        ndcg, hr, mrr = zip(*scores)
        message = ['RESULTS:',
                   f'NDCG@10: {np.mean(ndcg):8.5f} +/- {np.std(ndcg):8.5f}',
                   f'HR@10: {np.mean(hr):8.5f} +/- {np.std(hr):8.5f}',
                   f'mrr@10: {np.mean(mrr):8.5f} +/- {np.std(mrr):8.5f}',
                   ]
        logger.info('\n'.join(message))