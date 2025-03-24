import os
import numpy as np
import tensorflow as tf
from model.logging import get_logger
from model.utils.params import process_params
from model.data.datasets import dataset_factory
from model.models import ModelFactory
from model.data.loaders import dataloader_factory
from tqdm import tqdm

# 选择用户 1
target_user_id = "1"

def entrypoint(params):
    logger = get_logger()
    tf.compat.v1.disable_eager_execution()
    # 处理参数
    training_params, model_params = process_params(params)
    model_name = model_params['name']
    dataset_params = params['dataset']
    cache_path = params['cache']['path']
    if model_params['type'] == 'tempo' and \
            model_params['name'] == 'tisasrec':
        params['cache']['path'] = os.path.join(
            params['cache']['path'],
            f'seqlen{model_params["params"]["seqlen"]}')
    # 创建模型目录（如果不存在）
    if not os.path.exists(training_params['model_dir']):
        os.makedirs(training_params['model_dir'], exist_ok=True)
    # logger.info(training_params['model_dir'])
    timespan = model_params['params'].get(
        'timespan', 256)
    # 加载数据集
    data = dataset_factory(params=params)

    # start model training
    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True
    with tf.compat.v1.Session(config=sess_config) as sess:
        # # 选择用户 1
        # target_user_id = "3"
        # uid_series = [target_user_id]

        model = ModelFactory.generate_model(sess=sess,
                                            params=training_params,
                                            n_users=data['n_users'],
                                            n_items=data['n_items'],
                                            command='eval')


        # 获取top-k推荐item及其得分
        batch_size = training_params['batch_size']
        num_scored_users = params['eval'].get('n_users')
        random_seeds = params['eval'].get('random_seeds')
        best_epoch = params['best_epoch']
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
        if 'negative_sampling' in params.get('case_study', {}):
            neg_sampling = params['case_study']['negative_sampling']['type']

        test_dataloader = dataloader_factory(
            data=data,
            batch_size=batch_size,
            seqlen=seqlen,
            mode='test',
            random_seed=42,  # 可以从配置中获取随机种子
            num_scored_users=num_scored_users,
            model_name=model_name,
            timespan=timespan,
            cache_path=cache_path,
            epoch=best_epoch,
            n_fism_items=n_fism_items,
            fism_sampling=fism_sampling,
            fism_type=fism_type,
            fism_beta=fism_beta,
            neg_sampling=neg_sampling
        )

        top_10_items = get_top_10_items(test_dataloader, model, target_user_id)
        if top_10_items:
            print(f"用户 {target_user_id} 预测分数前 10 的item：")
            for item_id, score in top_10_items:
                print(f"item ID: {item_id}, 分数: {score}")
        else:
            print(f"未找到用户 {target_user_id} 的数据。")

# 原始
# def get_top_10_items(dataloader, model, target_user_id):#
#     n_batches = dataloader.get_num_batches()
#     for _ in tqdm(range(1, n_batches), desc='Evaluating...'):
#         # 获取批次数据
#         batch_data = dataloader.next_batch()
#         feed_dict = model.build_feedict(batch_data,
#                                         is_training=False)
#         # 从模型获取预测结果
#         predictions = -model.predict(feed_dict)
#         # 获取用户 ID 和测试物品 ID
#         user_ids = batch_data[0]
#         test_item_ids = batch_data[2]
#         for i, user_id in enumerate(user_ids):
#             if str(user_id) == target_user_id:
#                 user_scores = predictions[i]
#                 # 找出分数最高的 10 项物品的索引
#                 top_10_indices = np.argsort(user_scores)[-10:][::-1]
#                 top_10_item_ids = [test_item_ids[i][idx] for idx in top_10_indices]
#                 top_10_scores = [user_scores[idx] for idx in top_10_indices]
#                 return list(zip(top_10_item_ids, top_10_scores))
#     return []
#

# item去重
def get_top_10_items(dataloader, model, target_user_id):
    top_items = []
    seen_items = set()
    n_batches = dataloader.get_num_batches()
    for _ in tqdm(range(1, n_batches), desc='Evaluating...'):
        # 获取批次数据
        batch_data = dataloader.next_batch()
        feed_dict = model.build_feedict(batch_data,
                                        is_training=False)
        # 从模型获取预测结果
        predictions = -model.predict(feed_dict)
        # 获取用户 ID 和测试物品 ID
        user_ids = batch_data[0]
        test_item_ids = batch_data[2]
        for i, user_id in enumerate(user_ids):
            if str(user_id) == target_user_id:
                user_scores = predictions[i]
                # 对预测分数进行排序
                sorted_indices = np.argsort(user_scores)[::-1]
                for idx in sorted_indices:
                    item_id = test_item_ids[i][idx]
                    score = user_scores[idx]
                    if item_id not in seen_items:
                        top_items.append((item_id, score))
                        seen_items.add(item_id)
                        if len(top_items) == 10:
                            return top_items
    return top_items
#
# def  get_top_10_items(dataloader, model, target_user_id):
#     n_batches = dataloader.get_num_batches()
#     correct_count_top1 = 0
#     correct_count_top10 = 0
#     total_count = 0
#     for _ in tqdm(range(1, n_batches), desc='Evaluating...'):
#         # 获取批次数据
#         batch_data = dataloader.next_batch()
#         user_ids = batch_data[0]
#         test_item_ids = batch_data[2]
#         for i, user_id in enumerate(user_ids):
#             if str(user_id) == target_user_id:
#                 # 取前 n - 1 个数据
#                 prev_items = test_item_ids[i][:-1]
#                 actual_nth_item = test_item_ids[i][-1]
#
#                 # 构造新的批次数据
#                 new_batch_data = list(batch_data)
#                 new_batch_data[2][i] = prev_items
#
#                 feed_dict = model.build_feedict(new_batch_data,
#                                                 is_training=False)
#                 # 从模型获取预测结果
#                 predictions = -model.predict(feed_dict)
#                 user_scores = predictions[i]
#                 # 找出预测分数最高的 10 个物品的索引
#                 top_10_indices = np.argsort(user_scores)[-10:][::-1]
#                 top_10_predicted_items = [test_item_ids[i][idx] for idx in top_10_indices]
#                 top_1_predicted_item = top_10_predicted_items[0]
#
#                 total_count += 1
#                 if top_1_predicted_item == actual_nth_item:
#                     correct_count_top1 += 1
#                 if actual_nth_item in top_10_predicted_items:
#                     correct_count_top10 += 1
#
#                 print(f"实际第 n 个物品 ID: {actual_nth_item}")
#                 print(f"预测前 10 个物品 ID: {top_10_predicted_items}")
#
#     if total_count > 0:
#         accuracy_top1 = correct_count_top1 / total_count
#         accuracy_top10 = correct_count_top10 / total_count
#         print(f"Top 1 预测准确率: {accuracy_top1 * 100:.2f}%")
#         print(f"Top 10 预测准确率: {accuracy_top10 * 100:.2f}%")
