import pandas as pd
import numpy as np
import transformers
from sklearn.metrics import classification_report, precision_recall_fscore_support
transformers.logging.set_verbosity_error()

def compute_performance(preds, true, trainvaltest, step):
    """
    :param preds: 模型预测的标签
    :param true: 真实标签
    :param trainvaltest: 一个字符串 表示当前的阶段（如"training"、"validation"、"test"）
    :param step: 当前训练步数 用于记录
    :return: 返回 F1 分数
    """
    print("preds: ", preds)
    print("true value: ", true)

    preds_np = preds.cpu().numpy()
    preds_np = np.argmax(preds_np, axis=1)
    true_np = true.cpu().numpy()
    print("-------------------------------------------------------------------------------------")
    print("真实值各个立场的数量: \n", pd.Series(true_np).value_counts())
    print("-------------------------------------------------------------------------------------")
    print("预测值各个立场的数量: \n", pd.Series(preds_np).value_counts())

    # 计算召回率
    print("-------------------------------------------------------------------------------------")
    print(trainvaltest + " classification_report for step: {}".format(step))
    print(classification_report(true_np, preds_np, labels=[0, 1, 2], digits=4))
    result_overall = precision_recall_fscore_support(true_np, preds_np, labels=[0, 1, 2], average=None, zero_division=1)
    result_macro = precision_recall_fscore_support(true_np, preds_np, labels=[0, 1, 2], average='macro',zero_division=1)

    result_every_type = [result_macro[0], result_macro[1], result_macro[2]]
    result_against = [result_overall[0][0], result_overall[1][0], result_overall[2][0]]
    result_favor = [result_overall[0][1], result_overall[1][1], result_overall[2][1]]
    result_none = [result_overall[0][2], result_overall[1][2], result_overall[2][2]]

    print("result_every_type:", result_every_type)
    print("result_favor:", result_favor)
    print("result_against:", result_against)
    print("result_none:", result_none)

    return result_macro[2]
