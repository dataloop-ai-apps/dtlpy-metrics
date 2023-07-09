import os
import numpy as np
import pandas as pd
import dtlpy as dl
import matplotlib.pyplot as plt
import datetime


# from typing import Dict, Union, List, Any

def calc_precision_recall(dataset_id: str,
                          model_id: str,
                          iou_threshold=0.01) -> pd.DataFrame:
    """
    Plot precision recall curve for a given metric threshold

    :param dataset_id: str dataset ID
    :param model_id: str model ID
    :return: dataframe with all the points to plot for the dataset and individual labels
    """
    ################################
    # get matched annotations data #
    ################################
    model_filename = f'{model_id}.csv'
    filters = dl.Filters(field='hidden', values=True)
    filters.add(field='name', values=model_filename)
    dataset = dl.datasets.get(dataset_id=dataset_id)
    items = list(dataset.items.list(filters=filters).all())
    if len(items) == 0:
        raise ValueError(f'No scores found for model ID {model_id}.')
    elif len(items) > 1:
        raise ValueError(f'Found {len(items)} items with name {model_id}.')
    else:
        scores_file = items[0].download()

    scores = pd.read_csv(scores_file)
    labels = dataset.labels
    label_names = [label.tag for label in labels]
    if len(label_names) == 0:
        label_names = list(pd.concat([scores.first_label, scores.second_label]).dropna().drop_duplicates())

    ##############################
    # calculate precision/recall #
    ##############################
    dataset_points = {'iou_threshold': {},
                      'data': {},  # "dataset" or "label"
                      'label_name': {},  # label name or NA,
                      'precision': {},
                      'recall': {},
                      'confidence': {}
                      }

    num_gts = sum(scores.first_id.notna())
    detections = scores[scores.second_id.notna()].copy()

    detections.sort_values('second_confidence', inplace=True, ascending=False, ignore_index=True)
    detections['true_positives'] = detections['geometry_score'] >= iou_threshold
    detections['false_positives'] = detections['geometry_score'] < iou_threshold

    # get dataset-level precision/recall
    dataset_fps = np.cumsum(detections['false_positives'])
    dataset_tps = np.cumsum(detections['true_positives'])
    dataset_recall = dataset_tps / num_gts
    dataset_precision = np.divide(dataset_tps, (dataset_fps + dataset_tps))

    [_,
     dataset_plot_precision,
     dataset_plot_recall,
     dataset_plot_confidence] = every_point_curve(recall=list(dataset_recall),
                                                  precision=list(dataset_precision),
                                                  confidence=list(detections['second_confidence']))

    dataset_points['iou_threshold'] = [iou_threshold] * len(dataset_plot_precision)
    dataset_points['data'] = ['dataset'] * len(dataset_plot_precision)
    dataset_points['label_name'] = ['NA'] * len(dataset_plot_precision)
    dataset_points['precision'] = dataset_plot_precision
    dataset_points['recall'] = dataset_plot_recall
    dataset_points['confidence'] = dataset_plot_confidence

    dataset_df = pd.DataFrame(dataset_points).drop_duplicates()

    ##########################################
    # calculate label-level precision/recall #
    ##########################################
    all_labels = pd.DataFrame(columns=dataset_df.columns)

    label_points = {key: {} for key in dataset_points}

    for label_name in list(set(label_names)):
        label_detections = detections.loc[
            (detections.first_label == label_name) | (detections.second_label == label_name)].copy()
        if label_detections.shape[0] == 0:
            label_plot_precision = [0]
            label_plot_recall = [0]
            label_plot_confidence = [0]
        else:
            label_detections.sort_values('second_confidence', inplace=True, ascending=False, ignore_index=True)

            label_fps = np.cumsum(label_detections['false_positives'])
            label_tps = np.cumsum(label_detections['true_positives'])
            label_recall = label_tps / num_gts
            label_precision = np.divide(label_tps, (label_fps + label_tps))

            [_,
             label_plot_precision,
             label_plot_recall,
             label_plot_confidence] = \
                every_point_curve(recall=list(label_recall),
                                  precision=list(label_precision),
                                  confidence=list(label_detections['second_confidence']))

        label_points['iou_threshold'] = [iou_threshold] * len(label_plot_precision)
        label_points['data'] = ['label'] * len(label_plot_precision)
        label_points['label_name'] = [label_name] * len(label_plot_precision)
        label_points['precision'] = label_plot_precision
        label_points['recall'] = label_plot_recall
        label_points['confidence'] = label_plot_confidence

        label_df = pd.DataFrame(label_points).drop_duplicates()
        all_labels = pd.concat([all_labels, label_df])

    ##################
    # concat all data #
    ##################
    plot_points = pd.concat([dataset_df, all_labels])

    return plot_points


def every_point_curve(recall: list, precision: list, confidence: list):
    """
    Calculate precision-recall curve from a list of precision & recall values
    :param recall: list of recall values
    :param precision: list of precision values
    :return: list of average precision all values, precision points, recall points
    """
    recall_points = np.concatenate([[0], recall, [1]])
    precision_points = np.concatenate([[0], precision, [0]])
    confidence_points = np.concatenate([[confidence[0]], confidence, [confidence[-1]]])

    # find the maximum precision between each recall value, backwards
    for i in range(len(precision_points) - 1, 0, -1):
        precision_points[i - 1] = max(precision_points[i - 1], precision_points[i])

    # build the simplified recall list, removing values when the precision doesnt change
    recall_intervals = []
    for i in range(len(recall_points) - 1):
        if recall_points[1 + i] != recall_points[i]:
            recall_intervals.append(i + 1)

    avg_precis = 0
    for i in recall_intervals:
        avg_precis = avg_precis + np.sum((recall_points[i] - recall_points[i - 1]) * precision_points[i])
    return [avg_precis,
            precision_points[0:len(precision_points) - 1],
            recall_points[0:len(precision_points) - 1],
            confidence_points[0:len(precision_points) - 1]]


def calc_confusion_matrix(dataset_id: str,
                          model_id: str,
                          metric: str,
                          show_unmatched=True):
    """
    Calculate confusion matrix for a given model and metric
    :param dataset_id:
    :param model_id:
    :param metric:
    :param show_unmatched: display extra column showing which GT annotations were not matched
    :return:
    """
    if metric.lower() == 'iou':
        metric = 'geometry_score'
    elif metric.lower() == 'accuracy':
        metric = 'label_score'

    model_filename = f'{model_id}.csv'
    filters = dl.Filters(field='hidden', values=True)
    filters.add(field='name', values=model_filename)
    dataset = dl.datasets.get(dataset_id=dataset_id)
    items = list(dataset.items.list(filters=filters).all())
    if len(items) == 0:
        raise ValueError(f'No scores found for model ID {model_id}.')
    elif len(items) > 1:
        raise ValueError(f'Found {len(items)} items with name {model_id}.')
    else:
        scores_file = items[0].download()

    scores = pd.read_csv(scores_file)
    labels = dataset.labels
    label_names = [label.tag for label in labels]

    if metric not in scores.columns:
        raise ValueError(f'{metric} metric not included in scores.')

    #########################
    # plot precision/recall #
    #########################
    # calc
    if labels is None:
        label_names = pd.concat([scores.first_label, scores.second_label]).dropna()

    scores_cleaned = scores.dropna().reset_index(drop=True)
    scores_labels = scores_cleaned[['first_label', 'second_label']]
    grouped_labels = scores_labels.groupby(['first_label', 'second_label']).size()

    conf_matrix = pd.DataFrame(index=label_names, columns=label_names, data=0)
    for label1, label2 in grouped_labels.index:
        # index/rows are the ground truth, cols are the predictions
        conf_matrix.loc[label1, label2] = grouped_labels.get((label1, label2), 0)

    return conf_matrix


def plot_precision_recall(plot_points: pd.DataFrame,
                          label_names=None,
                          local_path=None):
    """
    Plot precision recall curve for a given metric threshold

    :param plot_points: dict generated from calculate_precision_recall with all the points to plot by label and
     the entire dataset. keys include: confidence threshold, iou threshold, dataset levels precision, recall, and
     confidence, and label-level precision, recall and confidence
    :param label_names: list of label names to plot
    :param local_path: path to save plot
    :return:
    """

    ###################
    # plot by dataset #
    ###################

    plt.figure()
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    # plot each label separately
    dataset_points = plot_points[plot_points['data'] == 'dataset']

    plt.plot(dataset_points['recall'],
             dataset_points['precision'])

    # plot the dataset level
    plot_filename = f"dataset_precision_recall_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png"
    if local_path is None:
        save_path = os.path.join(os.getcwd(), '../.dataloop', plot_filename)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
    else:
        save_path = os.path.join(local_path, plot_filename)
        plt.savefig(save_path)
    plt.close()

    print(f'saved dataset precision recall plot to {save_path}')

    #################
    # plot by label #
    #################
    all_labels = plot_points[plot_points['data'] == 'label']

    if (label_names is None) or (bool(label_names) is False):
        label_names = all_labels['label_name'].copy().drop_duplicates()

    plt.figure()
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)
    plt.legend()

    # plot each label separately
    for label_name in label_names:
        label_points = all_labels[all_labels['label_name'] == label_name].copy()

        plt.plot(label_points['recall'],
                 label_points['precision'],
                 label=[label_name])

    # plot the dataset level
    plot_filename = f"label_precision_recall_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png"
    if local_path is None:
        save_path = os.path.join(os.getcwd(), '../.dataloop', plot_filename)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
    else:
        save_path = os.path.join(local_path, plot_filename)
        plt.savefig(save_path)
    plt.close()

    print(f'saved labels precision recall plot to {save_path}')
    return save_path


if __name__ == '__main__':
    dl.setenv('new-dev')
    if dl.token_expired():
        dl.login()

    dataset_id = '648f333a943352d180df011a'
    model_id = '649076c45a9c968a5c32ed65'

    plot_points = calc_precision_recall(dataset_id=dataset_id,
                                        model_id=model_id)
    plot_precision_recall(plot_points)

    metric = 'accuracy'
    conf_table = calc_confusion_matrix(dataset_id=dataset_id,
                                       model_id=model_id,
                                       metric=metric)
    print("columns are model predictions, rows are ground truth labels")
    print(conf_table)
    print()
