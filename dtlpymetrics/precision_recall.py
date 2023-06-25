import os
import numpy as np
import pandas as pd
import dtlpy as dl
import matplotlib.pyplot as plt

# from typing import Dict, Union, List, Any

def calc_precision_recall(dataset_id: str,
                          model_id: str,
                          conf_threshold=0.5,
                          iou_threshold=0.5):
    # method_type='every_point'):
    """
    Plot precision recall curve for a given metric threshold

    :param dataset_id: str dataset ID
    :param model_id: str model ID
    :param conf_threshold: 
    :param iou_threshold:
    :return:
    """

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

    ##############################
    # calculate precision/recall #
    #############################
    # calc
    if labels is None:
        labels = pd.concat([scores.first_label, scores.second_label]).dropna()

    plot_points = {'conf_threshold': conf_threshold,
                   'iou_threshold': iou_threshold,
                   'labels': {},
                   'precision': [],
                   'recall': []
                   }

    num_gts = sum(scores.first_id.notna())
    detections = scores[scores.second_id.notna()].copy()
    passed_detections = detections[detections['second_confidence'] > conf_threshold].copy()

    passed_detections.sort_values('second_confidence', inplace=True, ascending=False, ignore_index=True)
    passed_detections['true_positives'] = passed_detections['geometry_score'] >= iou_threshold
    passed_detections['false_positives'] = passed_detections['geometry_score'] < iou_threshold

    # get dataset-level precision/recall
    dataset_fps = np.cumsum(passed_detections['false_positives'])
    dataset_tps = np.cumsum(passed_detections['true_positives'])
    dataset_recall = dataset_tps / num_gts
    dataset_precision = np.divide(dataset_tps, (dataset_fps + dataset_tps))

    [_,
     dataset_plot_precision,
     dataset_plot_recall,
     dataset_plot_score] = every_point_curve(dataset_recall,
                                             dataset_precision,
                                             passed_detections[
                                                 'second_confidence'])
    plot_points['precision'] = dataset_plot_precision
    plot_points['recall'] = dataset_plot_recall
    plot_points['score'] = dataset_plot_score

    # get label-level precision/recall
    for label in list(set(label_names)):
        label_positives = passed_detections[passed_detections.first_label == label].copy()
        if label_positives.shape[0] == 0:
            label_plot_precision = [0]
            label_plot_recall = [0]
            label_plot_score = [0]
        else:
            label_positives.sort_values('second_confidence', inplace=True, ascending=False, ignore_index=True)

            label_fps = np.cumsum(label_positives['false_positives'])
            label_tps = np.cumsum(label_positives['true_positives'])
            label_recall = label_tps / num_gts
            label_precision = np.divide(label_tps, (label_fps + label_tps))

            [_,
             label_plot_precision,
             label_plot_recall,
             label_plot_score] = every_point_curve(label_recall,
                                                   label_precision,
                                                   label_positives['second_confidence'])

        plot_points['labels'].update({label: {
            'precision': label_plot_precision,
            'recall': label_plot_recall,
            'score': label_plot_score}})

    return plot_points


def every_point_curve(recall, precision, score):
    """
    Calculate precision-recall curve from a list of precision & recall values
    :param precision: list of precision values
    :param recall: list of recall values
    :param score:
    :return:
    """
    recall_points = np.concatenate([[0], recall, [1]])
    precision_points = np.concatenate([[0], precision, [1]])
    score_points = np.concatenate([[score.iloc[0]], score, [score.iloc[-1]]])

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
            score_points[0:len(precision_points) - 1]]


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
        labels = pd.concat([scores.first_label, scores.second_label]).dropna()

    scores_cleaned = scores.dropna().reset_index(drop=True)
    scores_labels = scores_cleaned[['first_label', 'second_label']]
    grouped_labels = scores_labels.groupby(['first_label', 'second_label']).size()

    conf_matrix = pd.DataFrame(index=label_names, columns=label_names, data=0)
    for label1, label2 in grouped_labels.index:
        # index/rows are the ground truth, cols are the predictions
        conf_matrix.loc[label1, label2] = grouped_labels.get((label1, label2), 0)

    return conf_matrix


def plot_precision_recall(plot_points: dict, local_path=None):
    """
    Plot precision recall curve for a given metric threshold

    :param plot_points: dictionary of precision/recall points
    :param local_path: path to save plot
    :return:
    """

    #################
    # plot by label #
    #################
    labels = list(plot_points['labels'].keys())

    plt.figure()
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    labels_df = pd.DataFrame.from_dict(plot_points['labels'], orient='index')
    for label in labels:
        plt.plot(plot_points['labels'][label]['recall'],
                 plot_points['labels'][label]['precision'],
                 label=[label])
    plt.legend()

    # plot_filename = f'precision_recall_{dataset_id}_{model_id}_{plot_points[metric]}_{metric_threshold}.png'
    plot_filename_by_labels = f'precision_recall_by_label.png'
    if local_path is None:
        save_path = os.path.join(os.getcwd(), '.dataloop', plot_filename_by_labels)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
    else:
        save_path = os.path.join(local_path, plot_filename_by_labels)
        plt.savefig(save_path)

    plt.close()

    print(f'saved precision recall plot to {save_path}')

    plot_data = pd.DataFrame({k: plot_points[k] for k in ('precision', 'recall', 'score')})
    plot_data.to_csv(os.path.join(os.path.dirname(save_path), 'precision_recall.csv'))

    return save_path


if __name__ == '__main__':
    dl.setenv('new-dev')

    dataset_id = '648f333a943352d180df011a'
    model_id = '649076c45a9c968a5c32ed65'

    plot_points = calc_precision_recall(dataset_id=dataset_id,
                                        model_id=model_id,
                                        conf_threshold=0.2)
    plot_precision_recall(plot_points)

    # metric = 'accuracy'
    # conf_table = calc_confusion_matrix(dataset_id=dataset_id,
    #                                    model_id=model_id,
    #                                    metric=metric)
    # print("columns are model predictions, rows are ground truth labels")
    # print(conf_table)
    print()
