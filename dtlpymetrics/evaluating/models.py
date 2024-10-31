import datetime
import logging
import os
import dtlpy as dl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import List

from ..dtlpy_scores import Score, ScoreType
from ..utils import plot_confusion_matrix

dl.use_attributes_2()
logger = logging.getLogger('scoring-and-metrics')


def confusion_matrix(dataset_id: str,
                     model_id: str,
                     metric: str,
                     show_unmatched=True) -> pd.DataFrame:
    """
    Calculate confusion matrix for a given model and metric (i.e. IOU, accuracy)

    :param dataset_id: str ID of test dataset
    :param model_id: str ID of model
    :param metric: name of the metric for comparing
    :param show_unmatched: display extra column showing which GT annotations were not matched
    :return: DataFrame with confusion matrix
    """
    if metric.lower() == 'iou':
        metric = 'geometry_score'
    elif metric.lower() == 'accuracy':
        metric = 'label_score'

    # TODO retrieve scores directly once available
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

    ###############################
    # create table of comparisons #
    ###############################
    if label_names is None:
        label_names = pd.concat([scores.first_label, scores.second_label]).dropna()

    scores_cleaned = scores.dropna().reset_index(drop=True)
    scores_labels = scores_cleaned[['first_label', 'second_label']]
    grouped_labels = scores_labels.groupby(['first_label', 'second_label']).size()

    conf_matrix = pd.DataFrame(index=label_names, columns=label_names, data=0)
    for label1, label2 in grouped_labels.index:
        # index/rows are the ground truth, cols are the predictions
        conf_matrix.loc[label1, label2] = grouped_labels.get((label1, label2), 0)

    return conf_matrix


def label_confusion_matrix(item: dl.Item,
                           scores: List[Score],
                           save_plot=True) -> pd.DataFrame:
    """
    Calculate confusion matrix from a set of label confusion scores
    :param item: dl.Item
    :param scores: list of scores
    :param save_plot: bool
    :return: confusion matrix as pd.DataFrame
    """
    scores_dl = []
    for score in scores:
        scores_dl.append(score)

    # ###############################
    # # create table of comparisons #
    # ###############################
    label_names = []
    for score in scores_dl:
        if score.type == ScoreType.LABEL_CONFUSION:
            if score.entity_id not in label_names:
                label_names.append(score.entity_id)
            if score.relative not in label_names:
                label_names.append(score.relative)

    conf_matrix = pd.DataFrame(index=label_names, columns=label_names)

    for score in scores_dl:
        if score.type == ScoreType.LABEL_CONFUSION:
            conf_matrix.loc[score.entity_id] = score.value
            conf_matrix.loc[score.entity_id, score.relative] = score.value

    conf_matrix.fillna(0, inplace=True)
    conf_matrix.rename(columns={None: 'unmatched'}, inplace=True)
    conf_matrix.rename(index={None: 'unmatched'}, inplace=True)
    label_names = ['unmatched' if label is None else label for label in label_names]

    if save_plot is True:
        if os.environ.get('SCORES_DEBUG_PATH', None) is not None:
            debug_path = os.environ.get('SCORES_DEBUG_PATH', None)

            plot_confusion_matrix(item_title=f'label confusion matrix {item.id}',
                                  filename=os.path.join(debug_path, 'label_confusion',
                                                        f'label_confusion_matrix_{item.id}.png'),
                                  matrix_to_plot=conf_matrix,
                                  axis_labels=label_names)

        else:
            plot_confusion_matrix(item_title=f'label confusion matrix {item.id}',
                                  filename=os.path.join('.dataloop', 'label_confusion',
                                                        f'label_confusion_matrix_{item.id}.png'),
                                  matrix_to_plot=conf_matrix,
                                  axis_labels=label_names)

    return conf_matrix


def get_model_scores_df(dataset: dl.Dataset, model: dl.Model) -> pd.DataFrame:
    """
    Retrieves the dataframe for all the scores for a given model on a dataset via a hidden csv file.
    :param dataset: Dataset where the model was evaluated
    :param model: Model entity
    :return: matched_annots_df: dataframe of all annotations in ground truth and model predictions
    """
    file_name = f'{model.id}.csv'
    local_path = os.path.join(os.getcwd(), '.dataloop', file_name)
    filters = dl.Filters(field='name', values=file_name)
    filters.add(field='hidden', values=True)
    pages = dataset.items.list(filters=filters)

    if pages.items_count > 0:
        for item in pages.all():
            item.download(local_path=local_path)
    else:
        raise ValueError(
            f'No matched annotations file found for model {model.id} on dataset {dataset.id}. Please evaluate model on the dataset first.')

    model_scores_df = pd.read_csv(local_path)
    return model_scores_df


def get_false_negatives(model: dl.Model, dataset: dl.Dataset) -> pd.DataFrame:
    """
    Retrieves the dataframe for all the scores for a given model on a dataset via a hidden csv file,
    and returns a dataframe with the properties of all the false negatives.
    :param model: Model entity
    :param dataset: Dataset where the model was evaluated
    :return: DataFrame with all the false negatives
    """
    file_name = f'{model.id}.csv'
    local_path = os.path.join(os.getcwd(), '../.dataloop', file_name)
    filters = dl.Filters(field='name', values=file_name)
    filters.add(field='hidden', values=True)
    pages = dataset.items.list(filters=filters)

    if pages.items_count > 0:
        for item in pages.all():
            item.download(local_path=local_path)
    else:
        raise ValueError(
            f'No scores file found for model {model.id} on dataset {dataset.id}. Please evaluate model on the dataset first.')

    scores_df = pd.read_csv(local_path)

    ########################
    # list false negatives #
    ########################
    model_fns = dict()
    annotation_to_item_map = {ann_id: item_id for ann_id, item_id in
                              zip(scores_df.first_id, scores_df.itemId)}
    fn_annotation_ids = scores_df[scores_df.second_id.isna()].first_id
    print(f'model: {model.name} with {len(fn_annotation_ids)} false negative')
    fn_items_ids = np.unique([annotation_to_item_map[ann_id] for ann_id in fn_annotation_ids])
    for i_id in fn_items_ids:
        if i_id not in model_fns:
            i_id: dl.Item
            url = dl.client_api._get_resource_url(
                "projects/{}/datasets/{}/items/{}".format(dataset.project.id, dataset.id, i_id))
            model_fns[i_id] = {'itemId': i_id,
                               'url': url}
        model_fns[i_id].update({model.name: True})

    model_fn_df = pd.DataFrame(model_fns.values()).fillna(False)
    model_fn_df.to_csv(os.path.join(os.getcwd(), f'{model.name}_false_negatives.csv'))

    return model_fn_df


def plot_precision_recall(plot_points: pd.DataFrame,
                          dataset_name=None,
                          label_names=None,
                          local_path=None):
    """
    Plot precision recall curve for a given metric threshold

    :param plot_points: dict generated from calculate_precision_recall with all the points to plot by label and
     the entire dataset. keys include: confidence threshold, iou threshold, dataset levels precision, recall, and
     confidence, and label-level precision, recall and confidence
    :param dataset_name: name of dataset to plot in legend
    :param label_names: list of label names to plot
    :param local_path: path to save plot
    :return: directory path where plots are saved
    """
    if local_path is None:
        root_dir = os.getcwd().split('dtlpymetrics')[0]
        save_dir = os.path.join(root_dir, 'dtlpymetrics', '../.dataloop')
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))
    else:
        save_dir = os.path.join(local_path)

    ###################
    # plot by dataset #
    ###################
    logger.info('Plotting precision recall')

    plt.figure()
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    # plot each label separately
    dataset_points = plot_points[plot_points['data'] == 'dataset']
    dataset_legend = f"{dataset_points['dataset_id'].iloc[0]}" if dataset_name is None else dataset_name

    plt.plot(dataset_points['recall'],
             dataset_points['precision'],
             label=dataset_legend)

    plt.legend(loc='upper right')

    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.grid()

    # plot the dataset level
    plot_filename = f"dataset_precision_recall_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png"
    save_path = os.path.join(save_dir, plot_filename)
    plt.savefig(save_path)
    # plt.close()
    logger.info(f'Saved dataset precision recall plot to {save_path}')

    #################
    # plot by label #
    #################
    all_labels = plot_points[plot_points['data'] == 'label']

    if (label_names is None) or (bool(label_names) is False):
        label_names = all_labels['label_name'].copy().drop_duplicates()

    plt.figure()
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    # plot each label separately
    for label_name in label_names:
        label_points = all_labels[all_labels['label_name'] == label_name].copy()

        plt.plot(label_points['recall'],
                 label_points['precision'],
                 label=label_name)

    plt.legend(loc='upper right')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.grid()

    # plot the dataset level
    plot_filename = f"label_precision_recall_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png"
    save_path = os.path.join(save_dir, plot_filename)
    plt.savefig(save_path)
    # plt.close()
    logger.info(f'Saved labels precision recall plot to {save_path}')

    return save_dir


def plot_annotators_matrix(item_title, filename, matrix_to_plot, axis_labels):
    """
    Plot confusion matrix between annotator pairs
    :param item_title: title of the item
    :param filename: path to save plot
    :param matrix_to_plot: confusion matrix
    :param axis_labels: list of labels for the axis
    :return filename: path to saved plot
    """
    # annotators matrix plot, per item
    mask = np.zeros_like(matrix_to_plot, dtype=bool)
    # mask[np.triu_indices_from(mask)] = True

    sns.set(rc={'figure.figsize': (20, 10),
                'axes.facecolor': 'white'},
            font_scale=2)
    sns_plot = sns.heatmap(matrix_to_plot,
                           annot=True,
                           mask=mask,
                           cmap='Blues',
                           xticklabels=axis_labels,
                           yticklabels=axis_labels,
                           vmin=0,
                           vmax=1)
    sns_plot.set(title=item_title)
    sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation=270)

    fig = sns_plot.get_figure()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename)
    plt.close()

    return filename
