# working script for checking the consensus functions

import dtlpy as dl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scoring  # former functions from dtlpy.ml


def plot_matrix(item_title, filename, matrix_to_plot, axis_labels, item=None, local_path=None):
    import os
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # annotators matrix plot, per item
    mask = np.zeros_like(matrix_to_plot, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

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

    if item is not None:
        newax = fig.add_axes((0.8, 0.8, 0.2, 0.2), anchor='NE', zorder=-1)

        im = plt.imread(item.download())
        newax.imshow(im)
        newax.axis('off')

    if local_path is None:
        save_path = os.path.join(root_dir, 'output', filename)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)
    else:
        plt.savefig(filename)
    plt.close()

    return True


def create_item_consensus_score(item: dl.Item, annotators: list):
    annotations = item.annotations.list()
    n_annotators = len(annotators)
    annots_by_annotator = {annotator: [] for annotator in annotators}
    # annotator_agreement = np.zeros((n_annotators, n_annotators))

    # group by some field (e.g. 'creator' or 'assignment id'), here we use annotator/creator
    for annotation in annotations:
        annotator = annotation.creator

        annots_by_annotator[annotator].append(annotation)

    # do pairwise comparisons of each annotator for all of their annotations on the item
    for i_annotator in range(n_annotators):
        for j_annotator in range(n_annotators):

            if i_annotator > j_annotator:
                annot_collection_1 = annots_by_annotator[annotators[i_annotator]]
                annot_collection_2 = annots_by_annotator[annotators[j_annotator]]

                create_scores(annot_collection_1=annot_collection_1,
                              annot_collection_2=annot_collection_2,
                              gt_is_first=False)

    return item


def create_scores(annot_collection_1, annot_collection_2, gt_is_first=True):
    project = annot_collection_1[0].item.project
    compare_type = dl.AnnotationType.BOX

    # compare bounding box annotations
    results = scoring.measure_annotations(
        annotations_set_one=annot_collection_1,
        annotations_set_two=annot_collection_2,
        compare_types=[compare_type],
        ignore_labels=False)

    # save each row of the results as a feature vector
    results_df = results[compare_type].to_df()
    results_df = results_df.where(pd.notnull(results_df), None)  # remove NaNs

    try:
        # feature_set = project.feature_sets.get(name='Annotation comparison')
        feature_set = project.feature_sets.get(feature_set_id='64465708025ac40ca5d2a6de')
    except dl.exceptions.NotFound:
        feature_set = project.feature_sets.create(name='Annotation comparison',
                                                  # set_type='annotation comparison: first annotation id, first creator, first label, first confidence, second id, second creator, second label, second confidence,annotation score, attribute score, geometry score, label score',
                                                  # set_type='blah blah, blah blah blah blah, blah',
                                                  set_type='blah blah, blah blah',
                                                  data_type=None,
                                                  entity_type=dl.FeatureEntityType.ITEM,
                                                  size=len(results_df))

    for index, row in results_df.iterrows():
        feature = feature_set.features.create(value=[list(row)],
                                              project_id=project.id,
                                              entity_id=row['item_id'])

    if gt_is_first:
        try:
            compare_annotations_iou = project.feature_sets.get(name='Annotations GT IOU')
        except dl.exceptions.NotFound:
            compare_annotations_iou = project.feature_sets.create(name='Annotation GT IOU',
                                                                  set_type='score',
                                                                  data_type=dl.FeatureDataType.ITEM_SCORE,
                                                                  entity_type=dl.FeatureEntityType.ANNOTATION,
                                                                  size=1)
    for annotation in annot_collection_2:
        compare_annotations_iou.features.create(value=[],
                                                project_id=project.id,
                                                entity_id=annotation.item.id,
                                                version='1.0.0')

    return results['total_mean_score']


def get_consensus_score(consensus_task, save_plot=False):
    annotators = []
    assignments = consensus_task.assignments.list()
    for assignment in assignments:
        annotators.append(assignment.annotator)
    n_annotators = len(annotators)

    # workaround for the task query returning all items, including hidden consensus clones
    filters = dl.Filters()
    filters.add(field='hidden', values=False)
    items = consensus_task.get_items(filters=filters).all()  # why is this "get_items" and not "list"?

    consensus_items_scores = {}
    # consensus_annotators_scores = []

    for item in items:
        # TODO is there a cleaner way to do this?
        if item.metadata['system']['refs'][0]['metadata']['status'] == 'consensus_done':
            item, score = create_item_consensus_score(item, annotators)
            consensus_items_scores.update({item.id: score})

    if save_plot:
        plot_matrix('consensus scores', 'consensus_scores.png', consensus_items_scores, annotators)

    return


if __name__ == '__main__':
    dl.setenv('prod')
    project = dl.projects.get('feature vectors')
    dataset = project.datasets.get('suim creatures')
    SAVE_PLOT = True

    consensus_task = dataset.tasks.get('check_consensus')  # 643be0e4bc2e4cb8b7c1a78d
    get_consensus_score(consensus_task, save_plot=SAVE_PLOT)
