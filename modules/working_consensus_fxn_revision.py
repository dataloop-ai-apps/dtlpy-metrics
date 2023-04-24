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
        for j_annotator in range(0, i_annotator + 1):
            annot_collection_1 = annots_by_annotator[annotators[i_annotator]]
            annot_collection_2 = annots_by_annotator[annotators[j_annotator]]

            create_annotation_scores(annot_collection_1=annot_collection_1,
                                     annot_collection_2=annot_collection_2,
                                     gt_is_first=False)

    return item


def create_annotation_scores(annot_collection_1, annot_collection_2, gt_is_first=True):
    project = annot_collection_1[0].item.project
    dataset = annot_collection_1[0].item.dataset
    score_names = ['IOU', 'label', 'attribute']
    results_columns = {'iou': 'geometry_score', 'label': 'label_score', 'attribute': 'attribute_score'}
    compare_type = dl.AnnotationType.BOX
    score_sets = {}

    for score_name in score_names:
        try:
            feature_set = project.feature_sets.get(feature_set_name=f'Consensus {score_name}')
        except dl.exceptions.NotFound:
            # create the feature set for each score type
            feature_set = project.feature_sets.create(name=f'Consensus {score_name}',
                                                      set_type='scores',
                                                      data_type=dl.FeatureDataType.ANNOTATION_SCORE,
                                                      # refs require data type
                                                      entity_type=dl.FeatureEntityType.ANNOTATION,
                                                      size=1)
        score_sets.update({score_name: feature_set})

    # compare bounding box annotations
    results = scoring.measure_annotations(
        annotations_set_one=annot_collection_1,
        annotations_set_two=annot_collection_2,
        compare_types=[compare_type],
        ignore_labels=False)

    results_df = results[compare_type].to_df()
    for i, row in results_df.iterrows():
        for score, feature_set in score_sets.items():
            if not gt_is_first:
                if row['first_id'] is not None:
                    feature1 = feature_set.features.create(value=[row[results_columns[score.lower()]]],
                                                           project_id=project.id,
                                                           entity_id=row['first_id'],
                                                           refs={'item': row['item_id'],
                                                                 'annotator': row['first_creator'],
                                                                 'dataset': dataset.id,
                                                                 'relative': row['second_id'],
                                                                 })
            if row['second_id'] is not None:
                feature2 = feature_set.features.create(value=[row[results_columns[score.lower()]]],
                                                       project_id=project.id,
                                                       entity_id=row['second_id'],
                                                       refs={'item': row['item_id'],
                                                             'annotator': row['second_creator'],
                                                             'dataset': dataset.id,
                                                             'relative': row['first_id'],
                                                             })
    return True


def calculate_consensus_score(consensus_task):
    annotators = []
    assignments = consensus_task.assignments.list()
    for assignment in assignments:
        annotators.append(assignment.annotator)

    # workaround for the task query returning all items, including hidden consensus clones
    filters = dl.Filters()
    filters.add(field='hidden', values=False)
    items = consensus_task.get_items(filters=filters).all()  # why is this "get_items" and not "list"?

    for item in items:
        # TODO is there a cleaner way to do this?
        if item.metadata['system']['refs'][0]['metadata']['status'] == 'consensus_done':
            item = create_item_consensus_score(item, annotators)

    return


if __name__ == '__main__':
    dl.setenv('prod')
    project = dl.projects.get('feature vectors')
    dataset = project.datasets.get('suim creatures')

    # clean up previous feature sets by the same name
    fsets = project.feature_sets.list()
    for fset in fsets:
        if 'Consensus' in fset.name:
            fset.delete()
            print(f'{fset.name} deleted')

    consensus_task = dataset.tasks.get('check_consensus')  # 643be0e4bc2e4cb8b7c1a78d
    calculate_consensus_score(consensus_task)
