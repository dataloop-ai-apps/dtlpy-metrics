# doesnt work

import os
import dtlpy as dl
import numpy as np

scorer = dl.AppModule(name='Scoring and metrics function',
                      description=''
                                  'Models can be evaluated with a model entity and '
                                  'configured dataset to compare predictions to ground '
                                  'truth annotations.')

score_names = ['IOU', 'label', 'attribute']
results_columns = {'iou': 'geometry_score', 'label': 'label_score', 'attribute': 'attribute_score'}


@scorer.set_init()
def setup():
    pass


@staticmethod
@scorer.add_function(display_name='Consensus scoring',
                     inputs={'consensus_task': dl.Task},
                     outputs={'score_summary': dict}
                     )
def calculate_consensus_items(consensus_task: dl.Task):
    """
    Calculate consensus scores for all items in a consensus task

    :param consensus_task:
    :return:
    """
    # workaround for the task query returning all items, including hidden consensus clones
    filters = dl.Filters()
    filters.add(field='hidden', values=False)
    items = consensus_task.get_items(filters=filters).all()  # why is this "get_items" and not "list"?

    for item in items:
        if item.metadata['system']['refs'][0]['metadata']['status'] == 'consensus_done':
            item = scorer.create_item_consensus_score(item)

    score_summary = {}

    for score_name in score_names:
        score_summary.update({score_name: []})
        feature_set = consensus_task.project.feature_sets.get(feature_set_name=f'Consensus {score_name}')

        for feature in feature_set.features.list().all():
            print(feature.value[0])
            score_summary[score_name].append(feature.value[0])

        print(f'Consensus average {score_name}: {np.mean(score_summary[score_name])}')

    return score_summary


@staticmethod
@scorer.add_function(display_name='Create item consensus score',
                     inputs={"item": "Item"},
                     outputs={"item": "Item"}
                     )
def create_item_consensus_score(item: dl.Item) -> dl.Item:
    ################################
    # find task ID to get the task #
    ################################

    metadata_list = item.metadata['system']['refs']
    for metadata_dict in metadata_list:
        if metadata_dict['type'] == 'task':
            task_id = metadata_dict['id']
            break

    try:
        consensus_task = dl.tasks.get(task_id=task_id)
    except dl.exceptions.NotFound:
        raise dl.exceptions.NotFound('Consensus task not found')

    ##################################
    # collect annotators to group by #
    ##################################
    annotators = []
    assignments = consensus_task.assignments.list()
    for assignment in assignments:
        annotators.append(assignment.annotator)

    #################################
    # sort annotations by annotator #
    #################################
    annotations = item.annotations.list()
    n_annotators = len(annotators)
    annots_by_annotator = {annotator: [] for annotator in annotators}

    # group by some field (e.g. 'creator' or 'assignment id'), here we use annotator/creator
    for annotation in annotations:
        annots_by_annotator[annotation.creator].append(annotation)

    # do pairwise comparisons of each annotator for all of their annotations on the item
    for i_annotator in range(n_annotators):
        for j_annotator in range(0, i_annotator + 1):
            annot_collection_1 = annots_by_annotator[annotators[i_annotator]]
            annot_collection_2 = annots_by_annotator[annotators[j_annotator]]

            scorer.create_annotation_scores(annot_collection_1=annot_collection_1,
                                            annot_collection_2=annot_collection_2,
                                            gt_is_first=False)

    return item
