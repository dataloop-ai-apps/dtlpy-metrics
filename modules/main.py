import dtlpy as dl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scoring import measure_annotations
from dtlpy.ml import metrics, predictions_utils
from pathlib import Path

from dtlpy import entities, utilities


@dl.Package.defs.module(name='scoring-and-metrics',
                        description='Scoring and metrics functions',
                        init_inputs={'task': entities.Task,
                                     'save_plot': bool}, )
class ScoringAndMetrics(dl.BaseServiceRunner):
    @staticmethod
    def get_and_calculate_consensus_items(consensus_task):
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
                item = ScoringAndMetrics.create_item_consensus_score(item, annotators)

        return

    @staticmethod
    def create_item_consensus_score(item: dl.Item, annotators: list):
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

                ScoringAndMetrics.create_annotation_scores(annot_collection_1=annot_collection_1,
                                                           annot_collection_2=annot_collection_2,
                                                           gt_is_first=False)

        return item

    @staticmethod
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
        results = measure_annotations(
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
