import dtlpy as dl
import numpy as np
from modules.scoring import measure_annotations


@dl.Package.decorators.module(name='scoring-and-metrics',
                              description='Scoring and metrics functions')
class ScoringAndMetrics(dl.BaseServiceRunner):
    """
    Scoring and metrics allows comparison between items, annotators, models, datasets, and tasks.

    """
    @staticmethod
    def get_and_calculate_consensus_items(consensus_task):
        # workaround for the task query returning all items, including hidden consensus clones
        filters = dl.Filters()
        filters.add(field='hidden', values=False)
        items = consensus_task.get_items(filters=filters).all()  # why is this "get_items" and not "list"?

        for item in items:
            # TODO is there a cleaner way to do this?
            if item.metadata['system']['refs'][0]['metadata']['status'] == 'consensus_done':
                item = ScoringAndMetrics.create_item_consensus_score(item)

        score_names = ['IOU', 'label', 'attribute']
        score_summary = {}

        for score_name in score_names:
            score_summary.update({score_name: []})
            feature_set = consensus_task.project.feature_sets.get(feature_set_name=f'Consensus {score_name}')

            for feature in feature_set.features.list().all():
                print(feature.value[0])
                score_summary[score_name].append(feature.value[0])

            print(f'Consensus average {score_name}: {np.mean(score_summary[score_name])}')

        return score_summary

    # @staticmethod
    # def get_and_calculate_dataset_items(consensus_task):
    #     annotators = []
    #     assignments = consensus_task.assignments.list()
    #     for assignment in assignments:
    #         annotators.append(assignment.annotator)
    #
    #     # workaround for the task query returning all items, including hidden consensus clones
    #     filters = dl.Filters()
    #     filters.add(field='hidden', values=False)
    #     items = consensus_task.get_items(filters=filters).all()  # why is this "get_items" and not "list"?
    #
    #     for item in items:
    #         # TODO is there a cleaner way to do this?
    #         if item.metadata['system']['refs'][0]['metadata']['status'] == 'consensus_done':
    #             item = ScoringAndMetrics.create_item_consensus_score(item)
    #
    #     return

    @staticmethod
    @dl.Package.decorators.function(display_name='Create item consensus score',
                                    inputs={"item": "Item"},
                                    outputs={"item": "Item"}
                                    )
    def create_item_consensus_score(item: dl.Item) -> dl.Item:
        metadata_list = item.metadata['system']['refs']
        for metadata_dict in metadata_list:
            if metadata_dict['type'] == 'task':
                task_id = metadata_dict['id']
                break

        consensus_task = dl.tasks.get(task_id=task_id)
        annotators = []
        assignments = consensus_task.assignments.list()
        for assignment in assignments:
            annotators.append(assignment.annotator)

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

                ScoringAndMetrics.create_scores_comparing_annotations(annot_collection_1=annot_collection_1,
                                                                      annot_collection_2=annot_collection_2,
                                                                      gt_is_first=False)

        return item

    @staticmethod
    def create_model_score(item_set_1, item_set_2=None, set_2_name=None):
        """
        Measures scores for a set of model predictions
        :param item_set_1: list of items
        :param item_set_2: (optional) if the second set of annotations are not from the same item
        :param set_2_name: the (model) name associated with the second set of annotations
        :return:
        """

        annot_set_1 = []
        annot_set_2 = []
        # if item_set_2 is None:
        #     # get the second set of annotations filtered from the same set of items
        #     for item in item_set_1:
        #         for a in item.annotations.list():
        #             if a.metadata['user']['model_name'] == set_2_name:
        #                 annot_set_2.append(a)
        #             if a.metadata['user'].get('model_name', None) is None:
        #                 annot_set_1.append(a)
        # else:
        #     for item in item_set_1:
        #         for a in item.annotations.list():
        #             if a.metadata['user'].get('model_name', None) is None:
        #                 annot_set_1.append(a)
        #     for item in item_set_2:
        #         # check that the annotation lists will match?
        #         for a in item.annotations.list():
        #             if a.metadata['user'].get('model_name', None) is None:
        #                 annot_set_2.append(a)

        pass

    @staticmethod
    @dl.Package.decorators.function(display_name='Compare annotations to score',
                                    inputs={"annot_collection_1": "List",
                                            "annot_collection_2": "List"},
                                    outputs={})
    def create_scores_comparing_annotations(annot_collection_1, annot_collection_2, gt_is_first=True,
                                            task_type=None):
        if task_type != 'compare projects':
            project = annot_collection_1[0].item.project
            dataset = annot_collection_1[0].item.dataset

        if task_type == 'consensus':
            feature_set_prefix = 'Consensus '

        score_names = ['IOU', 'label', 'attribute']
        results_columns = {'iou': 'geometry_score', 'label': 'label_score', 'attribute': 'attribute_score'}
        compare_type = dl.AnnotationType.BOX
        score_sets = {}

        for score_name in score_names:
            try:
                feature_set = project.feature_sets.get(feature_set_name=f'{feature_set_prefix}{score_name}')
            except dl.exceptions.NotFound:
                # create the feature set for each score type
                feature_set = project.feature_sets.create(name=f'{feature_set_prefix}{score_name}',
                                                          set_type='scores',
                                                          # refs require data type
                                                          data_type=dl.FeatureDataType.ANNOTATION_SCORE,
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
        return  # TODO: return some sort of summary? bc each pair of matched annotations will have 2 feature vectors

    @staticmethod
    def consensus_items_summary(feature_set):
        features = feature_set.features.list().all()
        # go through each feature, average the score
        pass

    @staticmethod
    def plot_precision_recall(annotation_scores, metric_threshold):
        import matplotlib.pyplot as plt
        from modules.lib.Evaluator import Evaluator
        from modules.lib.utils import MethodAveragePrecision

        #########################
        # plot precision/recall #
        #########################
        # calc
        method = MethodAveragePrecision.EveryPointInterpolation
        # method = MethodAveragePrecision.ElevenPointInterpolation
        plt.figure()
        for label in annotation_scores.label.unique():
            confidence_df = annotation_scores[annotation_scores.label == label]
            for model_name in confidence_df.model_name.unique():
                if model_name == 'gt':
                    continue
                model_confidence_df = confidence_df[annotation_scores.model_name == model_name]
                model_confidence_df.sort_values('confidence', inplace=True, ascending=True)
                true_positives = model_confidence_df.iou >= metric_threshold
                false_positives = model_confidence_df.iou < metric_threshold
                #
                num_gts = confidence_df[annotation_scores.model_name == 'gt'].shape[0]

                #
                acc_fps = np.cumsum(false_positives)
                acc_tps = np.cumsum(true_positives)
                recall = acc_tps / num_gts
                precision = np.divide(acc_tps, (acc_fps + acc_tps))
                # # Depending on the method, call the right implementation
                if method == MethodAveragePrecision.EveryPointInterpolation:
                    [avg_precis, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(recall, precision)
                else:
                    [avg_precis, mpre, mrec, _] = Evaluator.ElevenPointInterpolatedAP(recall, precision)
                # plt.plot(recall, precision, label=[label, model_name])
                plt.plot(mrec, mpre, label=[label, model_name])
        plt.legend()

        pass

    @staticmethod
    def get_scores_as_df(score: dict):
        # reconstruct the results from the score dict?

        pass


if __name__ == '__main__':
    scores = ScoringAndMetrics()
    # consensus_task = dl.tasks.get(task_name='pipeline consensus test (test tasks)')
    consensus_task = dl.tasks.get(task_id='644a307ae052f434dab98ff3')
    scores.get_and_calculate_consensus_items(consensus_task)
