import dtlpy as dl
import numpy as np
import pandas as pd

from dtlpymetrics.metrics_utils import measure_annotations

score_names = ['IOU', 'label', 'attribute']
results_columns = {'iou': 'geometry_score', 'label': 'label_score', 'attribute': 'attribute_score'}
compare_type = dl.AnnotationType.BOX


@dl.Package.decorators.module(name='scoring-and-metrics',
                              description='Scoring and metrics functions')
class ScoringAndMetrics(dl.BaseServiceRunner):
    """
    Scoring and metrics allows comparison between items, annotators, models, datasets, and tasks.

    """

    @staticmethod
    def calculate_consensus_items(consensus_task: dl.Task):
        # workaround for the task query returning all items, including hidden consensus clones
        filters = dl.Filters()
        filters.add(field='hidden', values=False)
        items = consensus_task.get_items(filters=filters).all()  # why is this "get_items" and not "list"?

        for item in items:
            # TODO is there a cleaner way to do this?
            if item.metadata['system']['refs'][0]['metadata']['status'] == 'consensus_done':
                item = ScoringAndMetrics.create_item_consensus_score(item)

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
    def calculate_dataset_scores(dataset1: dl.Dataset, dataset2: dl.Dataset):
        # annotators = []
        # assignments = consensus_task.assignments.list()
        # for assignment in assignments:
        #     annotators.append(assignment.annotator)
        #
        # # workaround for the task query returning all items, including hidden consensus clones
        # filters = dl.Filters()
        # filters.add(field='hidden', values=False)
        # items = consensus_task.get_items(filters=filters).all()  # why is this "get_items" and not "list"?
        #
        # for item in items:
        #     # TODO is there a cleaner way to do this?
        #     if item.metadata['system']['refs'][0]['metadata']['status'] == 'consensus_done':
        #         item = ScoringAndMetrics.create_item_consensus_score(item)

        # return featureset id?
        return dataset1, dataset2

    @staticmethod
    @dl.Package.decorators.function(display_name='Create item consensus score',
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

        consensus_task = dl.tasks.get(task_id=task_id)

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

                ScoringAndMetrics.create_annotation_scores(annot_collection_1=annot_collection_1,
                                                           annot_collection_2=annot_collection_2,
                                                           gt_is_first=False)

        return item

    @staticmethod
    def create_model_score(item_set: list,
                           model_name: str = None,
                           ignore_labels=False):
        """
        Measures scores for a set of model predictions compared against ground truth annotations.

        :param item_set: list of items. When two sets are provided, this is the ground truth set
        :param model_name: Model name associated with the non-GT annotations
        :param ignore_labels: bool, True means every annotation will be cross-compared regardless of label
        :return:
        """

        project = item_set[0].project
        dataset = item_set[0].dataset
        tags = ['model evaluation', model_name]

        annot_set_1 = []
        annot_set_2 = []

        if model_name is None:
            return False, 'No model name found for the second set of annotations, please provide model name.'

        #################################
        # list of item annotation lists #
        #################################
        for item in item_set:
            item_annots_1 = []
            item_annots_2 = []
            for annotation in item.annotations.list():
                if annotation.metadata['user'].get('model', None) is None:
                    item_annots_1.append(annotation)
                elif annotation.metadata['user']['model']['name'] == model_name:
                    item_annots_2.append(annotation)
            annot_set_1.append(item_annots_1)
            annot_set_2.append(item_annots_2)

        if not item_annots_1:
            return False, 'No ground truth annotations found. Please ensure there are annotations on the items.'
        if not item_annots_2:
            return False, 'No model annotations found. Please ensure there are model annotations on the items.'

        #########################################################
        # Compare annotations and return concatenated dataframe #
        #########################################################
        all_results = pd.DataFrame()
        for i in range(len(item_set)):
            # compare annotations for each item
            results = measure_annotations(annotations_set_one=annot_set_1[i],
                                          annotations_set_two=annot_set_2[i],
                                          ignore_labels=ignore_labels)
            results_df = results[compare_type].to_df()

            all_results = pd.concat([all_results, results_df],
                                    ignore_index=True)

        ###################################################
        # Create/get feature sets for IOU/label/attribute #
        ###################################################
        feature_set_prefix = 'Model '  # + model_name + ' '
        score_sets = {}
        for score_name in score_names:
            try:
                feature_set = project.feature_sets.get(feature_set_name=f'{feature_set_prefix}{score_name}')
                # clean up previous scores from this model if they exist
                for feature in feature_set.features.list().all():
                    feature.delete()
            except dl.exceptions.NotFound:
                # create the feature set for each score type
                feature_set = project.feature_sets.create(name=f'{feature_set_prefix}{score_name}',
                                                          set_type='model',
                                                          # refs require data type
                                                          data_type=dl.FeatureDataType.ANNOTATION_SCORE,
                                                          entity_type=dl.FeatureEntityType.ANNOTATION,
                                                          size=1,
                                                          tags=tags)
            score_sets.update({score_name: feature_set})

        ####################################################
        # iterate through rows to create pairs of features #
        ####################################################
        for i, row in all_results.iterrows():
            for score, feature_set in score_sets.items():
                # save only true positives and false positives
                if row['second_id'] is not None:
                    feature_set.features.create(value=[row[results_columns[score.lower()]]],
                                                project_id=project.id,
                                                entity_id=row['second_id'],
                                                refs={'item': row['item_id'],
                                                      'dataset': dataset.id,
                                                      'relative': row['first_id'],
                                                      })

        return True, 'Successfully created model scores.'

    @staticmethod
    @dl.Package.decorators.function(display_name='Compare annotations to score',
                                    inputs={"annot_collection_1": "List",
                                            "annot_collection_2": "List"},
                                    outputs={})
    def create_annotation_scores(annot_collection_1,
                                 annot_collection_2,
                                 gt_is_first=True,
                                 task_type=None) -> dict:
        """

        :param annot_collection_1:
        :param annot_collection_2:
        :param gt_is_first:
        :param task_type:
        :return: dict of feature sets, indexed by the type of score (e.g. IOU)
        """
        if task_type != 'compare projects':
            project = annot_collection_1[0].item.project
            dataset = annot_collection_1[0].item.dataset

        if task_type == 'consensus':
            feature_set_prefix = 'Consensus '

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
        return score_sets  # TODO: return some sort of summary? bc each pair of matched annotations will have 2 feature vectors

    @staticmethod
    def consensus_items_summary(feature_set):
        # TODO should receive items, and query by that, not by feature set

        # go through each feature, average the score
        features = feature_set.features.list().all()
        return np.mean([feature.value for feature in features])

    @staticmethod
    def plot_precision_recall(annotation_scores, metric_threshold):
        """
        Plot precision recall curve for a given metric threshold

        :param annotation_scores:
        :param metric_threshold:
        :return:
        """
        import matplotlib.pyplot as plt
        from dtlpymetrics.lib.Evaluator import Evaluator
        from dtlpymetrics.lib.utils import MethodAveragePrecision

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

        return

    @staticmethod
    def get_scores_as_df(score: dict):
        # reconstruct the results from the score dict?

        pass

    @staticmethod
    def item_scores_to_df(items_list: list, feature_set: dl.FeatureSet):
        scores_df = pd.DataFrame()
        score_type = feature_set.name.split(' ')[-1]

        for item in items_list:
            item_scores = pd.DataFrame()
            # item = dl.items.get(None, '6453bc227edd9a6a11237ab0')
            filters = dl.Filters(use_defaults=False,
                                 custom_filter={
                                     '$and':
                                         [
                                             {'refs': {'item': item.id}},
                                             {'featureSetId': feature_set.id},
                                             {'dataType': 'annotationScore'}
                                         ]
                                 },
                                 resource=dl.FiltersResource.FEATURE
                                 )
            # dl.features.list(filters=filters).print()
            features = dl.features.list(filters=filters).all()
            for feature in features:
                feature_dict = {"annotation id": feature.entity_id,
                                "item id": feature.refs['item'],
                                "score": score_type,
                                "value": feature.value[0],
                                "GT annotation id": feature.refs['relative'],
                                }
                item_scores = pd.concat([item_scores, pd.DataFrame(feature_dict, index=[0])])

            scores_df = pd.concat([scores_df, item_scores],
                                  ignore_index=True)

        return scores_df


if __name__ == '__main__':
    # ########################
    # # Consensus task test ##
    # ########################
    # scores = ScoringAndMetrics()
    # # consensus_task = dl.tasks.get(task_name='pipeline consensus test (test tasks)')
    # consensus_task = dl.tasks.get(task_id='644a307ae052f434dab98ff3')
    # scores.calculate_consensus_items(consensus_task)

    ####################
    # Model evaluation #
    ####################
    dl.setenv('rc')
    project = dl.projects.get('Model mgmt demo')
    # dataset = dl.datasets.get(None, '6450c91e200a21ed641a332a') # active learning
    dataset = project.datasets.get('test model eval')
    items_list = list(dataset.items.list().all())

    ########################################
    # evaluate the model and create scores #
    ########################################
    success, message = ScoringAndMetrics.create_model_score(item_set=items_list,
                                                            model_name='cloned_yolo_demo')
    # ignore_labels=True)
    print(message)

    fset = project.feature_sets.get('Model IOU')
    ScoringAndMetrics.item_scores_to_df(items_list, fset)
