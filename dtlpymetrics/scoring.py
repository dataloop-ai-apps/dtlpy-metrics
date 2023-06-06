import os
import dtlpy as dl
import numpy as np
import pandas as pd

from dtlpymetrics.metrics_utils import measure_annotations

score_names = ['IOU', 'label', 'attribute']
results_columns = {'iou': 'geometry_score', 'label': 'label_score', 'attribute': 'attribute_score'}
all_compare_types = [dl.AnnotationType.BOX,
                     dl.AnnotationType.CLASSIFICATION,
                     dl.AnnotationType.POLYGON,
                     dl.AnnotationType.POINT,
                     dl.AnnotationType.SEGMENTATION]


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
    def calculate_annotator_scores():
        ### Call create_annotation_scores, get the feature set, and then group by annotator
        # this requires being able to query scores by annotator, which is currently not supported by R&D

        return

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
    def create_model_score(dataset: dl.Dataset = None,
                           filters: dl.Filters = None,
                           model: dl.Model = None,
                           match_threshold: float = 0.5,
                           ignore_labels=False,
                           compare_types=None):
        """
        Measures scores for a set of model predictions compared against ground truth annotations.

        :param dataset: Dataset associated with the ground truth annotations
        :param filters: DQL Filter for retrieving the test items
        :param model: Model for evaluating predictions
        :param match_threshold: IoU threshold to count as a match
        :param ignore_labels: bool, True means every annotation will be cross-compared regardless of label
        :param compare_types: annotation types to compare
        :return:
        """

        if dataset is None:
            return False, 'No dataset provided, please provide a dataset.'
        if model is None:
            return False, 'No model provided, please provide a model.'
        if filters is None:
            items_list = list(dataset.items.list().all())
        else:
            items_list = list(dataset.items.list(filters=filters).all())
        if compare_types is None:
            compare_types = all_compare_types
        if not isinstance(compare_types, list):
            compare_types = [compare_types]

        annot_set_1 = []
        annot_set_2 = []

        if model.name is None:
            return False, 'No model name found for the second set of annotations, please provide model name.'
        if not items_list:
            return False, 'No items found in the dataset, please check the dataset and filters.'

        #################################
        # list of item annotation lists #
        #################################
        for item in items_list:
            item_annots_1 = []
            item_annots_2 = []
            for annotation in item.annotations.list():
                if annotation.metadata.get('user', {}).get('model') is None:
                    item_annots_1.append(annotation)
                elif annotation.metadata['user']['model']['name'] == model.name:
                    item_annots_2.append(annotation)
            annot_set_1.append(item_annots_1)
            annot_set_2.append(item_annots_2)

        if not item_annots_1:
            return False, 'No ground truth annotations found. Please ensure there are annotations on the items.'

        #########################################################
        # Compare annotations and return concatenated dataframe #
        #########################################################
        all_results = pd.DataFrame()
        for i in range(len(items_list)):
            # compare annotations for each item
            # print(f'item {i}: GT annots {len(annot_set_1[i])}, model annots {len(annot_set_2[i])}')
            results = measure_annotations(annotations_set_one=annot_set_1[i],
                                          annotations_set_two=annot_set_2[i],
                                          match_threshold=match_threshold,
                                          ignore_labels=ignore_labels,
                                          compare_types=compare_types)
            for compare_type in compare_types:
                results_df = results[compare_type].to_df()
                results_df['item_id'] = [items_list[i].id] * results_df.shape[0]
                results_df['annotation_type'] = [compare_type] * results_df.shape[0]
                all_results = pd.concat([all_results, results_df],
                                        ignore_index=True)

        ###############################################
        # Save results to csv for IOU/label/attribute #
        ###############################################
        # "/.modelscores/modelId.csv"
        all_results['model_id'] = [model.id] * all_results.shape[0]
        all_results['dataset_id'] = [dataset.id] * all_results.shape[0]

        if not os.path.isdir(os.path.join(os.getcwd(), '.dataloop')):
            os.mkdir(os.path.join(os.getcwd(), '.dataloop'))
        scores_filepath = os.path.join(os.getcwd(), '.dataloop', f'{model.id}.csv')

        all_results.to_csv(scores_filepath, index=False)
        item = dataset.items.upload(local_path=scores_filepath,
                                    remote_path=f'/.modelscores',
                                    overwrite=True)

        return True, f'Successfully created model scores and saved as item {item.id}.'


    @staticmethod
    @dl.Package.decorators.function(display_name='Compare annotations to score',
                                    inputs={"annot_collection_1": "List",
                                            "annot_collection_2": "List"},
                                    outputs={})
    def create_annotation_scores(annot_collection_1,
                                 annot_collection_2,
                                 gt_is_first=True,
                                 task_type=None,
                                 compare_types=None) -> dict:
        """

        :param annot_collection_1:
        :param annot_collection_2:
        :param gt_is_first:
        :param task_type:
        :param compare_types:
        :return: dict of feature sets, indexed by the type of score (e.g. IOU)
        """
        if compare_types is None:
            compare_types = all_compare_types

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
            compare_types=[compare_types],
            ignore_labels=False)

        for compare_type in compare_types:
            results_df = results[compare_type].to_df()

        # TODO: update when feature vectors are working
        #######################
        # Create feature sets #
        #######################
        # for i, row in results_df.iterrows():
        #     for score, feature_set in score_sets.items():
        #         if row['first_id'] is not None:
        #             feature1 = feature_set.features.create(value=[row[results_columns[score.lower()]]],
        #                                                    project_id=project.id,
        #                                                    entity_id=row['first_id'],
        #                                                    refs={'item': row['item_id'],
        #                                                          'annotator': row['first_creator'],
        #                                                          'dataset': dataset.id,
        #                                                          'relative': row['second_id'],
        #                                                          })
        #         if row['second_id'] is not None:
        #             feature2 = feature_set.features.create(value=[row[results_columns[score.lower()]]],
        #                                                    project_id=project.id,
        #                                                    entity_id=row['second_id'],
        #                                                    refs={'item': row['item_id'],
        #                                                          'annotator': row['second_creator'],
        #                                                          'dataset': dataset.id,
        #                                                          'relative': row['first_id'],
        #                                                          })
        return score_sets  # TODO: return some sort of summary? bc each pair of matched annotations will have 2 feature vectors


    @staticmethod
    def consensus_items_summary(feature_set):
        # TODO should receive items, and query by that, not by feature set

        # go through each feature, average the score
        features = feature_set.features.list().all()
        return np.mean([feature.value for feature in features])


    @staticmethod
    def plot_precision_recall(scores: pd.DataFrame,
                              metric: str,
                              metric_threshold=0.5,
                              labels=None,
                              local_path=None):
        """
        Plot precision recall curve for a given metric threshold

        :param scores: dataframe of all the annotation scores
        :param metric: name of the column in the scores dataframe to use as the metric
        :param metric_threshold: threshold for which to calculate TP/FP
        :return:
        """
        import matplotlib.pyplot as plt
        from dtlpymetrics.tools.Evaluator import Evaluator
        from dtlpymetrics.tools.utils import MethodAveragePrecision

        if metric.lower() == 'iou':
            metric = 'geometry_score'
        elif metric.lower() == 'accuracy':
            metric = 'label_score'

        if metric not in scores.columns:
            raise ValueError(f'{metric} metric not included in scores.')

        #########################
        # plot precision/recall #
        #########################
        # calc
        method = MethodAveragePrecision.EveryPointInterpolation
        # method = MethodAveragePrecision.ElevenPointInterpolation
        plt.figure()
        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)

        if labels is None:
            labels = pd.concat([scores.first_label, scores.second_label]).dropna()

        for label in labels.unique():
            label_confidence_df = scores[scores.first_label == label].copy()

            label_confidence_df.sort_values('second_confidence', inplace=True, ascending=True)
            true_positives = label_confidence_df.geometry_score >= metric_threshold  # geometry score is IOU
            false_positives = label_confidence_df.geometry_score < metric_threshold
            #
            num_gts = sum(scores.first_id.notna())

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
            plt.plot(mrec, mpre, label=[label])  # , model_name])
        plt.legend()

        model_id = scores["model_id"][0]
        plot_filename = f'precision_recall_{model_id}_{metric}_{metric_threshold}.png'
        if local_path is None:
            save_path = os.path.join(os.getcwd(), '.dataloop', plot_filename)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path)
        else:
            plt.savefig(plot_filename)
        plt.close()

        print(f'saved precision recall plot to {plot_filename}')
        return [mrec, mpre]


    @staticmethod
    def get_scores_df(model: dl.Model, dataset: dl.Dataset):
        """
        Retrieves the dataframe for all the scores for a given model on a dataset via a hidden csv file.
        :param model: Model entity
        :param dataset: Dataset where the model was evaluated
        :return:
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
                f'No scores file found for model {model.id} on dataset {dataset.id}. Please evaluate model on the dataset first.')

        scores_df = pd.read_csv(local_path)
        return scores_df

    # @staticmethod
    # def item_scores_to_df(items_list: list, feature_set: dl.FeatureSet):
    #     scores_df = pd.DataFrame()
    #     score_type = feature_set.name.split(' ')[-1]
    #
    #     for item in items_list:
    #         item_scores = pd.DataFrame()
    #         # item = dl.items.get(None, '6453bc227edd9a6a11237ab0')
    #         filters = dl.Filters(use_defaults=False,
    #                              custom_filter={
    #                                  '$and':
    #                                      [
    #                                          {'refs': {'item': item.id}},
    #                                          {'featureSetId': feature_set.id},
    #                                          {'dataType': 'annotationScore'}
    #                                      ]
    #                              },
    #                              resource=dl.FiltersResource.FEATURE
    #                              )
    #         # dl.features.list(filters=filters).print()
    #         features = dl.features.list(filters=filters).all()
    #         for feature in features:
    #             feature_dict = {"annotation id": feature.entity_id,
    #                             "item id": feature.refs['item'],
    #                             "score": score_type,
    #                             "value": feature.value[0],
    #                             "GT annotation id": feature.refs['relative'],
    #                             }
    #             item_scores = pd.concat([item_scores, pd.DataFrame(feature_dict, index=[0])])
    #
    #         scores_df = pd.concat([scores_df, item_scores],
    #                               ignore_index=True)
    #
    #     return scores_df
