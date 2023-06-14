import os
import dtlpy as dl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtlpymetrics.metrics_utils import measure_annotations

score_names = ['IOU', 'label', 'attribute']
results_columns = {'iou': 'geometry_score', 'label': 'label_score', 'attribute': 'attribute_score'}
all_compare_types = [dl.AnnotationType.BOX,
                     dl.AnnotationType.CLASSIFICATION,
                     dl.AnnotationType.POLYGON,
                     dl.AnnotationType.POINT,
                     dl.AnnotationType.SEGMENTATION]

scorer = dl.AppModule(name='Scoring and metrics function',
                      description='Functions for calculating scores between annotations.'
                      )


@dl.Package.decorators.module(name='scoring-and-metrics',
                              description='Scoring and metrics functions')
class ScoringAndMetrics(dl.BaseServiceRunner):
    """
    Scoring and metrics allows comparison between items, annotators, models, datasets, and tasks.
    """

    @staticmethod
    @dl.Package.decorators.function(display_name='Calculate the consensus task score',
                                    inputs={"consensus_task": dl.Task},
                                    outputs={"score_summary": dict,
                                             "consensus_task": dl.Task}
                                    )
    def calculate_consensus_score(consensus_task: dl.Task):
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
                item = ScoringAndMetrics.create_item_consensus_score(item)

        score_summary = {}

        for score_name in score_names:
            score_summary.update({score_name: []})
            feature_set = consensus_task.project.feature_sets.get(feature_set_name=f'Consensus {score_name}')

            for feature in feature_set.features.list().all():
                print(feature.value[0])
                score_summary[score_name].append(feature.value[0])

            print(f'Consensus average {score_name}: {np.mean(score_summary[score_name])}')

        return score_summary, consensus_task

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
            compare_types = [compare_types]  # TODO add validation for compare_type and model output type

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
        feature_set_prefix = 'Consensus ' if task_type == 'consensus' else ''

        score_sets = {}

        # TODO: update when feature vectors are working
        # for score_name in score_names:
        #     try:
        #         feature_set = project.feature_sets.get(feature_set_name=f'{feature_set_prefix}{score_name}')
        #     except dl.exceptions.NotFound:
        #         # create the feature set for each score type
        #         feature_set = project.feature_sets.create(name=f'{feature_set_prefix}{score_name}',
        #                                                   set_type='scores',
        #                                                   # refs require data type
        #                                                   data_type=dl.FeatureDataType.ANNOTATION_SCORE,
        #                                                   entity_type=dl.FeatureEntityType.ANNOTATION,
        #                                                   size=1)
        #     score_sets.update({score_name: feature_set})

        # compare bounding box annotations
        results = measure_annotations(
            annotations_set_one=annot_collection_1,
            annotations_set_two=annot_collection_2,
            compare_types=[compare_types],
            ignore_labels=False)

        for compare_type in compare_types:
            results_df = results[compare_type].to_df()

        # TODO: update when feature vectors are working
        ###################
        # Create features #
        ###################
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

    def plot_precision_recall(plot_points: dict, local_path=None):
        """
        Plot precision recall curve for a given metric threshold

        :param plot_points: dictionary of precision/recall points
        :param local_path: path to save plot
        :return:
        """

        labels = list(plot_points['labels'].keys())

        plt.figure()
        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)

        for label in labels:
            plt.plot(plot_points['labels'][label]['recall'],
                     plot_points['labels'][label]['precision'],
                     label=[label])
        plt.legend()

        # plot_filename = f'precision_recall_{dataset_id}_{model_id}_{plot_points[metric]}_{metric_threshold}.png'
        plot_filename = f'precision_recall.png'
        if local_path is None:
            save_path = os.path.join(os.getcwd(), '.dataloop', plot_filename)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path)
        else:
            save_path = os.path.join(local_path, plot_filename)
            plt.savefig(save_path)

        plt.close()

        print(f'saved precision recall plot to {save_path}')
        return save_path

    @staticmethod
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
                       'dataset_precision': [],
                       'dataset_recall': []
                       }

        num_gts = sum(scores.first_id.notna())

        scores_positives = scores[scores['geometry_score'] > iou_threshold].copy()

        scores_positives.sort_values('second_confidence', inplace=True, ascending=True, ignore_index=True)
        scores_positives['true_positives'] = scores_positives['second_confidence'] >= conf_threshold
        scores_positives['false_positives'] = scores_positives['second_confidence'] < conf_threshold

        # get dataset-level precision/recall
        dataset_fps = np.cumsum(scores_positives['false_positives'])
        dataset_tps = np.cumsum(scores_positives['true_positives'])
        dataset_recall = dataset_tps / num_gts
        dataset_precision = np.divide(dataset_tps, (dataset_fps + dataset_tps))

        [_, dataset_plot_precision, dataset_plot_recall] = ScoringAndMetrics.every_point_curve(dataset_recall, dataset_precision)
        plot_points['dataset_precision'] = dataset_plot_precision
        plot_points['dataset_recall'] = dataset_plot_recall

        # get label-level precision/recall
        for label in list(set(label_names)):
            label_positives = scores_positives[scores_positives.first_label == label].copy()
            label_positives.sort_values('second_confidence', inplace=True, ascending=True, ignore_index=True)

            label_fps = np.cumsum(label_positives['false_positives'])
            label_tps = np.cumsum(label_positives['true_positives'])
            label_recall = label_tps / num_gts
            label_precision = np.divide(label_tps, (label_fps + label_tps))

            [_, label_plot_precision, label_plot_recall] = ScoringAndMetrics.every_point_curve(label_recall, label_precision)

            plot_points['labels'].update({label: {
                'precision': label_plot_precision,
                'recall': label_plot_recall}})

        return plot_points

    @staticmethod
    def every_point_curve(recall: list, precision: list):
        """
        Calculate precision-recall curve from a list of precision & recall values
        :param recall: list of recall values
        :param precision: list of precision values
        :return:
        """
        recall_points = np.concatenate([[0], recall, [1]])
        precision_points = np.concatenate([[0], precision, [1]])

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
                recall_points[0:len(precision_points) - 1]]

    @staticmethod
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


def create_faas():
    pass


if __name__ == '__main__':
    # create_faas()

    project = dl.projects.get("feature vectors")
    codebase = project.codebases.pack()

    module = dl.PackageModule.from_entry_point(entry_point='scoring.py')

    package_name = 'consensus_scoring'
    package = project.packages.push(package_name=package_name,
                                    package_type='ml',
                                    codebase=codebase,  # can also use src_path
                                    modules=[module],
                                    is_global=False)
                                    # service_config=service_config,
                                    # slots=slots,
                                    # metadata=metadata)
