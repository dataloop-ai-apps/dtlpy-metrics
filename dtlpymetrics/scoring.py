import logging
import os
from typing import List

import dtlpy as dl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtlpymetrics.metrics_utils import measure_annotations
import datetime
from dtlpymetrics.dtlpy_scores import Score, Scores, ScoreType

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
logger = logging.getLogger('scoring-and-metrics')


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
    def calculate_consensus_task_score(consensus_task: dl.Task):
        """
        Calculate consensus scores for all items in a consensus task

        :param consensus_task:
        :return:
        """
        # workaround for the task query returning all items, including hidden consensus clones
        filters = dl.Filters()
        filters.add(field='hidden', values=False)
        items = consensus_task.get_items(filters=filters).all()

        for item in items:
            if item.metadata['system']['refs'][0]['metadata']['status'] == 'consensus_done':
                item = ScoringAndMetrics.create_consensus_item_score(item, consensus_task)

        score_summary = {}

        # this is the old way of uploading scores/features
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
        #         item = ScoringAndMetrics.create_consensus_item_score(item, consensus_task)

        # return scores?
        return dataset1, dataset2

    @staticmethod
    def calculate_annotator_scores():
        ### Call create_annotation_scores, get annotation scores and group by annotator
        # this requires being able to query scores by annotator, which is currently not supported by R&D

        return

    @staticmethod
    @dl.Package.decorators.function(display_name='Create item consensus score',
                                    inputs={"item": "Item"},
                                    outputs={"item": "Item"}
                                    )
    def create_consensus_item_score(item: dl.Item,
                                    context: dl.Context = None) -> dl.Item:
        """
        Create a consensus score for an item

        The first set of annotations is considered the reference set.
        :param item:
        :param context:
        :return: item
        """
        # ################################
        # # find task ID to get the task #
        # ################################
        #
        # metadata_list = item.metadata['system']['refs']
        # for metadata_dict in metadata_list:
        #     if metadata_dict['type'] == 'task':
        #         task_id = metadata_dict['id']
        #         break

        ##################################
        # collect annotators to group by #
        ##################################
        annotators = []
        consensus_task = context.task
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
        annotation_scores = []
        for i_annotator in range(n_annotators):
            for j_annotator in range(0, i_annotator + 1):
                annot_collection_1 = annots_by_annotator[annotators[i_annotator]]
                annot_collection_2 = annots_by_annotator[annotators[j_annotator]]

                pairwise_scores = ScoringAndMetrics.create_annotation_scores(annot_collection_1=annot_collection_1,
                                                                             annot_collection_2=annot_collection_2)
                annotation_scores.extend(pairwise_scores)

        # calculate item overall score as the average of all scores
        item_score_total = 0
        for annotation_score in annotation_scores:
            item_score_total += annotation_score.value

        #############################
        # upload scores to platform #
        #############################

        # clean previous scores first
        dl_scores = Scores(client_api=dl.client_api)
        dl_scores.delete(context={'itemId': item.id,
                                  'taskId': consensus_task.id})
        dl_annot_scores = dl_scores.create(annotation_scores)
        logger.info(f'Uploaded {len(dl_annot_scores)} annotation scores to platform.')

        item_score = Score(type=ScoreType.ITEM_OVERALL,
                           value=item_score_total / len(annotation_scores),
                           entity_id=item.id,
                           task_id=consensus_task.id,
                           item_id=item.id,
                           dataset_id=item.dataset.id)
        dl_item_score = dl_scores.create([item_score])
        logger.info(f'Uploaded overall score for item {item.id} to platform.')

        return item

    @staticmethod
    def create_model_score(dataset: dl.Dataset = None,
                           filters: dl.Filters = None,
                           model: dl.Model = None,
                           ignore_labels=False,
                           compare_types=None):
        """
        Measures scores for a set of model predictions compared against ground truth annotations.

        :param dataset: Dataset associated with the ground truth annotations
        :param filters: DQL Filter for retrieving the test items
        :param model: Model for evaluating predictions
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
            if compare_types not in model.output_type:  # TODO check this validation logic
                raise ValueError(
                    f'Annotation type {compare_types} does not match model output type {model.output_type}')
            compare_types = [compare_types]

        annot_set_1 = []
        annot_set_2 = []

        if model.name is None:
            return False, 'No model name found for the second set of annotations, please provide model name.'
        if not items_list:
            return False, 'No items found in the dataset, please check the dataset and filters.'

        ########################################
        # Create list of item annotation lists #
        ########################################
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

        #########################################################
        # Compare annotations and return concatenated dataframe #
        #########################################################
        all_results = pd.DataFrame()
        for i in range(len(items_list)):
            # compare annotations for each item
            # print(f'item {i}: GT annots {len(annot_set_1[i])}, model annots {len(annot_set_2[i])}')
            if len(annot_set_1[i]) == 0 and len(annot_set_2[i]) == 0:
                continue
            else:
                results = measure_annotations(annotations_set_one=annot_set_1[i],
                                              annotations_set_two=annot_set_2[i],
                                              match_threshold=0.01,  # to get all possible matches
                                              ignore_labels=ignore_labels,
                                              compare_types=compare_types)
                for compare_type in compare_types:
                    try:
                        results_df = results[compare_type].to_df()
                    except KeyError:
                        continue
                    results_df['item_id'] = [items_list[i].id] * results_df.shape[0]
                    results_df['annotation_type'] = [compare_type] * results_df.shape[0]
                    all_results = pd.concat([all_results, results_df],
                                            ignore_index=True)

        ###############################################
        # Save results to csv for IOU/label/attribute #
        ###############################################
        # TODO save via feature vectors when ready
        # file format "/.modelscores/modelId.csv"
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
    @dl.Package.decorators.function(display_name='Compare two sets of annotations for scoring',
                                    inputs={"annot_collection_1": "List",
                                            "annot_collection_2": "List"},
                                    outputs={})
    def create_annotation_scores(annot_collection_1,
                                 annot_collection_2,
                                 task_type=None,
                                 compare_types=None) -> List[Score]:
        """

        The first annotation collection is considered as the reference, and the second set is the test.
        If we switch the order of the annotation collections, the scores remain the same but the user id context changes.

        :param annot_collection_1:
        :param annot_collection_2:
        :param gt_is_first:
        :param task_type:
        :param compare_types:
        :return: dict of feature sets, indexed by the type of score (e.g. IOU)
        """
        if compare_types is None:
            compare_types = all_compare_types

        # compare bounding box annotations
        results = measure_annotations(
            annotations_set_one=annot_collection_1,
            annotations_set_two=annot_collection_2,
            compare_types=[compare_types],
            ignore_labels=False)

        all_results = pd.DataFrame()
        for compare_type in compare_types:
            try:
                results_df = results[compare_type].to_df()
                all_results = pd.concat([all_results, results_df])
            except KeyError:
                continue

        scores = []
        # TODO add support for the user to choose which score types to save
        score_types = [ScoreType.ANNOTATION_IOU, ScoreType.ANNOTATION_LABEL]

        for i, row in all_results.iterrows():
            for score_type in score_types:
                score = Score(type=score_type.value,
                              value=row[results_columns[score_type.value.lower()]],
                              entity_id=row['second_id'],
                              user_id=row['second_user_id'],
                              item_id=row['item_id'],
                              dataset_id=row['dataset_id'])
                scores.append(score)

        return scores

    @staticmethod
    def plot_precision_recall(plot_points: pd.DataFrame,
                              label_names=None,
                              local_path=None):
        """
        Plot precision recall curve for a given metric threshold

        :param plot_points: dict generated from calculate_precision_recall with all the points to plot by label and
         the entire dataset. keys include: confidence threshold, iou threshold, dataset levels precision, recall, and
         confidence, and label-level precision, recall and confidence
        :param label_names: list of label names to plot
        :param local_path: path to save plot
        :return:
        """

        ###################
        # plot by dataset #
        ###################

        plt.figure()
        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)

        # plot each label separately
        dataset_points = plot_points[plot_points['data'] == 'dataset']

        plt.plot(dataset_points['recall'],
                 dataset_points['precision'])

        # plot the dataset level
        plot_filename = f"dataset_precision_recall_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png"
        if local_path is None:
            save_path = os.path.join(os.getcwd(), '.dataloop', plot_filename)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path)
        else:
            save_path = os.path.join(local_path, plot_filename)
            plt.savefig(save_path)
        plt.close()

        print(f'saved dataset precision recall plot to {save_path}')

        #################
        # plot by label #
        #################
        all_labels = plot_points[plot_points['data'] == 'label']

        if (label_names is None) or (bool(label_names) is False):
            label_names = all_labels['label_name'].copy().drop_duplicates()

        plt.figure()
        plt.xlim(0, 1.1)
        plt.ylim(0, 1.1)
        plt.legend()

        # plot each label separately
        for label_name in label_names:
            label_points = all_labels[all_labels['label_name'] == label_name].copy()

            plt.plot(label_points['recall'],
                     label_points['precision'],
                     label=[label_name])

        # plot the dataset level
        plot_filename = f"label_precision_recall_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png"
        if local_path is None:
            save_path = os.path.join(os.getcwd(), '.dataloop', plot_filename)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path)
        else:
            save_path = os.path.join(local_path, plot_filename)
            plt.savefig(save_path)
        plt.close()

        print(f'saved labels precision recall plot to {save_path}')
        return save_path

    @staticmethod
    def calc_precision_recall(dataset_id: str,
                              model_id: str,
                              iou_threshold=0.01) -> pd.DataFrame:
        # method_type='every_point'):
        """
        Plot precision recall curve for a given metric threshold

        :param dataset_id: str dataset ID
        :param model_id: str model ID
        :return: dataframe with all the points to plot for the dataset and individual labels
        """
        ################################
        # get matched annotations data #
        ################################
        # TODO use feature management once available
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
        if len(label_names) == 0:
            label_names = list(pd.concat([scores.first_label, scores.second_label]).dropna().drop_duplicates())

        ##############################
        # calculate precision/recall #
        ##############################
        dataset_points = {'iou_threshold': {},
                          'data': {},  # "dataset" or "label"
                          'label_name': {},  # label name or NA,
                          'precision': {},
                          'recall': {},
                          'confidence': {}
                          }

        num_gts = sum(scores.first_id.notna())
        detections = scores[scores.second_id.notna()].copy()

        detections.sort_values('second_confidence', inplace=True, ascending=False, ignore_index=True)
        detections['true_positives'] = detections['geometry_score'] >= iou_threshold
        detections['false_positives'] = detections['geometry_score'] < iou_threshold

        # get dataset-level precision/recall
        dataset_fps = np.cumsum(detections['false_positives'])
        dataset_tps = np.cumsum(detections['true_positives'])
        dataset_recall = dataset_tps / num_gts
        dataset_precision = np.divide(dataset_tps, (dataset_fps + dataset_tps))

        [_,
         dataset_plot_precision,
         dataset_plot_recall,
         dataset_plot_confidence] = \
            ScoringAndMetrics.every_point_curve(recall=list(dataset_recall),
                                                precision=list(dataset_precision),
                                                confidence=list(detections['second_confidence']))

        dataset_points['iou_threshold'] = [iou_threshold] * len(dataset_plot_precision)
        dataset_points['data'] = ['dataset'] * len(dataset_plot_precision)
        dataset_points['label_name'] = ['NA'] * len(dataset_plot_precision)
        dataset_points['precision'] = dataset_plot_precision
        dataset_points['recall'] = dataset_plot_recall
        dataset_points['confidence'] = dataset_plot_confidence

        dataset_df = pd.DataFrame(dataset_points).drop_duplicates()

        ##########################################
        # calculate label-level precision/recall #
        ##########################################
        all_labels = pd.DataFrame(columns=dataset_df.columns)

        label_points = {key: {} for key in dataset_points}

        for label_name in list(set(label_names)):
            label_detections = detections.loc[
                (detections.first_label == label_name) | (detections.second_label == label_name)].copy()
            if label_detections.shape[0] == 0:
                label_plot_precision = [0]
                label_plot_recall = [0]
                label_plot_confidence = [0]
            else:
                label_detections.sort_values('second_confidence', inplace=True, ascending=False, ignore_index=True)

                label_fps = np.cumsum(label_detections['false_positives'])
                label_tps = np.cumsum(label_detections['true_positives'])
                label_recall = label_tps / num_gts
                label_precision = np.divide(label_tps, (label_fps + label_tps))

                [_,
                 label_plot_precision,
                 label_plot_recall,
                 label_plot_confidence] = \
                    ScoringAndMetrics.every_point_curve(recall=list(label_recall),
                                                        precision=list(label_precision),
                                                        confidence=list(label_detections['second_confidence']))

            label_points['iou_threshold'] = [iou_threshold] * len(label_plot_precision)
            label_points['data'] = ['label'] * len(label_plot_precision)
            label_points['label_name'] = [label_name] * len(label_plot_precision)
            label_points['precision'] = label_plot_precision
            label_points['recall'] = label_plot_recall
            label_points['confidence'] = label_plot_confidence

            label_df = pd.DataFrame(label_points).drop_duplicates()
            all_labels = pd.concat([all_labels, label_df])

        ##################
        # concat all data #
        ##################
        plot_points = pd.concat([dataset_df, all_labels])

        return plot_points

    @staticmethod
    def every_point_curve(recall: list, precision: list, confidence: list):
        """
        Calculate precision-recall curve from a list of precision & recall values
        :param recall: list of recall values
        :param precision: list of precision values
        :return: list of average precision all values, precision points, recall points
        """
        recall_points = np.concatenate([[0], recall, [1]])
        precision_points = np.concatenate([[0], precision, [0]])
        confidence_points = np.concatenate([[confidence[0]], confidence, [confidence[-1]]])

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
                confidence_points[0:len(precision_points) - 1]]

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
            label_names = pd.concat([scores.first_label, scores.second_label]).dropna()

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


if __name__ == '__main__':
    # create_faas()
    # project = dl.projects.get("feature vectors")
    # codebase = project.codebases.pack()
    #
    # module = dl.PackageModule.from_entry_point(entry_point='scoring.py')
    #
    # package_name = 'consensus_scoring'
    # package = project.packages.push(package_name=package_name,
    #                                 package_type='ml',
    #                                 codebase=codebase,  # can also use src_path
    #                                 modules=[module],
    #                                 is_global=False)
    # service_config=service_config,
    # slots=slots,
    # metadata=metadata)

    import json

    # from Or's project, which has a precision recall plot to compare
    dl.setenv('new-dev')
    dataset_id = '648f5926943352ccaddf0149'
    model_id = '648ffafe28146328fb4e96b3'

    # # models to rerun and upload: 649076c45a9c968a5c32ed65, 6490b666d8a0841b563176f6

    test_filter = '{"filter": {"$and": [{"hidden": false}, {"$or": [{"metadata": {"system": {"tags": {"test": true}}}}]}, {"type": "file"}]}, "page": 0, "pageSize": 1000, "resource": "items"}'
    filters = dl.Filters(custom_filter=json.loads(test_filter))
    filters = dl.Filters()

    # for mohamed
    # dl.setenv('rc')
    # dataset_id = '649306160310862369075fb2'
    # model_id = '6493061a65aa855b8898fb00'
    # filters = dl.Filters()

    model = dl.models.get(None, model_id)
    dataset = dl.datasets.get(dataset_id=dataset_id)

    pages = dataset.items.list(filters=filters)
    print(f'items in test set: {pages.items_count}')

    # success, message = ScoringAndMetrics.create_model_score(dataset=dataset,
    #                                                         filters=filters,
    #                                                         model=model,
    #                                                         compare_types=model.output_type)
    # print(message)

    # model_scores = ScoringAndMetrics.get_scores_df(model=model, dataset=dataset)
    # metric_names = ['accuracy', 'iou', 'confidence']
    #
    plot_points = ScoringAndMetrics.calc_precision_recall(dataset_id=dataset.id,
                                                          model_id=model.id)

    labels = [label.tag for label in dataset.labels]
    save_path = ScoringAndMetrics.plot_precision_recall(plot_points=plot_points,
                                                        label_names=labels)
    # from pathlib import Path
    #
    # plot_points.to_csv(Path(Path(save_path).parent, '.dataloop', 'plot_points.csv'))
    # #
    # print()
