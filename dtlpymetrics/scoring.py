import os

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
        :param compare_types:
        :return:
        """

        if dataset is None:
            return False, 'No dataset provided, please provide a dataset.'

        items_list = list(dataset.items.list(filters=filters).all())

        annot_set_1 = []
        annot_set_2 = []

        if model.name is None:
            return False, 'No model name found for the second set of annotations, please provide model name.'

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
        if not item_annots_2:
            return False, 'No model annotations found. Please ensure there are model annotations on the items.'

        #########################################################
        # Compare annotations and return concatenated dataframe #
        #########################################################
        all_results = pd.DataFrame()
        for i in range(len(items_list)):
            # compare annotations for each item
            print(f'item {i}: GT annots {len(annot_set_1[i])}, model annots {len(annot_set_2[i])}')
            results = measure_annotations(annotations_set_one=annot_set_1[i],
                                          annotations_set_two=annot_set_2[i],
                                          match_threshold=match_threshold,
                                          ignore_labels=ignore_labels,
                                          compare_types=compare_types)

            results_df = results[compare_type].to_df()

            all_results = pd.concat([all_results, results_df],
                                    ignore_index=True)

        ###############################################
        # Save results to csv for IOU/label/attribute #
        ###############################################
        # "/.modelscores/modelId.csv"
        scores_filepath = os.path.join(os.getcwd(), '.output', f'{model.id}.csv')
        all_results.to_csv(scores_filepath, index=False)
        item = dataset.items.upload(local_path=scores_filepath,
                                    remote_path=f'/.modelscores',
                                    overwrite=True)

        return True, f'Successfully created model scores and saved to {scores_filepath}.'

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
        from dtlpymetrics.lib.Evaluator import Evaluator
        from dtlpymetrics.lib.utils import MethodAveragePrecision

        #########################
        # plot precision/recall #
        #########################
        # calc
        method = MethodAveragePrecision.EveryPointInterpolation
        # method = MethodAveragePrecision.ElevenPointInterpolation
        plt.figure()

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

        plot_filename = f'precision_recall_{model.id}_{metric}_{metric_threshold}.png'
        if local_path is None:
            save_path = os.path.join(os.getcwd(), '.output', plot_filename)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path)
        else:
            plt.savefig(plot_filename)
        plt.close()

        print(f'saved precision recall plot to {plot_filename}')
        return

    @staticmethod
    def get_scores_df(model: dl.Model, dataset: dl.Dataset):
        """
        Retrieves the dataframe for all the scores for a given model on a dataset via a hidden csv file.
        :param model: Model entity
        :param dataset: Dataset where the model was evaluated
        :return:
        """
        scores_filepath = f'/.output/{model.id}.csv'
        filters = dl.Filters(field='name', values=f'{model.id}.csv')
        filters.add(field='hidden', values=True)
        pages = dataset.items.list(filters=filters)

        if pages.items_count > 0:
            for item in pages.all():
                item.download(local_path=scores_filepath)

        scores_df = pd.read_csv(scores_filepath)
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
    import dtlpy as dl

    test_type = 'model eval'
    if test_type == 'consensus':
        ########################
        # Consensus task test ##
        ########################
        scores = ScoringAndMetrics()
        # consensus_task = dl.tasks.get(task_name='pipeline consensus test (test tasks)')
        consensus_task = dl.tasks.get(task_id='644a307ae052f434dab98ff3')
        scores.calculate_consensus_items(consensus_task)

    if test_type == 'model eval':
        ####################
        # Model evaluation #
        ####################
        dl.setenv('rc')

        # project = dl.projects.get('Active Learning')
        # project = dl.projects.get('Model mgmt demo')

        # dataset = dl.datasets.get(None, '6450c91e200a21ed641a332a')  # active learning dataset in model mgmt
        # dataset = project.datasets.get('active learning')

        # project = dl.projects.get('Fruit - Model Mgmt')
        # dataset = project.datasets.get('Fruit')

        ############################################
        # Create new dataset and model predictions #
        ############################################

        # dataset_origin = project.datasets.get('taco_trash')
        # dataset = project.datasets.clone(dataset_id=dataset_origin.id,
        #                                  clone_name='taco_trash_eval')

        ##########################
        #  Find and deploy model #
        ##########################
        # filters = dl.Filters(resource=dl.FiltersResource.MODEL,
        #                      field='name', values='*yolov5*')
        # dl.models.list(filters=filters).print()
        #
        # public_model = dl.models.get('yolov5-original')
        # model = project.models.clone(from_model=public_model,
        #                              model_name='taco_cloned_yolo_demo')
        # model.deploy()

        # project = dl.projects.get('Model mgmt demo')
        # dataset = project.datasets.get('cloned taco trash')  # 1500 items
        # model = dl.models.get(None, '645a25e5bdef8b500213ded6')

        project = dl.projects.get('Active Learning')
        dataset = project.datasets.get('active learning test')

        def create_model():
            model_origin = dl.models.get(None, '6462417fcc4a02b5bea99e14')
            model = project.models.clone(from_model=model_origin,
                                         model_name='cloned_yolo_v8',
                                         project_id='f7d43fec-2823-4871-b0a0-1b76a75a2d61')
            model.deploy()

        model = project.models.get('cloned_yolo_v8')

        items_list = list(dataset.items.list().all())
        items_ids = [item.id for item in items_list]
        model.predict(item_ids=items_ids)

        ########################################
        # Evaluate the model and create scores #
        ########################################

        # filters = dl.Filters(resource=dl.FiltersResource.MODEL,
        #                      field='name', values='yolov8')
        # dl.models.list(filters=filters).print()
        # dl.models.list().print()

        # model = project.models.get('cloned_yolo_demo')
        # model = dl.models.get('yolov3')
        # model = project.models.get('clone_predict_pretrained-yolo-v5-small')
        success, message = ScoringAndMetrics.create_model_score(dataset=dataset,
                                                                model=model,
                                                                match_threshold=0.3,
                                                                ignore_labels=True)
        print(message)

        # fset = project.feature_sets.get('Model IOU')
        # model_scores = dataset.items.get(filepath=f'/.modelscores/{model.id}.csv')
        # ScoringAndMetrics.item_scores_df(model_scores)
        model_scores = ScoringAndMetrics.get_scores_df(model=model, dataset=dataset)

        score_columns = ['iou', 'confidence']
        ScoringAndMetrics.plot_precision_recall(scores=model_scores,
                                                metric=score_columns[0],
                                                metric_threshold=0.5)
