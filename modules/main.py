import dtlpy as dl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dtlpy.ml import metrics, predictions_utils
from pathlib import Path
from scoring import measure_annotations
from dtlpy import entities, utilities


@dl.Package.defs.module(name='scoring-and-metrics',
                        description='Scoring and metrics functions',
                        init_inputs={'task': entities.Task,
                                     'save_plot': bool}, )
class ScoringAndMetrics(dl.BaseServiceRunner):

    @staticmethod
    def calculate_consensus_score(consensus_task, save_plot=False):
        dataset = consensus_task.dataset
        project = dataset.project

        annotators = []
        n_annotators = consensus_task.metadata['system']['consensusAssignees']

        # workaround for the task query returning all items, including hidden consensus clones
        filters = dl.Filters()
        filters.add(field='hidden', values=False)
        items = consensus_task.get_items(filters=filters).all()  # why is this "get_items" and not "list"?

        consensus_items_scores = {}
        consensus_annotators_scores = []

        for item in items:
            annotator_agreement = np.zeros((n_annotators, n_annotators))
            annotations = item.annotations.list()

            try:
                annots_by_annotator = {annotator: [] for annotator in annotators}
            except NameError:
                annots_by_annotator = {}

            if item.metadata['system']['refs'][0]['metadata'][
                'status'] == 'consensus_done':  # TODO is this a clean way to do this?

                # group by some field (e.g. 'creator' or 'assignment id'), here we use annotator/creator
                for annotation in annotations:
                    annotator = annotation.creator

                    if annotator not in annotators:
                        annotators.append(annotator)
                        annots_by_annotator.update({annotator: []})

                    annots_by_annotator[annotator].append(annotation)

                # do pairwise comparisons of each annotator for all of their annotations on the item
                for i_annotator in range(n_annotators):
                    for j_annotator in range(n_annotators):

                        if i_annotator > j_annotator:
                            annot_collection_1 = annots_by_annotator[annotators[i_annotator]]
                            annot_collection_2 = annots_by_annotator[annotators[j_annotator]]

                            # compare annotations via
                            results = measure_annotations(
                                annotations_set_one=annot_collection_1,
                                annotations_set_two=annot_collection_2,
                                compare_types=[dl.AnnotationType.BOX],
                                ignore_labels=False)
                            annotator_agreement[i_annotator, j_annotator] = results['total_mean_score']

                if save_plot:
                    item_title = f'consensus scores for {item.name}'
                    filename = f'consensus_{Path(item.name).stem}_{item.id}.png'
                    ScoringAndMetrics.plot_matrix(item_title, filename, annotator_agreement, annotators, item)

                # calculate mean agreement across all annotators for lower half of the matrix, without the diagonal
                mean_item_consensus = np.mean(annotator_agreement[np.tril_indices(n=n_annotators, k=-1)])
                print(f'item consensus mean annotator score: {mean_item_consensus}')

                consensus_items_scores.update({item.id: mean_item_consensus})
                consensus_annotators_scores.append(annotator_agreement)

        stacked_scores = np.stack(consensus_annotators_scores, axis=0)

        mean_task_consensus = np.mean(stacked_scores, axis=0)

        ##############################
        # Save annotators IOU matrix #
        ##############################

        consensus_title = f'{dataset.name} task: average consensus score by annotator'
        filename = f'task_consensus_scores_{dataset.name}.png'
        ScoringAndMetrics.plot_matrix(consensus_title, filename, mean_task_consensus, annotators)

        ###################
        # Upload features #
        ###################

        try:
            items_consensus_set = project.feature_sets.get('consensus_items_scores')

        except dl.exceptions.NotFound:
            items_consensus_set = project.feature_sets.create(name='consensus_items_scores',
                                                              set_type='score',
                                                              data_type=dl.FeatureDataType.ITEM_SCORE,
                                                              entity_type=dl.FeatureEntityType.ITEM,
                                                              size=1)

        # return one score for the item as a result of the consensus comparison
        for i_item, item_id in enumerate(consensus_items_scores.keys()):
            items_consensus_set.features.create(value=[consensus_items_scores[item_id]],
                                                project_id=project.id,
                                                entity_id=item_id,
                                                version='1.0.0')

        return items_consensus_set

    @staticmethod
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
