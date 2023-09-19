import logging
import dtlpy as dl
from dtlpymetrics.dtlpy_scores import Score, ScoreType
from dtlpymetrics.utils import mean_or_default, calculate_annotation_score, add_score_context


def get_image_scores(annots_by_assignment: dict,
                     assignments_by_annotator: dict,
                     task: dl.Task,
                     item: dl.Item,
                     score_types: list,
                     task_type: str = 'testing',
                     logger: logging.Logger = None):
    """
    Calculate scores for an image item

    @type task: object
    @param annots_by_assignment:
    @param assignments_by_annotator:
    @param task:
    @param item:
    @param score_types:
    @param task_type:
    @param logger:

    @return:
    """
    ####################
    # calculate scores #
    ####################
    labels = dl.recipes.get(task.recipe_id).ontologies.list()[0].labels_flat_dict.keys()
    confusion_by_label = {l: {m: 0 for m in labels} for l in labels}

    # compare between each assignment and create Score entities
    all_scores = list()

    # do pairwise comparisons of each assignment for all annotations on the item
    for i_assignment, assignment_annotator_i in enumerate(annots_by_assignment):
        if task_type == "testing" and assignment_annotator_i != 'ref':
            # if "testing", compare only to ref
            continue
        for j_assignment, assignment_annotator_j in enumerate(annots_by_assignment):
            # don't compare a set to itself
            if i_assignment == j_assignment:
                continue
            # skip ref in inner loop
            if assignment_annotator_j == 'ref':
                continue
            logger.info(
                f'Comparing assignee: {assignment_annotator_i!r} with assignee: {assignment_annotator_j!r}')
            annot_collection_1 = annots_by_assignment[assignment_annotator_i]
            annot_collection_2 = annots_by_assignment[assignment_annotator_j]
            # score types that can be returned: ANNOTATION_IOU, ANNOTATION_LABEL, ANNOTATION_ATTRIBUTE
            pairwise_scores = calculate_annotation_score(annot_collection_1=annot_collection_1,
                                                         annot_collection_2=annot_collection_2,
                                                         ignore_labels=True,
                                                         match_threshold=0.01,
                                                         score_types=score_types)

            # update scores with context
            updated_scores = []
            for score in pairwise_scores:
                updated_score = add_score_context(score=score,
                                                  user_id=assignment_annotator_j,
                                                  task_id=task.id,
                                                  assignment_id=assignments_by_annotator[assignment_annotator_j].id,
                                                  item_id=item.id)
                updated_scores.append(updated_score)

            raw_annotation_scores = [score for score in updated_scores if score.type != ScoreType.LABEL_CONFUSION]
            confusion_scores = [score for score in updated_scores if score.type == ScoreType.LABEL_CONFUSION]

            # calc general label confusion for the entire annotation collection/item
            for score in confusion_scores:
                if score.entity_id not in confusion_by_label:
                    confusion_by_label[score.entity_id] = dict()
                if score.relative not in confusion_by_label[score.entity_id]:
                    confusion_by_label[score.entity_id][score.relative] = 0
                confusion_by_label[score.entity_id][score.relative] += score.value

            # calc overall annotation
            user_annotation_overalls = list()
            for annotation in annot_collection_2:  # go over all annotations from the "test" set
                single_annotation_scores = mean_or_default(arr=[score.value
                                                                for score in raw_annotation_scores
                                                                if score.entity_id == annotation.id],
                                                           default=1)
                # ANNOTATION_OVERALL
                user_annotation_overalls.append(single_annotation_scores)
                annotation_overall = Score(type=ScoreType.ANNOTATION_OVERALL,
                                           value=single_annotation_scores,
                                           entity_id=annotation.id,
                                           task_id=task.id,
                                           item_id=item.id,
                                           user_id=assignment_annotator_j,
                                           dataset_id=item.dataset.id)
                all_scores.append(annotation_overall)

            # calc user confusion
            user_confusion_score = Score(type=ScoreType.USER_CONFUSION,
                                         value=mean_or_default(arr=user_annotation_overalls,
                                                               default=1),
                                         entity_id=assignment_annotator_j,
                                         user_id=assignment_annotator_j,
                                         relative=assignment_annotator_i,  # this can be "ref"
                                         task_id=task.id,
                                         item_id=item.id,
                                         dataset_id=item.dataset.id)
            all_scores.append(user_confusion_score)
            all_scores.extend(raw_annotation_scores)

    return all_scores


def get_video_scores(annotations_by_frame: dict,
                     assignments_by_annotator: dict,
                     task: dl.Task,
                     item: dl.Item,
                     score_types: list,
                     task_type: str = 'testing',
                     logger: logging.Logger = None):
    """
    Generate scores for a video item

    @param task:
    @param item:
    @param score_types:
    @param task_type:
    @param annotations_by_frame:
    @param assignments_by_annotator:
    @param logger:
    @return:
    """

    ####################
    # calculate scores #
    ####################
    all_scores_by_annotation = dict()

    for frame, annots_by_assignment in annotations_by_frame.items():
        # compare between each assignment and create Score entities
        frame_scores = list()

        # do pairwise comparisons of each assignment for all annotations on the item
        for i_assignment, assignment_annotator_i in enumerate(annots_by_assignment):
            if task_type == "testing" and assignment_annotator_i != 'ref':
                # if "testing", compare only to ref
                continue
            for j_assignment, assignment_annotator_j in enumerate(annots_by_assignment):
                # don't compare a set to itself
                if i_assignment == j_assignment:
                    continue
                # skip ref in inner loop
                if assignment_annotator_j == 'ref':
                    continue
                logger.info(
                    f'Comparing assignee: {assignment_annotator_i!r} with assignee: {assignment_annotator_j!r}')
                annot_collection_1 = annots_by_assignment[assignment_annotator_i]
                annot_collection_2 = annots_by_assignment[assignment_annotator_j]

                # score types that can be returned: ANNOTATION_IOU, ANNOTATION_LABEL, ANNOTATION_ATTRIBUTE
                pairwise_scores = calculate_annotation_score(annot_collection_1=annot_collection_1,
                                                             annot_collection_2=annot_collection_2,
                                                             ignore_labels=True,
                                                             include_confusion=False,
                                                             match_threshold=0.01,
                                                             score_types=score_types)

                updated_scores = []
                # update scores with context
                for score in pairwise_scores:
                    updated_score = add_score_context(score=score,
                                                      user_id=assignment_annotator_j,
                                                      task_id=task.id,
                                                      assignment_id=assignments_by_annotator[assignment_annotator_j].id,
                                                      item_id=item.id,
                                                      dataset_id=item.dataset.id)
                    updated_scores.append(updated_score)

                raw_annotation_scores = [score for score in updated_scores if score.type != ScoreType.LABEL_CONFUSION]
                frame_scores.extend(raw_annotation_scores)

                # calc overall annotation
                user_annotation_overalls = list()
                for annotation in annot_collection_2:  # go over all annotations from the "test" set
                    single_annotation_slice_score = mean_or_default(arr=[score.value
                                                                         for score in raw_annotation_scores
                                                                         if score.entity_id == annotation.id],
                                                                    default=1)
                    # overall slice score
                    user_annotation_overalls.append(single_annotation_slice_score)
                    annotation_overall = Score(type=ScoreType.ANNOTATION_OVERALL,
                                               value=single_annotation_slice_score,
                                               entity_id=annotation.id,
                                               task_id=task.id,
                                               item_id=item.id,
                                               user_id=assignment_annotator_j,
                                               dataset_id=item.dataset.id)
                    frame_scores.append(annotation_overall)

        for score in frame_scores:
            if score.entity_id not in all_scores_by_annotation:
                all_scores_by_annotation[score.entity_id] = list()
            all_scores_by_annotation[score.entity_id].append(score)

    # once each frame's score is calculated, take the average score of all frames
    all_scores = list()
    for annotation_id, annotation_frame_scores in all_scores_by_annotation.items():
        all_scores.append(Score(type=ScoreType.ANNOTATION_OVERALL,
                                value=mean_or_default(arr=[score.value for score in annotation_frame_scores if
                                                           score.type == ScoreType.ANNOTATION_OVERALL.value],
                                                      default=1),
                                entity_id=annotation_id,
                                task_id=task.id,
                                item_id=item.id,
                                dataset_id=item.dataset.id))
        all_scores.append(Score(type=ScoreType.ANNOTATION_LABEL,
                                value=mean_or_default(arr=[score.value for score in annotation_frame_scores if
                                                           score.type == ScoreType.ANNOTATION_LABEL.value],
                                                      default=1),
                                entity_id=annotation_id,
                                task_id=task.id,
                                item_id=item.id,
                                dataset_id=item.dataset.id))
        all_scores.append(Score(type=ScoreType.ANNOTATION_IOU,
                                value=mean_or_default(arr=[score.value for score in annotation_frame_scores if
                                                           score.type == ScoreType.ANNOTATION_IOU.value],
                                                      default=1),
                                entity_id=annotation_id,
                                task_id=task.id,
                                item_id=item.id,
                                dataset_id=item.dataset.id))
        all_scores.append(Score(type=ScoreType.ANNOTATION_ATTRIBUTE,
                                value=mean_or_default(arr=[score.value for score in annotation_frame_scores if
                                                           score.type == ScoreType.ANNOTATION_ATTRIBUTE.value],
                                                      default=1),
                                entity_id=annotation_id,
                                task_id=task.id,
                                item_id=item.id,
                                dataset_id=item.dataset.id))

    return all_scores
