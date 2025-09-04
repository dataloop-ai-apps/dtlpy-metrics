from typing import List
import logging
import dtlpy as dl
import numpy as np

from ..dtlpy_scores import Score, ScoreType, Scores
from ..utils import mean_or_default, add_score_context, check_if_video, calculate_annotation_score
# Import kappa functions locally to avoid circular imports

logger = logging.getLogger("scoring-and-metrics")


def calc_task_score(task: dl.Task, score_types=None, upload=False) -> dict:
    """
    Calculate scores for all items in a quality task, based on the item scores from each assignment.

    :param task: dl.Task entity
    :param score_types: optional list of ScoreTypes to calculate (e.g. [ScoreType.ANNOTATION_IOU, ScoreType.ANNOTATION_LABEL])
    :param upload: bool, default False means scores will be saved locally (optional)
    :return: dict of scores with item id as key and list of Scores as value
    """
    # determine task type
    if task.metadata["system"].get("consensusTaskType") not in ["qualification", "honeypot", "consensus"]:
        raise ValueError("Task type is not suitable for scoring")

    if task.metadata["system"].get("consensusTaskType") in ["honeypot", "qualification"]:
        # qualification and honeypot scoring is completed on cloned items for each assignee
        filters = dl.Filters()
        filters.add(field="hidden", values=True)  # return only the clones
        pages = task.get_items(filters=filters)

    else:  # for consensus
        # default filter for quality tasks is hidden = True, so this hides the clones and returns only original items
        filters = dl.Filters()
        filters.add(field="hidden", values=False)
        pages = task.get_items(filters=filters, get_consensus_items=True)

    items_scores = dict()
    for item in pages.all():
        item_refs = item.metadata["system"]["refs"]
        current_task_done = False
        for ref in item_refs:
            if ref["id"] != task.id:
                continue
            # for testing tasks, check if the item is complete via metadata
            elif ref.get("metadata", None) is None:
                continue
            elif ref.get("metadata").get("status", None) in ["completed", "consensus_done"]:
                current_task_done = True
                break
            else:
                logger.info(f"Item {item.id} is not complete, skipping scoring")
                continue
        if current_task_done is True:
            items_scores[item.id] = calc_task_item_score(item=item, task=task, score_types=score_types, upload=upload)

    return items_scores


def calc_task_item_score(item: dl.Item, task: dl.Task, score_types=None, upload=True) -> List[Score]:
    """
    Create scores for items in a task. This is the main function for creating score entities

    In the case of qualification and honeypot, the first set of annotations is considered the reference set.
    In the case of consensus, annotations are compared twice-- once as a reference set, and once as a test set.
    :param item: dl.Item entity
    :param task: dl.Task entity
    :param score_types: list of ScoreTypes to calculate (e.g. [ScoreType.ANNOTATION_IOU, ScoreType.ANNOTATION_LABEL]) (optional)
    :param upload: bool, default True means scores will be uploaded to the platform (optional)
    :return: list of Scores
    """
    logger.info(f"Starting scoring for item: {item.id} and task: {task.id}")
    if task.metadata["system"].get("consensusTaskType") == "consensus":
        task_type = "consensus"
    elif task.metadata["system"].get("consensusTaskType") in ["qualification", "honeypot"]:
        task_type = "testing"
    else:
        raise ValueError("Task type is not suitable for scoring.")

    ###################################
    # collect assignments for sorting #
    ###################################
    assignments = task.assignments.list()

    is_quality_task = _check_task_type(task_type=task_type, task=task, item=item)

    if is_quality_task is False:
        logging.info("Item was not annotated via quality task. No scores were created.")
        all_scores = list()
    else:
        [assignments_by_id, assignments_by_annotator] = _sort_assignments(
            task_type=task_type, item=item, assignments=assignments
        )
        # if no assignments are associated with this item
        if len(assignments_by_id) == 0:
            raise ValueError(
                f"No assignments found for task {task.id} and item {item.id}. Please check that the task was properly "
                f"configured and completed."
            )

        #########################################
        # sort annotations and calculate scores #
        #########################################
        annotations = item.annotations.list()
        annotators_list = [assignment.annotator for assignment in assignments_by_id.values()]
        logger.info(f"Starting scoring for assignments: {annotators_list}")

        is_video = check_if_video(item=item)
        if is_video is True:  # video items
            annotations_by_frame = _split_video_to_frames(
                annotations=annotations, item=item, task=task, task_type=task_type, assignments_by_id=assignments_by_id
            )
            all_scores = get_video_scores(
                annotations_by_frame=annotations_by_frame,
                assignments_by_annotator=assignments_by_annotator,
                task=task,
                item=item,
                score_types=score_types,
                task_type=task_type,
            )
        else:  # image items
            annots_by_assignment = _sort_annotations(
                task=task, item=item, task_type=task_type, annotations=annotations, assignments_by_id=assignments_by_id
            )
            all_scores = get_image_scores(
                annots_by_assignment=annots_by_assignment,
                assignments_by_annotator=assignments_by_annotator,
                task=task,
                item=item,
                score_types=score_types,
                task_type=task_type,
            )

        # overall item score is an average of all overall annotation scores
        item_overall = [score.value for score in all_scores if score.type == ScoreType.ANNOTATION_OVERALL.value]

        item_score = Score(
            type=ScoreType.ITEM_OVERALL,
            value=mean_or_default(arr=item_overall, default=1),
            entity_id=item.id,
            task_id=task.id,
            item_id=item.id,
            dataset_id=item.dataset.id,
        )
        all_scores.append(item_score)

        # calculate kappa scores from annotations grouped by item
        kappa_scores = calc_item_kappa_scores(all_scores, item.id, task.id, item.dataset.id)
        all_scores.extend(kappa_scores)

        #############################
        # upload scores to platform #
        #############################
        if upload is True:
            logger.info(f"Deleting all scores with context item ID: {item.id} and task ID: {task.id}")
            dl_scores = Scores(client_api=dl.client_api)
            dl_scores.delete(context={"itemId": item.id, "taskId": task.id})
            dl_scores = dl_scores.create(all_scores)
            logger.info(f"Uploaded {len(dl_scores)} scores to platform.")
    return all_scores


def _check_task_type(task_type, task: dl.Task, item: dl.Item) -> bool:
    run_fxn = False
    if task_type == "consensus":
        consensus_assignment = task.metadata["system"]["consensusAssignmentId"]
        all_item_refs = item.metadata["system"]["refs"]
        if consensus_assignment is not None:
            for ref_dict in all_item_refs:
                if ref_dict["id"] == consensus_assignment:
                    run_fxn = True
    else:
        if item.dir.startswith("/.consensus/"):
            run_fxn = True

    return run_fxn


def _sort_assignments(task_type: str, item: dl.Item, assignments: list) -> list:
    if task_type == "consensus":
        # get only assignments that had a consensus for this item:
        filters = dl.Filters(use_defaults=False)
        filters.add("spec.parentDatasetItemId", item.id)
        filters.add("dir", "/.consensus/*")
        asg_pages = item.dataset.items.list(filters=filters)
        consensus_assignments = list()
        for asg_items in asg_pages.all():
            refs = asg_items.metadata["system"]["refs"]
            for ref in refs:
                if ref.get("type") == "assignment":
                    consensus_assignments.append(ref["id"])
    else:
        consensus_assignments = [asg.id for asg in assignments]

    # create lookup dictionaries getting assignments by id or annotator
    assignments_by_id = {}
    assignments_by_annotator = {}
    for assignment in assignments:
        if assignment.id in consensus_assignments:
            assignments_by_id[assignment.id] = assignment
            assignments_by_annotator[assignment.annotator] = assignment

    return [assignments_by_id, assignments_by_annotator]


def _sort_annotations(
    task: dl.Task, item: dl.Item, task_type: str, annotations: List[dl.Annotation], assignments_by_id: dict
) -> dict:
    # group by some field (e.g. 'creator' or 'assignment id'), here we use assignment id
    annots_by_assignment = {assignment.annotator: [] for assignment in assignments_by_id.values()}
    for annotation in annotations:
        # default is "ref"
        # TODO handle models
        assignment_id = annotation.metadata["system"].get("assignmentId", "ref")
        task_id = annotation.metadata["system"].get("taskId", None)
        if task_id == task.id:
            assignment_annotator = assignments_by_id[assignment_id].annotator
            annots_by_assignment[assignment_annotator].append(annotation)
        else:
            # TODO comparing annotations from another task
            continue

    # add in reference annotations if testing task
    if task_type == "testing":
        # get all ref from the src item
        src_item = dl.items.get(item_id=item._src_item)
        annots_by_assignment["ref"] = src_item.annotations.list()
    return annots_by_assignment


def _split_video_to_frames(
    annotations: dl.AnnotationCollection, item: dl.Item, task: dl.Task, task_type: str, assignments_by_id: dict
) -> dict:
    """
    Hidden function to split video annotations frame by frame and sort by assignment
    :param annotations:
    :param item:
    :param task:
    :param task_type:
    :param assignments_by_id:
    :return:
    """
    # get max frames for all annotations
    try:
        num_frames = int(item.metadata["system"]["ffmpeg"]["nb_read_frames"])
    except KeyError:
        end_frames = [ann.end_frame for ann in annotations]
        num_frames = np.max(end_frames) + 1

    all_annotation_slices = dict()
    for f in range(num_frames):
        all_annotation_slices[f] = annotations.get_frame(frame_num=f)
    annotations_by_frame = {}

    # within each frame, sort all annotation slices to their corresponding assignment/annotator
    for frame, annotation_slices in all_annotation_slices.items():
        frame_annots_by_assignment = {assignment.annotator: [] for assignment in assignments_by_id.values()}
        for annotation_slice in annotation_slices:
            # TODO compare annotations between models
            # default is "ref", if no assignment ID is found
            assignment_id = annotation_slice.metadata["system"].get("assignmentId", "ref")
            task_id = annotation_slice.metadata["system"].get("taskId", None)
            if task_id == task.id:
                assignment_annotator = assignments_by_id[assignment_id].annotator
                frame_annots_by_assignment[assignment_annotator].append(annotation_slice)
            else:
                # TODO comparing annotations from another task
                continue
        annotations_by_frame[frame] = frame_annots_by_assignment

    # add in reference annotations if testing task
    if task_type == "testing":
        # get all ref from the src item
        src_item = dl.items.get(item_id=item._src_item)
        # ref_annots_by_frame = get_annotations_from_frames(src_item.annotations.list())
        ref_annotations = src_item.annotations.list()
        num_frames = src_item.metadata["system"]["nb_frames"]
        for f in range(num_frames):
            annotations_by_frame[f]["ref"] = ref_annotations.get_frame(frame_num=f)
            # annotations_by_frame[frame]['ref'] = annotation_slices
    return annotations_by_frame


def get_image_scores(
    annots_by_assignment: dict,
    assignments_by_annotator: dict,
    item: dl.Item,
    task: dl.Task = None,
    task_type: str = None,
    score_types: list = None,
) -> list:
    """
    Calculate scores for an image item

    :param annots_by_assignment: dict of annotations grouped by assignment
    :param assignments_by_annotator: dict of assignments groups by annotator
    :param item: dl.Item
    :param task: dl.Task
    :param task_type: str, 'testing' or 'consensus'
    :param score_types: list of score types to be calculated (optional)
    :return all_scores: list of all annotation and item scores

    """
    ####################
    # calculate scores #
    ####################
    # compare between each assignment and create Score entities
    all_scores = list()

    # do pairwise comparisons of each assignment for all annotations on the item
    for i_assignment, assignment_annotator_i in enumerate(annots_by_assignment):
        if task_type == "testing" and assignment_annotator_i != "ref":
            # if "testing", compare only to ref
            continue
        for j_assignment, assignment_annotator_j in enumerate(annots_by_assignment):
            # don't compare a set to itself
            if i_assignment == j_assignment:
                continue
            # skip ref in inner loop
            if assignment_annotator_j == "ref":
                continue
            logger.info(f"Comparing assignee: {assignment_annotator_i!r} with assignee: {assignment_annotator_j!r}")
            annot_collection_1 = annots_by_assignment[assignment_annotator_i]
            annot_collection_2 = annots_by_assignment[assignment_annotator_j]
            # score types that can be returned: ANNOTATION_IOU, ANNOTATION_LABEL, ANNOTATION_ATTRIBUTE
            pairwise_scores = calculate_annotation_score(
                annot_collection_1=annot_collection_1,
                annot_collection_2=annot_collection_2,
                ignore_labels=False,
                ignore_attributes=True,
                ignore_geometry=True,
                match_threshold=0.01,
                score_types=score_types,
            )
            for score in pairwise_scores:
                if score.type == ScoreType.USER_CONFUSION:
                    updated_score = add_score_context(
                        score=score,
                        user_id=assignment_annotator_j,
                        task_id=task.id,
                        entity_id=assignment_annotator_j,
                        relative=assignment_annotator_i,
                        assignment_id=assignments_by_annotator[assignment_annotator_j].id,
                        item_id=item.id,
                    )
                else:
                    updated_score = add_score_context(
                        score=score,
                        user_id=assignment_annotator_j,
                        task_id=task.id,
                        assignment_id=assignments_by_annotator[assignment_annotator_j].id,
                        item_id=item.id,
                    )
                all_scores.append(updated_score)
    # accumulate label confusion for all compares
    confusion_scores = list()
    for i_score, score in reversed(list(enumerate(all_scores))):
        if score.type == ScoreType.LABEL_CONFUSION:
            confusion_scores.append(score)
            all_scores.pop(i_score)
    confusion_dict = dict()
    for score in confusion_scores:
        if score.entity_id not in confusion_dict:
            confusion_dict[score.entity_id] = dict()
        if score.relative not in confusion_dict[score.entity_id]:
            confusion_dict[score.entity_id][score.relative] = 0
        confusion_dict[score.entity_id][score.relative] += 1
    for entity_id, v in confusion_dict.items():
        for relative, count in v.items():
            all_scores.append(
                Score(
                    type=ScoreType.LABEL_CONFUSION,
                    value=count,
                    entity_id=entity_id,  # assignee label
                    relative=relative,
                    task_id=task.id,
                    item_id=item.id,
                )
            )

    # mean over all ANNOTATION_OVERALL for each annotation id
    annotation_overalls = list()
    for i_score, score in reversed(list(enumerate(all_scores))):
        if score.type == ScoreType.ANNOTATION_OVERALL:
            annotation_overalls.append(score)
            all_scores.pop(i_score)

    unique_annotation_ids = np.unique([score.entity_id for score in annotation_overalls])
    for annotation_id in unique_annotation_ids:
        overalls = [score for score in annotation_overalls if score.entity_id == annotation_id]
        # this is a matching score between annotations to make it a probability we will add the current self match as 1
        # for instance, if we had [A,A,B], and the current is A, the overall probability is 2/3
        overalls_values = [s.value for s in overalls]
        overalls_values.append(1)  # the match to the current annotation, this will it the probability
        user_id = overalls[0].user_id
        assignment_id = overalls[0].assignment_id
        # add joint overall (single one for each annotation
        all_scores.append(
            Score(
                type=ScoreType.ANNOTATION_OVERALL,
                value=mean_or_default(arr=overalls_values, default=0),
                entity_id=annotation_id,
                user_id=user_id,
                task_id=task.id,
                assignment_id=assignment_id,
                item_id=item.id,
            )
        )
    return all_scores


def get_video_scores(
    annotations_by_frame: dict,
    assignments_by_annotator: dict,
    item: dl.Item,
    task: dl.Task = None,
    score_types: list = None,
    task_type: str = None,
):
    """
    Create scores for a video item

    :param annotations_by_frame: dict
    :param assignments_by_annotator: dict
    :param task: dl.Task
    :param item: dl.Item
    :param score_types: list of scores
    :param task_type: str
    :return all_scores: list of all annotation and item scores
    """
    ####################
    # calculate scores #
    ####################
    all_scores_by_frame = dict()
    ann_ids = list()
    for frame, annots_by_assignment in annotations_by_frame.items():
        # compare between each assignment and create Score entities
        frame_scores = list()

        # do pairwise comparisons of each assignment for all annotations on the item
        for i_assignment, assignment_annotator_i in enumerate(annots_by_assignment):
            if task_type == "testing" and assignment_annotator_i != "ref":
                # if "testing", compare only to ref
                continue
            for j_assignment, assignment_annotator_j in enumerate(annots_by_assignment):
                # don't compare a set to itself
                if i_assignment == j_assignment:
                    continue
                # skip ref in inner loop
                if assignment_annotator_j == "ref":
                    continue
                logger.info(f"Comparing assignee: {assignment_annotator_i!r} with assignee: {assignment_annotator_j!r}")
                annot_collection_1 = annots_by_assignment[assignment_annotator_i]
                annot_collection_2 = annots_by_assignment[assignment_annotator_j]
                ann_ids.extend([ann.id for ann in annot_collection_1])
                ann_ids.extend([ann.id for ann in annot_collection_2])
                # score types that can be returned: ANNOTATION_IOU, ANNOTATION_LABEL, ANNOTATION_ATTRIBUTE
                pairwise_scores = calculate_annotation_score(
                    annot_collection_1=annot_collection_1,
                    annot_collection_2=annot_collection_2,
                    ignore_labels=False,
                    ignore_attributes=True,
                    ignore_geometry=True,
                    match_threshold=0.01,
                    score_types=score_types,
                )
                for score in pairwise_scores:
                    if score.type == ScoreType.USER_CONFUSION:
                        updated_score = add_score_context(
                            score=score,
                            user_id=assignment_annotator_j,
                            task_id=task.id,
                            entity_id=assignment_annotator_j,
                            relative=assignment_annotator_i,
                            assignment_id=assignments_by_annotator[assignment_annotator_j].id,
                            item_id=item.id,
                        )
                    else:
                        updated_score = add_score_context(
                            score=score,
                            user_id=assignment_annotator_j,
                            task_id=task.id,
                            assignment_id=assignments_by_annotator[assignment_annotator_j].id,
                            item_id=item.id,
                        )
                    frame_scores.append(updated_score)
        all_scores_by_frame[frame] = frame_scores

    # once each frame's score is calculated, take the average score of all frames
    all_scores = list()
    unique_annotation_ids = np.unique(ann_ids)

    confusion_scores = list()
    for frame, scores in all_scores_by_frame.items():
        for score in scores:
            if score.type == ScoreType.LABEL_CONFUSION:
                confusion_scores.append(score)

    confusion_dict = dict()
    for score in confusion_scores:
        if score.entity_id not in confusion_dict:
            confusion_dict[score.entity_id] = dict()
        if score.relative not in confusion_dict[score.entity_id]:
            confusion_dict[score.entity_id][score.relative] = 0
        confusion_dict[score.entity_id][score.relative] += 1
    for entity_id, v in confusion_dict.items():
        for relative, count in v.items():
            all_scores.append(
                Score(
                    type=ScoreType.LABEL_CONFUSION,
                    value=count,
                    entity_id=entity_id,  # assignee label
                    relative=relative,
                    task_id=task.id,
                    item_id=item.id,
                )
            )

    for annotation_id in unique_annotation_ids:
        annotation_frame_scores = [
            frame_score
            for frame_scores in all_scores_by_frame.values()
            for frame_score in frame_scores
            if frame_score.entity_id == annotation_id
        ]
        all_scores.append(
            Score(
                type=ScoreType.ANNOTATION_OVERALL,
                value=mean_or_default(
                    arr=[
                        score.value
                        for score in annotation_frame_scores
                        if score.type == ScoreType.ANNOTATION_OVERALL.value
                    ],
                    default=1,
                ),
                entity_id=annotation_id,
                task_id=task.id,
                item_id=item.id,
                dataset_id=item.dataset.id,
            )
        )
        all_scores.append(
            Score(
                type=ScoreType.ANNOTATION_LABEL,
                value=mean_or_default(
                    arr=[
                        score.value
                        for score in annotation_frame_scores
                        if score.type == ScoreType.ANNOTATION_LABEL.value
                    ],
                    default=1,
                ),
                entity_id=annotation_id,
                task_id=task.id,
                item_id=item.id,
                dataset_id=item.dataset.id,
            )
        )
        all_scores.append(
            Score(
                type=ScoreType.ANNOTATION_IOU,
                value=mean_or_default(
                    arr=[
                        score.value for score in annotation_frame_scores if score.type == ScoreType.ANNOTATION_IOU.value
                    ],
                    default=1,
                ),
                entity_id=annotation_id,
                task_id=task.id,
                item_id=item.id,
                dataset_id=item.dataset.id,
            )
        )
        all_scores.append(
            Score(
                type=ScoreType.ANNOTATION_ATTRIBUTE,
                value=mean_or_default(
                    arr=[
                        score.value
                        for score in annotation_frame_scores
                        if score.type == ScoreType.ANNOTATION_ATTRIBUTE.value
                    ],
                    default=1,
                ),
                entity_id=annotation_id,
                task_id=task.id,
                item_id=item.id,
                dataset_id=item.dataset.id,
            )
        )

    return all_scores


def calc_item_kappa_scores(all_scores: List[Score], item_id: str, task_id: str, dataset_id: str) -> List[Score]:
    """
    Calculate Cohen's and Fleiss' kappa scores for an item from existing annotation scores.
    
    For exactly 2 annotators: Calculate Cohen's kappa
    For 3+ annotators: Calculate Fleiss' kappa (and optionally pairwise Cohen's kappa)
    
    :param all_scores: List of existing scores from calc_task_item_score
    :param item_id: Item ID
    :param task_id: Task ID  
    :param dataset_id: Dataset ID
    :return: List of kappa Score objects
    """
    kappa_scores = []
    
    try:
        # Get unique annotators from scores
        unique_annotators = set()
        for score in all_scores:
            if score.type == ScoreType.ANNOTATION_OVERALL:
                annotator_id = score.context.get('assignmentId') or score.user_id
                if annotator_id:
                    unique_annotators.add(annotator_id)
        
        unique_annotators = list(unique_annotators)
        num_annotators = len(unique_annotators)
        
        if num_annotators < 2:
            logger.info(f"Insufficient annotators ({num_annotators}) for kappa calculation on item {item_id}")
            return kappa_scores
            
        # For exactly 2 annotators: Use Cohen's kappa
        elif num_annotators == 2:
            try:
                annotator1, annotator2 = unique_annotators
                kappa_value = _calculate_cohens_kappa(all_scores, annotator1, annotator2)
                
                cohen_score = Score(
                    type=ScoreType.USER_CONFUSION,  # Reuse existing type for kappa between users
                    value=kappa_value,
                    entity_id=f"{item_id}_cohens_kappa",
                    task_id=task_id,
                    item_id=item_id,
                    dataset_id=dataset_id,
                    context={
                        'metric': 'cohens_kappa',
                        'annotator1': annotator1,
                        'annotator2': annotator2,
                        'itemId': item_id,
                        'taskId': task_id
                    }
                )
                kappa_scores.append(cohen_score)
                logger.info(f"Calculated Cohen's kappa ({kappa_value:.3f}) for 2 annotators on item {item_id}")
                
            except (ValueError, ZeroDivisionError) as e:
                logger.warning(f"Could not calculate Cohen's kappa for item {item_id}: {e}")
        
        # For 3+ annotators: Use Fleiss' kappa
        else:  # num_annotators >= 3
            try:
                fleiss_value = _calculate_fleiss_kappa(all_scores)
                
                fleiss_score = Score(
                    type=ScoreType.ITEM_OVERALL,  # Item-level agreement metric
                    value=fleiss_value,
                    entity_id=f"{item_id}_fleiss_kappa",
                    task_id=task_id,
                    item_id=item_id,
                    dataset_id=dataset_id,
                    context={
                        'metric': 'fleiss_kappa',
                        'num_annotators': num_annotators,
                        'annotators': unique_annotators,
                        'itemId': item_id,
                        'taskId': task_id
                    }
                )
                kappa_scores.append(fleiss_score)
                logger.info(f"Calculated Fleiss' kappa ({fleiss_value:.3f}) for {num_annotators} annotators on item {item_id}")
                
            except (ValueError, ZeroDivisionError) as e:
                logger.warning(f"Could not calculate Fleiss' kappa for item {item_id}: {e}")
        
    except Exception as e:
        logger.error(f"Error calculating kappa scores for item {item_id}: {e}")
    
    return kappa_scores


def _calculate_cohens_kappa(scores, annotator1_id: str, annotator2_id: str) -> float:
    """
    Calculate Cohen's kappa for agreement between two specific annotators.
    Simplified version of the function from evaluating.tasks to avoid circular imports.
    """
    from collections import defaultdict
    
    # Extract annotation scores for each annotator
    annotator1_scores = {}
    annotator2_scores = {}

    for score in scores:
        if score.type == ScoreType.ANNOTATION_OVERALL:
            # Use assignment ID if available, otherwise use user ID
            annotator_id = score.context.get('assignmentId') or score.user_id
            annotation_id = score.entity_id

            if annotator_id == annotator1_id:
                annotator1_scores[annotation_id] = score.value
            elif annotator_id == annotator2_id:
                annotator2_scores[annotation_id] = score.value

    # Find common annotations rated by both annotators
    common_annotations = set(annotator1_scores.keys()) & set(annotator2_scores.keys())

    if len(common_annotations) == 0:
        raise ValueError("No common annotations found between the two annotators")

    # Convert scores to binary categories (agree/disagree) based on threshold
    threshold = 0.5

    # Create agreement matrix
    agreements = []
    
    for annotation_id in common_annotations:
        score1 = annotator1_scores[annotation_id] >= threshold
        score2 = annotator2_scores[annotation_id] >= threshold

        if score1 == score2:
            agreements.append(1)
        else:
            agreements.append(0)

    # Calculate observed agreement
    po = sum(agreements) / len(agreements)

    # Calculate expected agreement by chance
    annotator1_passes = sum(1 for aid in common_annotations if annotator1_scores[aid] >= threshold)
    annotator2_passes = sum(1 for aid in common_annotations if annotator2_scores[aid] >= threshold)

    p1_pass = annotator1_passes / len(common_annotations)
    p2_pass = annotator2_passes / len(common_annotations)

    p1_fail = 1 - p1_pass
    p2_fail = 1 - p2_pass

    # Expected agreement by chance
    pe = (p1_pass * p2_pass) + (p1_fail * p2_fail)

    # Cohen's kappa
    if pe == 1.0:
        return 1.0  # Perfect agreement case

    kappa = (po - pe) / (1 - pe)
    return kappa


def _calculate_fleiss_kappa(scores) -> float:
    """
    Calculate Fleiss' kappa for agreement among multiple annotators.
    Simplified version of the function from evaluating.tasks to avoid circular imports.
    """
    from collections import defaultdict
    
    # Group scores by annotation and annotator
    annotation_ratings = defaultdict(list)
    annotators = set()

    # Use 0.5 as threshold for binary classification (agree/disagree)
    threshold = 0.5

    for score in scores:
        if score.type == ScoreType.ANNOTATION_OVERALL:
            # Use assignment ID if available, otherwise use user ID
            annotator_id = score.context.get('assignmentId') or score.user_id
            annotation_id = score.entity_id

            # Convert score to binary rating
            rating = 1 if score.value >= threshold else 0

            annotation_ratings[annotation_id].append((annotator_id, rating))
            annotators.add(annotator_id)

    if len(annotators) < 2:
        raise ValueError("Fleiss' kappa requires at least 2 annotators")

    # Filter annotations that have ratings from multiple annotators
    valid_annotations = {
        ann_id: ratings
        for ann_id, ratings in annotation_ratings.items()
        if len(set(r[0] for r in ratings)) >= 2  # At least 2 different annotators
    }

    if len(valid_annotations) == 0:
        raise ValueError("No annotations found with ratings from multiple annotators")

    # Create rating matrix: annotations x categories (agree/disagree)
    n_annotations = len(valid_annotations)
    n_categories = 2  # Binary: agree (1) or disagree (0)
    n_raters = len(annotators)

    # Count matrix: each cell [i,j] contains count of raters who assigned category j to annotation i
    rating_matrix = np.zeros((n_annotations, n_categories))

    for i, (annotation_id, ratings) in enumerate(valid_annotations.items()):
        # Count ratings for each category
        category_counts = [0, 0]  # [disagree_count, agree_count]

        for annotator_id, rating in ratings:
            category_counts[rating] += 1

        rating_matrix[i] = category_counts

    # Calculate proportion of all assignments to each category
    p_j = np.sum(rating_matrix, axis=0) / (n_annotations * n_raters)

    # Calculate P_i (extent of agreement for annotation i)
    P_i = np.zeros(n_annotations)
    for i in range(n_annotations):
        r_ij = rating_matrix[i]
        # Number of raters who actually rated this annotation
        n_i = np.sum(r_ij)
        if n_i > 1:
            P_i[i] = (np.sum(r_ij * (r_ij - 1))) / (n_i * (n_i - 1))
        else:
            P_i[i] = 0  # Cannot calculate agreement with only one rater

    # Mean of P_i values
    P_bar = np.mean(P_i)

    # Expected agreement by chance
    P_e = np.sum(p_j**2)

    # Fleiss' kappa
    if P_e == 1.0:
        return 1.0  # Perfect agreement case

    kappa = (P_bar - P_e) / (1 - P_e)
    return max(0.0, kappa)  # Ensure non-negative result
