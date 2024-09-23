# Functions available in dtlpymetrics:

* `calculate_task_score`
    * Description: This function takes calculates scores for the relevant items contained within the task.
    * Input:
        * `task`: the relevant task entity
        * `score_types`: optional list for specifying which scores are to be calculated
    * Output:
        * `task`: the original task entity

* `create_task_item_score`
    * Description: This function takes items from a quality task and calculates annotations scores for a given item,
      including overall annotation scores and the overall item score.

      Scores calculated will vary depending on the item type. For example, images will have individual annotation
      scores, overall annotation scores, label confusion scores, and user confusion scores. Videos will first calculate
      scores frame by frame, and then annotation scores will be the average across all frames where the annotation is
      present.

    * Input:
        * `item`: item whose annotations are to be scored
        * `task`: optional task entity for the task that the item belongs to
        * `context`: optional context dictionary for the relevant IDs associated with the item
        * `score_types`: optional list for specifying which scores are to be calculated
    * Output:
        * `item`: the original item entity

* `create_model_score`
    * Description: This function takes model predictions (i.e. annotations) and calculates the score for each
      prediction.
    * Input:
        * `dataset`: the dataset entity containing the model predictions
        * `model`: the model entity that produced the predictions
        * `filters`: optional DQL filter for selecting the relevant dataset items
        * `ignore_labels`: optional boolean for whether to ignore the labels in the model predictions
        * `match_threshold`: optional float for IOU threshold for matching model predictions to ground truth annotations
        * `compare_types`: optional list for specifying which annotation types are to be compared


* `get_model_scores_df`
    * Description: This function generates the pandas dataframe of model scores.
    * Input:
        * `dataset`: the dataset entity containing the test items that were predicted on
        * `model`: the model entity that produced the predictions
    * Output:
        * `model_scores_df`: pandas dataframe containing the scores

* `get_image_scores`
    * Description: This function takes items that are images and calculates the scores.
    * Input:
      * `item`: item whose annotations are to be scored
      *

* `get_video_scores`
* `check_if_video`
* `plot_matrix`
* `add_score_context`

* `calculate_annotation_score`
    * Description: This function takes annotations and calculates the score for each annotation.
    * Input:
        * `annot_collection_1`: set 1 of a dl.AnnotationCollection or list of annotations
        * `annot_collection_2`: set 2 of a dl.AnnotationCollection or list of annotations
        * `ignore_labels`: optional boolean for whether to ignore the labels in the model predictions
        * `include_confusion`: optional boolean for whether to include confusion scores in the output
        * `match_threshold`: optional float for IOU threshold for matching model predictions to ground truth annotations
        * `compare_types`: optional list for specifying which annotation types are to be compared
        * `score_types`: optional list for specifying which scores are to be calculated

