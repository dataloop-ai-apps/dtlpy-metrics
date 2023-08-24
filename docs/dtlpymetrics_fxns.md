# Functions available in dtlpymetrics:

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
* `calculate_task_score`
* `create_task_item_score`
* `create_model_score`
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

