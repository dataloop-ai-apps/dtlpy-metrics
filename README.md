# Scoring and metrics app

![Annotators confusion matrix](assets/annotators_matrix.png)

## Description

This app improves the efficiency of annotating data by improving annotation quality while reducing the time required to
produce them.

The components of this app are:

1. Functions to calculate scores for quality tasks and model predictions.
2. Custom nodes that can be added to pipelines to calculate scores when a task's quality items are completed.

Currently, there is support for scores on quality tasks (i.e qualification, honeypot, and consensus).

Annotation types supported include:
- 
- bounding boxes
- polygons


## Python installation

```shell
pip install git+https://github.com/dataloop-ai-apps/dtlpy-metrics
```

## Tutorial and How-To

See [this tutorial](docs/Quality task scoring tutorial.ipynb) for details on scoring for Dataloop "quality tasks" (
qualification, honeypot, and consensus tasks).

## Functions

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

See [this page](docs/dtlpymetrics_fxns.md) for details on additional functions.

## Contributions, Bugs and Issues - How to Contribute

We welcome anyone to help us improve this app.  
[Here](CONTRIBUTING.md) are detailed instructions to help you open a bug or ask for a feature request
