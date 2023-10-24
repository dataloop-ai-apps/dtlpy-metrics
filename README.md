# Scoring and metrics app


## Description

This app improves the efficiency of annotating data by improving annotation quality while reducing the time required to
produce them.

The components of this app are:

1. Functions to calculate scores for quality tasks and model predictions.
2. Custom nodes that can be added to pipelines to calculate scores when a task's quality items are completed.

Also check [this notebook](docs/metrics.ipynb) for more information

## Quality Flows
To understand more about each of these tasks, refer to the main Dataloop documentation linked:

1. [Qualification tasks](https://dataloop.ai/docs/qualification-honeypot)
2. [Honeypot tasks](https://dataloop.ai/docs/qualification-honeypot#honeypot)
3. [Consensus tasks](https://dataloop.ai/docs/consensus)


In general, an annotator will receive an assignment to complete their annotation task. For a given item in a consensus task, 
each assignment will be cross-compared with every other assignment. In the case of qualification and honeypot tasks, 
each item will only have one assignment associated with it. 

### What's Supported?

Supported file types:
- image
- video

Scoring is currently supported for quality tasks with the following annotation types (with geometry score method in parentheses, where applicable):
- classification
- bounding box (IOU)
- polygon (IOU)
- segmentation (IOU)
- point (distance)


## Score Types

During scoring, the following scores will be created for each annotation:

- `raw_annotation_scores` -  e.g. geometry, label, attribute
- `annotation_overall` - the mean of each annotation’s raw scores
- `user_confusion_score` - the mean of every annotation overall score, relative to ref or another assignee
- `item_confusion_score` - the count of the number of label pairs associated with the assignee’s label, relative to the reference’s label
- `item_overall_score` - the mean value of *each* annotation overall score associated with an item

**1) Raw annotation scores:** 

There are three types of scores for annotations: geometry (such as IOU), label, and attribute. These scores can be determined by the user, and the default is to include all three scores, and the default value is 1 (which can be modified).

**2) Annotation overall**

This is the mean value for all raw annotation scores per annotation. 

**3) User confusion score**

The value of this score represents the mean annotation score a given assignee has, relative to raw scores when comparing it to another set of annotations (either the reference or another assignee). 

**4) Item confusion score**

The value of this score represents the count for a label annotated by a given assignee, relative to label each label class in the other set of annotations (either reference or another assignee).

**5) Item overall score**

This is the mean value of all annotations associated with an item, averaging the mean overall annotation score.

Any calculated and uploaded scores will replace any previous scores for all items of a given task.

_Note about videos_: Video scores will differ slightly from image scores. Video sores are calculated frame by frame, and then specific annotation scores will be the average of these scores across all relevant frames for that specific annotation. Confusion scores are not calculated due to the multi-frame nature of videos. Item overall scores remain an average of all annotations of the video item.

## Confusion Example

There are generally two kinds of scores: regular scores, and “confusion” scores. 

Regular scores show the level of agreement or overlap between two sets of annotations. They use the ID of the entities being compared for the `entityID` and `relative` fields. This can be for comparing annotations or items. `value` will typically be a number between 0 and 1. 


There are two types of confusion scores: item label confusion, and user confusion. **Item label confusion** shows the number of instances in which an assignee’s label corresponds with the ground truth labels. 

_Ground truth annotations_:

![Cat v dog](assets/cat_dog_annotations_1.png)

`item = dl.items.dl(item_id='64c0fc0730b03f27ca3a58db')`

_Assignee annotations_:

![Cat v dog](assets/cat_dog_annotations_2.png)

`item = dl.items.dl(item_id='64c0f2e1ec9103d52eaedbe2')`


In this example item, the ground truth has 3 for each cat and dog class. The assignee however, labels 1 as cat and 5 as dog. This would result in the following item label confusion scores:

```python
{
        "type": "label_confusion",
        "value": 1,
        "entityId": "cat",
        "context": {
            "relative": "cat",
            "taskId": "<TASK_ID>",
            "itemId": "<ITEM_ID">,
            "datasetId": "<DATASET_ID>"
        }
},
{
        "type": "label_confusion",
        "value": 3,
        "entityId": "dog",
        "context": {
            "relative": "dog",
            "taskId": "<TASK_ID>",
            "itemId": "<ITEM_ID">,
            "datasetId": "<DATASET_ID>"
        }
},
{
        "type": "label_confusion",
        "value": 2,
        "entityId": "dog",
        "context": {
            "relative": "cat",
            "taskId": "<TASK_ID>",
            "itemId": "<ITEM_ID">,
            "datasetId": "<DATASET_ID>"
        }
}
```

## Python installation

```shell
pip install dtlpymetrics
```

## Functions

See [this page](docs/dtlpymetrics_fxns.md) for details on additional functions.

## Contributions, Bugs and Issues - How to Contribute

We welcome anyone to help us improve this app.  
[Here](CONTRIBUTING.md) are detailed instructions to help you open a bug or ask for a feature request
