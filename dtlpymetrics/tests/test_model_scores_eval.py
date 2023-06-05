import dtlpy as dl
from dtlpymetrics.scoring import ScoringAndMetrics


def create_yolo_clone():
    model_origin = dl.models.get(None, '6462417fcc4a02b5bea99e14')
    model = project.models.clone(from_model=model_origin,
                                 model_name='cloned_yolo_v8',
                                 project_id='f7d43fec-2823-4871-b0a0-1b76a75a2d61')
    model.deploy()


def check_predictions(model):
    # check if one of the test items has already been predicted by this model
    unpredicted = 0
    for item_id in items_ids:
        item = dl.items.get(None, item_id)
        if check_item(model, item):
            print('model has made predictions')
            break
        else:
            unpredicted += 1

    if unpredicted == len(items_ids):
        print('model has not made predictions')
        model.predict(items_ids)


def check_item(model: dl.Model, item: dl.Item):
    annotations = item.annotations.list()
    for annotation in annotations:
        if annotation.metadata.get('user', {}).get('model', {}).get('model_id', None) == model.id:
            print('already predicted')
            return True
    return False


if __name__ == "__main__":
    dl.setenv('rc')

    # resnet
    project = dl.projects.get('Active Learning 1.3')
    dataset = project.datasets.get('big cats TEST evaluate')
    filters = dl.Filters(field='dir', values='/test')

    model = project.models.get(None, '6473185c93bd97c6a30a47b9')  # resnet fine-tuned, deployed

    # create predictions on the test set for this model
    items_list = list(dataset.items.list(filters=filters).all())
    items_ids = [item.id for item in items_list]

    check_predictions(model=model)

    ########################################
    # Evaluate the model and create scores #
    ########################################

    success, message = ScoringAndMetrics.create_model_score(dataset=dataset,
                                                            filters=filters,
                                                            model=model,
                                                            match_threshold=0.5,
                                                            # ignore_labels=True,
                                                            compare_types=[dl.ANNOTATION_TYPE_CLASSIFICATION])
    print(message)

    model_scores = ScoringAndMetrics.get_scores_df(model=model, dataset=dataset)
    metric_names = ['accuracy', 'iou', 'confidence']
    ScoringAndMetrics.plot_precision_recall(scores=model_scores,
                                            metric=metric_names[0],
                                            metric_threshold=0.5)

    print()
