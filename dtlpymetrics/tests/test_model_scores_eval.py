import json
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

        if model.status == 'trained':
            print('deploying model service and predicting')
            model.deploy()
        model.predict(items_ids)
        model.status = 'trained'
        model.update()


def check_item(model: dl.Model, item: dl.Item):
    annotations = item.annotations.list()
    for annotation in annotations:
        if annotation.metadata.get('user', {}).get('model', {}).get('model_id', None) == model.id:
            print('already predicted')
            return True
    return False


if __name__ == "__main__":
    import dtlpy as dl
    ENV = 'prod'
    dl.setenv(ENV)

    if ENV == 'rc':
        project = dl.projects.get('Active Learning 1.3')
        dataset = project.datasets.get('model eval test')
        model = project.models.get(None, '6481f19bf18d2526d10af94c')
    elif ENV == 'prod':
        project = dl.projects.get('CVPR 2023 Demo')
        dataset = project.datasets.get('Ground Truth for Active Learning')
        model = project.models.get(None, '6490054acb7eb972681fe982')

    test_filter = '{"filter":{"$and":[{"hidden":false},{"$or":[{"metadata":{"system":{"tags":{"train":true}}}}]},{"type":"file"}]},"page":0,"pageSize":1000,"resource":"items"}'
    filters = dl.Filters(custom_filter=json.loads(test_filter))
    print(dataset.items.list(filters=filters).items_count)


    # create predictions on the test set for this model
    items_list = list(dataset.items.list(filters=filters).all())
    items_ids = [item.id for item in items_list]

    # check_predictions(model=model)

    ########################################
    # Evaluate the model and create scores #
    ########################################

    success, message = ScoringAndMetrics.create_model_score(dataset=dataset,
                                                            filters=filters,
                                                            model=model,
                                                            match_threshold=0.5,
                                                            # ignore_labels=True,
                                                            compare_types=model.output_type)
    print(message)

    model_matches = ScoringAndMetrics.get_model_scores_df(model=model, dataset=dataset)
    metric_names = ['accuracy', 'iou', 'confidence']

    plot_points = ScoringAndMetrics.calc_precision_recall(dataset_id=dataset.id,
                                                          model_id=model.id,
                                                          conf_threshold=0.2)

    save_path = ScoringAndMetrics.plot_precision_recall(plot_points)

    print()
