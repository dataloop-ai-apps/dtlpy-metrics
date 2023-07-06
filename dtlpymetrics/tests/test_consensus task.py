import dtlpy as dl
from dtlpymetrics.scoring import scorer

dl.setenv('rc')

########################
# Consensus task test ##
########################
# consensus_task = dl.tasks.get(task_name='pipeline consensus test (test tasks)')
consensus_task = dl.tasks.get(task_id='644a307ae052f434dab98ff3')
scorer.calculate_consensus_task_score(consensus_task)

if __name__ == '__main++__':
    import json

    # from Or's project, which has a precision recall plot to compare
    dl.setenv('new-dev')
    dataset_id = '648f5926943352ccaddf0149'
    model_id = '648ffafe28146328fb4e96b3'

    # # models to rerun and upload: 649076c45a9c968a5c32ed65, 6490b666d8a0841b563176f6

    test_filter = '{"filter": {"$and": [{"hidden": false}, {"$or": [{"metadata": {"system": {"tags": {"test": true}}}}]}, {"type": "file"}]}, "page": 0, "pageSize": 1000, "resource": "items"}'
    filters = dl.Filters(custom_filter=json.loads(test_filter))
    filters = dl.Filters()

    model = dl.models.get(None, model_id)
    dataset = dl.datasets.get(dataset_id=dataset_id)

    pages = dataset.items.list(filters=filters)
    print(f'items in test set: {pages.items_count}')

    # success, message = scorer.create_model_score(dataset=dataset,
    #                                                         filters=filters,
    #                                                         model=model,
    #                                                         compare_types=model.output_type)
    # print(message)

    # model_scores = scorer.get_scores_df(model=model, dataset=dataset)
    # metric_names = ['accuracy', 'iou', 'confidence']
    #
    plot_points = scorer.calc_precision_recall(dataset_id=dataset.id,
                                               model_id=model.id)

    labels = [label.tag for label in dataset.labels]
    save_path = scorer.plot_precision_recall(plot_points=plot_points,
                                             label_names=labels)
    # from pathlib import Path
    #
    # plot_points.to_csv(Path(Path(save_path).parent, '.dataloop', 'plot_points.csv'))
    # #
    # print()
