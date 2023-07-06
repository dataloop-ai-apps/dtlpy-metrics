from dtlpymetrics.scoring import calc_precision_recall, plot_precision_recall, create_model_score, get_scores_df
import dtlpy as dl
import logging
import pandas as pd

dl.setenv('prod')
logging.basicConfig(level='DEBUG')

project = dl.projects.get(project_id='275c7a96-99ed-4bd8-b798-61b0dcbc5a8d')
dataset = dl.datasets.get(dataset_id='6492f0aaa93c6d3a0e1c164b')
filters = dl.Filters(field='dir', values='/test')

# project.models.list().print()

model_names = ['yolov8-2023',  'yolov8-2023-finetune']
model_results = {model_name: pd.DataFrame() for model_name in model_names}
for model_name in model_names:
    # model = project.models.get(model_name='yolov8-2023')
    model = project.models.get(model_name=model_name)
    create_model_score(dataset=dataset,
                       filters=filters,
                       model=model,
                       compare_types=dl.AnnotationType.BOX)
    model_scores = get_scores_df(model=model, dataset=dataset)

    model_results[model_name] = pd.concat([model_results[model_name], model_scores],
                                          ignore_index=True)

    plot_points = calc_precision_recall(dataset_id=dataset.id,
                                        model_id=model.id,
                                        iou_threshold=0.5)

    save_dir = plot_precision_recall(plot_points)

    print(f'plots found in {save_dir}')
