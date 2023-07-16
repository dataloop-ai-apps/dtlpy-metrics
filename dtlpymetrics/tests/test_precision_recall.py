import dtlpy as dl
from dtlpymetrics.precision_recall import calc_precision_recall, plot_precision_recall, calc_confusion_matrix
from dtlpymetrics.scoring import create_model_score

dl.setenv('prod')
if dl.token_expired():
    dl.login()

# dataset_id = '648f333a943352d180df011a'
# model_id = '649076c45a9c968a5c32ed65'

dataset_id = '6492f0aaa93c6d3a0e1c164b'
project = dl.projects.get(project_id='275c7a96-99ed-4bd8-b798-61b0dcbc5a8d')
dataset = project.datasets.get(dataset_id=dataset_id)
filters = dl.Filters(field='dir', values='/test')

model = project.models.get(model_name='yolov8-2023') #-finetune')
# model_results = create_model_score(dataset=dataset,
#                                    filters=filters,
#                                    model=model,
#                                    compare_types=['box'],
#                                    ignore_labels=False,
#                                    match_threshold=0.5)

plot_points = calc_precision_recall(dataset_id=dataset_id,
                                    model_id=model.id,
                                    iou_threshold=0.5,
                                    method_type='every_point')
                                    # n_points=11)
plot_precision_recall(plot_points=plot_points,
                      dataset_name=dataset.name)
