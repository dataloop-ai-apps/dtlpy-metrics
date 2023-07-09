import dtlpy as dl
from dtlpymetrics.scoring import calc_precision_recall, plot_precision_recall, calc_confusion_matrix

dl.setenv('rc')
if dl.token_expired():
    dl.login()

dataset_id = '648f333a943352d180df011a'
# model_id = '649076c45a9c968a5c32ed65'

project = dl.projects.get(project_id='275c7a96-99ed-4bd8-b798-61b0dcbc5a8d')
dataset = project.datasets.get(dataset_id=dataset_id)
model = project.models.get(model_name='yolov8-2023')

plot_points = calc_precision_recall(dataset_id=dataset_id,
                                    model_id=model.id)
plot_precision_recall(plot_points)

# metric = 'accuracy'
# conf_table = calc_confusion_matrix(dataset_id=dataset_id,
#                                    model_id=model_id,
#                                    metric=metric)
# print("columns are model predictions, rows are ground truth labels")
# print(conf_table)
# print()
