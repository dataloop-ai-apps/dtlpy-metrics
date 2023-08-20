import dtlpy as dl
from dtlpymetrics.scoring import create_task_item_score

vid_gt = dl.items.get(item_id='64be6ae759c68826d2488345')
vid_test = dl.items.get(item_id='64c8f0f9b118f7360a1a071d')

qualification_task = dl.tasks.get(task_id='64c8f0e4a3f0796c68c35532')

vid_item_scores = create_task_item_score(item=vid_gt,
                                         task=qualification_task)
