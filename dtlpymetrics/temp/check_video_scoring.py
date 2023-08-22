import datetime
import os
import logging

import dtlpy as dl
from dtlpymetrics.scoring import calculate_task_score, create_task_item_score

logger = logging.getLogger()

# FOR VIDEO SCORING, POLYGON ANNOTATION, CONSENSUS TASK
# setup
dl.setenv("prod")
project = dl.projects.get(project_id='7aae403c-fc5a-4a09-af50-56a903be62ad')  # ScoutCam project

logger.info('[SETUP] - done getting entities')
now = datetime.datetime.now().isoformat(sep='.', timespec='seconds').replace(':', '.').replace('-', '.')
# assets_path = os.path.join(os.getcwd(), 'assets')  # './tests/assets'
test_dump_path = os.path.join('dtlpymetrics', now)

# item = dl.items.get(item_id='64c8f0efb118f77c071a0706')  # the middle video
qualification_task = project.tasks.get(task_id='64c8f0e4a3f0796c68c35532')
qualification_task_scores = calculate_task_score(task=qualification_task)

# dl_scores = Scores(client_api=dl.client_api)
# dl_scores.get(score_id='64c8f0e4a3f0796c68c35532')

print()


# vid_gt = dl.items.get(item_id='64be6ae759c68826d2488345')
# vid_test = dl.items.get(item_id='64c8f0f9b118f7360a1a071d')
#
# qualification_task = dl.tasks.get(task_id='64c8f0e4a3f0796c68c35532')
#
# vid_item_scores = create_task_item_score(item=vid_gt,
#                                          task=qualification_task)
