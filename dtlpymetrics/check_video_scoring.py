import datetime
import os
import logging

import dtlpy as dl
from dtlpymetrics.scoring import calculate_task_score
from dtlpymetrics.dtlpy_scores import ScoreType, Scores, Score

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
qual_poly = project.tasks.get(task_id='64c8f0e4a3f0796c68c35532')
qual_poly_scores = calculate_task_score(task=qual_poly)

# dl_scores = Scores(client_api=dl.client_api)
# dl_scores.get(score_id='64c8f0e4a3f0796c68c35532')

