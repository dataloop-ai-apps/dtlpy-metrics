import dtlpy as dl
from dtlpymetrics.scoring import ScoringAndMetrics

dl.setenv('rc')

########################
# Consensus task test ##
########################
scores = ScoringAndMetrics()
# consensus_task = dl.tasks.get(task_name='pipeline consensus test (test tasks)')
consensus_task = dl.tasks.get(task_id='644a307ae052f434dab98ff3')
scores.calculate_consensus_items(consensus_task)
