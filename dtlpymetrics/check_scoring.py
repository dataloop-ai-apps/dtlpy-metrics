import dtlpy as dl
from dtlpymetrics.scoring import ScoringAndMetrics

dl.setenv('rc')
project = dl.projects.get('feature vectors')
dataset = project.datasets.get('waterfowl')

fsets = project.feature_sets.list()
for fset in fsets:
    if 'Consensus' in fset.name:
        fset.delete()
        print(f'{fset.name} deleted')

# project.feature_sets.list().print()
# fset = project.feature_sets.get('Consensus IOU')

consensus_task = dataset.tasks.get('pipeline consensus test (test tasks)')
scoring = ScoringAndMetrics()
scoring.calculate_consensus_score(consensus_task)

#
# #################################
# # Check feature vectors in prod #
# #################################
#
# feature = feature_set.features.list()[0][0]
# item = dl.items.get(None, feature.entity_id)
# filters = dl.Filters(use_defaults=False,
#                      resource=dl.FiltersResource.FEATURE,
#                      custom_filter={
#                          '$and': [{'entityId': item.id},
#                                   {'dataType': 'itemScore'}]
#                      })
#
# pages_feat = dl.features.list(filters=filters)
#
# dl.client_api.print_request()
# print(pages_feat.items_count)
#
# for feature in pages_feat.all():
#     found_feature_set = dl.feature_sets.get(None, feature.feature_set_id)
#     print(feature.id, feature.value, found_feature_set.name)