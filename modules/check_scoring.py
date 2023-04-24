import dtlpy as dl
from modules import scoring

dl.setenv('prod')
project = dl.projects.get('feature vectors')
dataset = project.datasets.get('suim creatures')
SAVE_PLOT = True

consensus_task = dataset.tasks.get('check_consensus')  # 643be0e4bc2e4cb8b7c1a78d
feature_set = scoring.calculate_consensus_score(consensus_task, save_plot=SAVE_PLOT)

print(f'Created consensus scores in this feature set: {feature_set.id}')


#################################
# Check feature vectors in prod #
#################################

feature = feature_set.features.list()[0][0]
item = dl.items.get(None, feature.entity_id)
filters = dl.Filters(use_defaults=False,
                     resource=dl.FiltersResource.FEATURE,
                     custom_filter={
                         '$and': [{'entityId': item.id},
                                  {'dataType': 'itemScore'}]
                     })

pages_feat = dl.features.list(filters=filters)

dl.client_api.print_request()
print(pages_feat.items_count)

for feature in pages_feat.all():
    found_feature_set = dl.feature_sets.get(None, feature.feature_set_id)
    print(feature.id, feature.value, found_feature_set.name)