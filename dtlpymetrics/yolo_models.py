import dtlpy as dl
import json

dl.setenv('rc')
project = dl.projects.get(project_name='Active Learning 1.3')

public_model_name = 'pretrained-yolo-v8'
filters = dl.Filters(resource=dl.FiltersResource.MODEL,
                     field='scope',
                     values='public')
filters.add(field='name', values=public_model_name)
yolos = list(project.models.list(filters=filters).all())
yolo = project.models.get(model_id=yolos[0].id)
yolo_package = yolo.package

methods = yolo_package.metadata['system']['ml']['supportedMethods'].copy()
for method in methods:
    if 'evaluate' in method.keys():
        methods.remove(method)

print(yolo_package.metadata['system']['ml']['supportedMethods'])
yolo_package.metadata['system']['ml']['supportedMethods'] = methods
yolo_package.update()

yolo = dl.models.get(model_id=yolo.id)
print(yolo.package.metadata['system']['ml']['supportedMethods'])

yolo_clone = dl.models.get(model_id='6481ccc1216aa32321ec4f4a')
yolo_package = yolo_clone.package
print(yolo_clone.package.metadata['system']['ml']['supportedMethods'])

yolo_clone = dl.models.get(model_id='6481ccc1216aa32321ec4f4a')
yolo_clone2 = dl.models.clone(from_model=yolo_clone,
                              model_name='yolo_clone2',
                              project_id=project.id)
filter_str = json.loads(
    '{"filter": {"$and": [{"hidden": false}, {"$or": [{"metadata": {"system": {"tags": {"train": true}}}}]}, {"type": "file"}]}, "page": 0, "pageSize": 1000, "resource": "items"}')
filters = dl.Filters(custom_filter=filter_str)
filters.prepare()
yolo_clone2.evaluate(dataset_id='648174bb56e25a28ae01b32e', filters=filters)
print(yolo_clone2.id)

# adapter = yolo_clone2.