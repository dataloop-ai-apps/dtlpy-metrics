import dtlpy as dl
import random
from dtlpymetrics.scoring import calculate_task_score, ScoreType


def setup_qual_gt(dataset: dl.Dataset):
    # FOR IOU SCORES
    # upload blank items + annotations
    img_path = r'G:\My Drive\DATASETS\blank_space.png'
    for i in range(3):
        try:
            filters = dl.Filters(field='name', values=f'blank_space_{i}.png')
            item = list(dataset.items.list(filters=filters).all())[0]

        except dl.exceptions.NotFound:
            item = dataset.items.upload(local_path=img_path,
                                        remote_name=f'blank_space_{i}.png')

        annotations = item.annotations.list()
        for annotation in annotations:
            annotation.delete()

        # create ground truth annotations
        builder = item.annotations.builder()
        for j in range(10):
            top = 200 * i
            left = 100 * j
            builder.add(
                annotation_definition=dl.Box(top=top,
                                             left=left,
                                             bottom=top + 100,
                                             right=left + 100,
                                             label='label1'))

        builder.upload()


def setup_qual_assignee(assignment: dl.Assignment,
                        recipe: dl.Recipe):
    # upload annotations with overlap IOUs of 0.1, 0.2, ..... 0.9
    items = list(assignment.get_items().all())
    context = {'taskId': assignment.task.id,
               'assignmentId': assignment.id,
               'recipeId': recipe.id,
               }

    for item in items:
        annotations = item.annotations.list()
        for annotation in annotations:
            annotation.delete()
        print(item)

    for item in items:
        print(item.filename)  # DEBUG
        builder = item.annotations.builder()

        i = int(item.filename.split('blank_space_')[-1].split('.')[0])
        for j in range(10):
            top = 200 * i
            left = 100 * j
            if j % 3 == 0:
                # every third label will disagree (label0)
                builder.add(
                    annotation_definition=dl.Box(top=top,
                                                 left=left,
                                                 bottom=top + (j * 10),
                                                 right=left + (j * 10),
                                                 label='label0'),
                    metadata={'system': context})

            else:
                builder.add(
                    annotation_definition=dl.Box(top=top,
                                                 left=left,
                                                 bottom=top + (j * 10),
                                                 right=left + (j * 10),
                                                 label='label1'),
                    metadata={'system': context})

        # builder.upload()
        item.annotations.upload(annotations=builder)

        # mark each item as complete
        item.update_status(status=dl.ItemStatus.COMPLETED)


def setup_consensus_bbox_assignee(assignment, recipe):
    filters = dl.Filters()
    filters.sort_by(field='filename', value=dl.FiltersOrderByDirection.ASCENDING)
    items = list(assignment.get_items(filters=filters).all())
    context = {'taskId': assignment.task.id,
               'assignmentId': assignment.id,
               'recipeId': recipe.id,
               }

    for i, item in enumerate(items):
        annotations = item.annotations.list()
        for annotation in annotations:
            if annotation.metadata.get('system', {}).get('assignmentId') == assignment.id:
                annotation.delete()

        # create consensus annotations with noise
        builder = item.annotations.builder()
        for j in range(10):
            top = 200 * i + random.randrange(-25, 25, 1)
            left = 100 * j + random.randrange(-25, 25, 1)
            builder.add(
                annotation_definition=dl.Box(top=top,
                                             left=left,
                                             bottom=top + 100,
                                             right=left + 100,
                                             label='label1'),
                metadata={'system': context})

        # builder.upload()
        item.annotations.upload(annotations=builder)

    return


if __name__ == '__main__':
    dl.setenv('rc')
    project = dl.projects.get(project_name='Quality Task Scores Testing')

    # ######################
    # # Qualification task #
    # ######################
    # try:
    #     dataset = project.datasets.get(dataset_name='bbox items')
    # except dl.exceptions.NotFound:
    #     dataset = project.datasets.create(dataset_name='bbox items')
    #     setup_qual_gt(dataset=dataset)
    #
    # task = project.tasks.get(task_name='qualification testing task')
    # assignment = project.assignments.get(assignment_name='qualification testing - bbox (Score-task-3) (1)')
    # recipe = dataset.recipes.get(recipe_id='64b011b946fcc124d980cff5')
    # setup_qual_assignee(assignment=assignment, recipe=recipe)
    #
    # #################
    # # Honeypot task #
    # #################
    # try:
    #     dataset = project.datasets.get(dataset_name='bbox items')
    # except dl.exceptions.NotFound:
    #     dataset = project.datasets.create(dataset_name='bbox items')
    #
    # task = project.tasks.get(task_name='honeypot testing task')
    # assignment = project.assignments.get(assignment_name='')
    # # recipe = dataset.recipes.get(recipe_id='64b011b946fcc124d980cff5')
    # recipe = dataset.recipes.list()[0]
    # setup_qual_assignee(assignment=assignment, recipe=recipe)

    ##################
    # Consensus task #
    ##################

    try:
        dataset = project.datasets.get(dataset_name='bbox items')
    except dl.exceptions.NotFound:
        dataset = project.datasets.create(dataset_name='bbox items')

    task = project.tasks.get(task_name='consensus testing task - bbox')
    # assignment = project.assignments.get(assignment_name='consensus task (Score-task-5) (2)')  # the other one is (1)
    assignment = project.assignments.get(assignment_id='64b3ce1e43645d3f2c2a5b4a')
    recipe = dataset.recipes.list()[0]
    setup_consensus_bbox_assignee(assignment=assignment, recipe=recipe)

    ###################
    # Label confusion #
    ###################
    # try:
    #     dataset = project.datasets.get(dataset_name='classification items')
    # except dl.exceptions.NotFound:
    #     dataset = project.datasets.create(dataset_name='classification items')
