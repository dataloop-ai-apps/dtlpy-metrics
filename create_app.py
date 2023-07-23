import dtlpy as dl
import os

project_id = os.environ.get('PROJECT_ID')

project = dl.projects.get(project_id=project_id)  # DataloopApps
print(f'publishing to project: {project.name}')

# publish dpk to app store
dpk = project.dpks.publish()
print(f'published successfully! dpk name: {dpk.name}, dpk id: {dpk.id}, version: {dpk.version}')


def update():
    filters = dl.Filters(field='dpkName', values=dpk.name, resource='apps')
    apps = dl.apps.list(filters=filters)
    for app in apps.all():
        print(app.dpk_name, app.dpk_version)
        app.dpk_version = dpk.version
        app.update()


print(
    """\
                 .  ---  .
               /          \
              |   O  _  O  |
              |   ./   \.  |
              /   `-._.-'   \
            .'  /         \  `.
        .-~.-~ /           \ ~-.~-.
    .-~ ~     |             |     ~ ~-.
    `- .      |             |      . -'
         ~ -  |             |  - ~
              \             /
            ___\           /___
            ~;_  >- . . -<  _i~
    """)
