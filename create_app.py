import subprocess
import argparse
import requests
import json
import os
import dtlpy as dl

PANEL_NAME = 'scoring-and-metrics'


def bump(bump_type):
    print(f'Bumping version')
    subprocess.check_output(f'bumpversion {bump_type}', shell=True)
    subprocess.check_output('git push --follow-tags', shell=True)
    # tag latest
    subprocess.check_output('git tag --delete latest', shell=True)
    subprocess.check_output('git push --delete origin latest', shell=True)
    subprocess.check_output('git tag latest', shell=True)
    subprocess.check_output('git push origin --tags', shell=True)


def publish_and_install(project):
    success = True
    env = dl.environment()
    with open('dataloop.json') as f:
        manifest = json.load(f)
    app_name = manifest['name']
    app_version = manifest['version']
    user = os.environ.get('GITHUB_ACTOR', dl.info()['user_email'])
    try:
        print(f'Deploying to env : {dl.environment()}')
        print(f'publishing to project: {project.name}')
        # publish dpk to app store
        dpk = project.dpks.publish()
        print(f'published successfully! dpk name: {dpk.name}, version: {dpk.version}, dpk id: {dpk.id}')

        filters = dl.Filters(resource='apps')
        filters.add(field='dpkName', values=dpk.name)
        apps = dl.apps.list(filters=filters)
        print(f'Found {apps.items_count} apps for the DPK. Updating...')
        for app in apps.all():
            try:
                print(app.name, app.project.name, app.dpk_version)
                if app.dpk_version != dpk.version:
                    print('updating')
                    app.dpk_version = dpk.version
                    app.update()
            except Exception:
                print(f'Failed updating for app {app.id}')

        print(f'Done!')

    except Exception:
        success = False
    finally:

        status_msg = ':heavy_check_mark: Success :rocket:' if success else ':x: Failure :cry:'

        msg = f"""{status_msg}
        *App*: `{app_name}:{app_version}` => *{env}* by {user}
        """
        webhook = os.environ.get('SLACK_WEBHOOK')
        if webhook is None:
            print('WARNING: SLACK_WEBHOOK is None, cannot report')
        else:
            resp = requests.post(url=webhook,
                                 json={
                                     "blocks": [
                                         {
                                             "type": "section",
                                             "text": {
                                                 "type": "mrkdwn",
                                                 "text": msg
                                             }
                                         }

                                     ]
                                 })
            print(resp)


if __name__ == "__main__":
    # bump(bump_type='patch')
    dl.setenv('rc')
    project = dl.projects.get('')
    publish_and_install(project_id=project.id)

    print(
        r"""\
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
