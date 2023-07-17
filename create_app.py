import dtlpy as dl
import os

project_id = os.environ.get('PROJECT_ID')
email = os.environ.get('BOT_EMAIL')
password = os.environ.get('BOT_PASSWORD')
env = os.environ.get('ENV')

dl.setenv(env)
print(f'login with email: {email}')
dl.login_m2m(email=email, password=password)

project = dl.projects.get(project_id=project_id)  # DataloopApps
print(f'publishing to project: {project.name}')

# publish dpk to app store
dpk = project.dpks.publish()
print(f'published successfully! dpk name: {dpk.name}, dpk id: {dpk.id}, version: {dpk.version}')
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
