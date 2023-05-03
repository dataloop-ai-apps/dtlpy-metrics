import dtlpy as dl
import subprocess

env = 'rc'
dl.setenv(env)

project = dl.projects.get(project_name='feature vectors')

# install
subprocess.check_call('bumpversion patch --allow-dirty', shell=True)

dpk = project.dpks.publish()

app = project.apps.get(app_name=dpk.display_name)
app.uninstall()
project.apps.install(dpk)

# dl.dpks.delete(dpk_id=dpk.id)

# dlp app publish --project-name "feature vectors"
# dlp app install --dpk-id 64413d3e3b6ed851f2d66501 --project-name "feature vectors"

# dlp app update --app-name "Scoring and metrics app" --new-version 1.0.1

# app id: 64413d723b6ed817a0d66503
# dpk id: 64413d3e3b6ed851f2d66501

# delete_dpk = project.dpks.delete('644132bf9a55951717ac05d0')
