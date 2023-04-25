import dtlpy as dl

env = 'rc'
dl.setenv(env)

project = dl.projects.get(project_name='feature vectors')

TO_INSTALL = True
dpk_name = 'scoring-and-metrics'

if TO_INSTALL:
    # install
    dpk = dl.dpks.init(name=dpk_name)
    project.dpks.publish(dpk)

    dpk = dl.dpks.get(dpk_name=dpk_name)

    app = project.apps.get(app_name=dpk.name)
    app.uninstall()
    project.apps.install(dpk)

else:
    # uninstall
    app = project.apps.get(app_name='Scoring and metrics app')
    app.uninstall()

    dpk = dl.dpks.get(dpk_name=dpk_name)
    dpk.delete()

dpk = dl.dpks.get(None, '64413bc73b6ed86ac9d664fe')
# dl.dpks.delete(dpk_id=dpk.id)

# dlp app publish --project-name "feature vectors"
# dlp app install --dpk-id 64413d3e3b6ed851f2d66501 --project-name "feature vectors"

# dlp app update --app-name "Scoring and metrics app" --new-version 1.0.1

# app id: 64413d723b6ed817a0d66503
# dpk id: 64413d3e3b6ed851f2d66501

delete_dpk = project.dpks.delete('644132bf9a55951717ac05d0')
