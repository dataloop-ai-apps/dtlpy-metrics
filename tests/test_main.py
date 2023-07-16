import os
import unittest
import dtlpy as dl
from dtlpymetrics.dtlpy_scores import ScoreType
from dtlpymetrics.scoring import calculate_task_score, calculate_confusion_matrix


class TestRunner(unittest.TestCase):
    def setUp(self):
        dl.setenv("rc")
        print(f'env -> {dl.environment()}')
        if dl.token_expired():
            email = os.environ.get('LOGIN_EMAIL', None)
            password = os.environ.get('LOGIN_EMAIL_PASSWORD', None)
            print(f'email {email}')
            if email and password:
                is_logged_in = dl.login_m2m(email, password)
                print(f'is_logged_in {is_logged_in}')
            else:
                dl.login()

        self.project = dl.projects.get(project_name='Quality Task Scores Testing')

        self.qualification_task = self.project.tasks.get(task_name='qualification testing task')
        self.honeypot_task = self.project.tasks.get(task_name='honeypot testing task')
        self.consensus_task = self.project.tasks.get(task_name='consensus testing task')
        self.label_confusion_task = self.project.tasks.get(task_name='qualification testing - confusion matrix')

    def test_qualification_task(self):
        print(f'qualification testing task dataset: {self.qualification_task.dataset}')
        self.qualification_task = calculate_task_score(task=self.label_confusion_task,
                                                       score_types=[ScoreType.ANNOTATION_LABEL,
                                                                    ScoreType.ANNOTATION_IOU])

    def test_honeypot_task(self):
        print(f'honeypot testing task dataset: {self.honeypot_task.dataset}')
        self.honeypot_task = calculate_task_score(task=self.honeypot_task, score_types=[ScoreType.ANNOTATION_LABEL])

    def test_consensus_task(self):
        print(f'consensus testing task dataset: {self.consensus_task.dataset}')
        self.consensus_task = calculate_task_score(task=self.consensus_task, score_types=[ScoreType.ANNOTATION_LABEL])

    def test_confusion_matrix(self):
        self.label_confusion_task = calculate_task_score(task=self.label_confusion_task,
                                                         score_types=[ScoreType.ANNOTATION_LABEL])
        self.label_confusion_task = calculate_confusion_matrix(task=self.label_confusion_task)


if __name__ == "__main__":
    unittest.main()
