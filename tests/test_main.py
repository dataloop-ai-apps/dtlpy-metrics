import unittest
import datetime
import logging
import json
import os

import dtlpy as dl
from dtlpymetrics.dtlpy_scores import ScoreType
from dtlpymetrics.scoring import calculate_task_score, calculate_confusion_matrix

logger = logging.getLogger()


class TestRunner(unittest.TestCase):
    def setUp(self):
        dl.setenv("rc")
        # print(f'env -> {dl.environment()}')
        # if dl.token_expired():
        #     email = os.environ.get('LOGIN_EMAIL', None)
        #     password = os.environ.get('LOGIN_EMAIL_PASSWORD', None)
        #     print(f'email {email}')
        #     if email and password:
        #         is_logged_in = dl.login_m2m(email, password)
        #         print(f'is_logged_in {is_logged_in}')
        #     else:
        #         dl.login()

        self.project = dl.projects.get(project_name='Quality Task Scores Testing')
        self.project = dl.projects.get(project_id='1c5f0cae-f2f8-48da-9429-9de875721759')
        logger.info('[SETUP] - done getting entities')
        self.test_dump_path = datetime.datetime.now().isoformat(sep='.', timespec='seconds').replace(':', '.').replace(
            '-', '.')
        os.environ['SCORES_DEBUG_PATH'] = f'./{self.test_dump_path}'

        self.qualification_task = self.project.tasks.get(task_name='qualification testing task')
        # self.honeypot_task = self.project.tasks.get(task_name='honeypot testing task')
        # self.consensus_task = self.project.tasks.get(task_name='consensus testing task')
        # self.label_confusion_task = self.project.tasks.get(task_name='qualification testing - confusion matrix')

    def tearDown(self) -> None:
        pass

    def test_qualification_task(self):
        logger.info(f'qualification testing task dataset: {self.qualification_task.dataset}')
        self.qualification_task = calculate_task_score(task=self.qualification_task,
                                                       score_types=[ScoreType.ANNOTATION_LABEL,
                                                                    ScoreType.ANNOTATION_IOU])

        for item in self.qualification_task.get_items().all():
            with open(os.path.join('assets', self.qualification_task.id, f'{item.id}.json'), 'r') as f:
                ref_scores = json.load(f)

            with open(os.path.join(self.test_dump_path, self.qualification_task.id, f'{item.id}.json'), 'r') as f:
                test_scores = json.load(f)
            self.assertListEqual(test_scores, ref_scores)

    # def test_honeypot_task(self):
    #     logger.info(f'honeypot testing task dataset: {self.honeypot_task.dataset}')
    #     self.honeypot_task = calculate_task_score(task=self.honeypot_task, score_types=[ScoreType.ANNOTATION_LABEL])
    # #
    # def test_consensus_task(self):
    #     logger.info(f'consensus testing task dataset: {self.consensus_task.dataset}')
    #     self.consensus_task = calculate_task_score(task=self.consensus_task, score_types=[ScoreType.ANNOTATION_LABEL])
    #
    # def test_confusion_matrix(self):
    #     self.label_confusion_task = calculate_task_score(task=self.label_confusion_task,
    #                                                      score_types=[ScoreType.ANNOTATION_LABEL])
    #     self.label_confusion_task = calculate_confusion_matrix(task=self.label_confusion_task)


if __name__ == "__main__":
    unittest.main()
