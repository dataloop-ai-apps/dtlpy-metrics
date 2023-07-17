import unittest
import datetime
import logging
import json
import os

import dtlpy as dl
from dtlpymetrics.dtlpy_scores import ScoreType
from dtlpymetrics.scoring import calculate_task_score, calculate_confusion_matrix_item

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

        # self.project = dl.projects.get(project_name='Quality Task Scores Testing')
        self.project = dl.projects.get(project_id='1c5f0cae-f2f8-48da-9429-9de875721759')  # Quality Task Scores Testing

        self.qualification_task = self.project.tasks.get(task_name='qualification testing task')  # c92
        self.honeypot_task = self.project.tasks.get(task_name='honeypot testing task')  # 7f7
        self.consensus_task_classification = self.project.tasks.get(
            task_name='consensus testing task - classification')  # b47
        self.consensus_task_bbox = self.project.tasks.get(task_name='consensus testing task - bbox')  # 855
        self.label_confusion_task = self.project.tasks.get(task_name='qualification testing - confusion matrix')  # e14

        logger.info('[SETUP] - done getting entities')

        self.test_dump_path = datetime.datetime.now().isoformat(sep='.', timespec='seconds').replace(':', '.').replace(
            '-', '.')
        os.environ['SCORES_DEBUG_PATH'] = f'./{self.test_dump_path}'

    def tearDown(self) -> None:
        # delete the test scores that were created to clean up
        pass

    # def test_qualification_task(self):
    #     logger.info(f'Starting qualification testing task with dataset: {self.qualification_task.dataset}')
    #     self.qualification_task = calculate_task_score(task=self.qualification_task,
    #                                                    score_types=[ScoreType.ANNOTATION_LABEL,
    #                                                                 ScoreType.ANNOTATION_IOU])
    #
    #     for item in self.qualification_task.get_items().all():
    #         with open(os.path.join('assets', self.qualification_task.id, f'{item.id}.json'), 'r') as f:
    #             ref_scores = json.load(f)
    #
    #         with open(os.path.join(self.test_dump_path, self.qualification_task.id, f'{item.id}.json'), 'r') as f:
    #             test_scores = json.load(f)
    #         logger.info('Comparing calculated scores with reference scores...')
    #         self.assertListEqual(test_scores, ref_scores)

    # def test_honeypot_task(self):
    #     logger.info(f'Starting honeypot testing task with dataset: {self.honeypot_task.dataset}')
    #     self.honeypot_task = calculate_task_score(task=self.honeypot_task,
    #                                               score_types=[ScoreType.ANNOTATION_LABEL])
    #
    #     ## currently, one honeypot item is missing annotations without an assignment ID
    #     # for item in self.honeypot_task.get_items().all():
    #     #     with open(os.path.join('assets', self.honeypot_task.id, f'{item.id}.json'), 'r') as f:
    #     #         ref_scores = json.load(f)
    #
    #     # for item in self.honeypot_task.get_items().all():
    #     #     with open(os.path.join(self.test_dump_path, self.honeypot_task.id, f'{item.id}.json'), 'r') as f:
    #     #        test_scores = json.load(f)
    #
    def test_consensus_tasks(self):
        logger.info(f'consensus testing task dataset: {self.consensus_task_classification.dataset}')
        logger.info('calculating scores for consensus classification task')
        self.consensus_task_classification = calculate_task_score(task=self.consensus_task_classification,
                                                                  score_types=[ScoreType.ANNOTATION_LABEL])

        # for item in self.consensus_task_classification.get_items().all():
        #     with open(os.path.join('assets', self.consensus_task_classification.id, f'{item.id}.json'), 'r') as f:
        #         ref_scores = json.load(f)

        filters = dl.Filters()
        filters.add(field='hidden', values=False)
        for item in self.consensus_task_classification.get_items(filters=filters, get_consensus_items=True).all():
            print(item.id)
            with open(os.path.join(self.test_dump_path, self.consensus_task_classification.id, f'{item.id}.json'),
                      'r') as f:
                test_scores = json.load(f)
    #
    #     logger.info('calculating scores for consensus object detection task')
    #     self.consensus_task_bbox = calculate_task_score(task=self.consensus_task_bbox,
    #                                                     score_types=[ScoreType.ANNOTATION_LABEL,
    #                                                                  ScoreType.ANNOTATION_IOU])
    #
    #     # for item in self.consensus_task_bbox.get_items().all():
    #     #     with open(os.path.join('assets', self.consensus_task_bbox.id, f'{item.id}.json'), 'r') as f:
    #     #         ref_scores = json.load(f)
    #
    #     for item in self.consensus_task_bbox.get_items().all():
    #         with open(os.path.join(self.test_dump_path, self.consensus_task_bbox.id, f'{item.id}.json'), 'r') as f:
    #             test_scores = json.load(f)

    # def test_confusion_matrix(self):
    #     self.label_confusion_task = calculate_task_score(task=self.label_confusion_task,
    #                                                      score_types=[ScoreType.ANNOTATION_LABEL])
    #
    #     for item in self.label_confusion_task.get_items().all():
    #         with open(os.path.join(self.test_dump_path, self.label_confusion_task.id, f'{item.id}.json'), 'r') as f:
    #             scores = json.load(f)
    #
    #         confusion_matrix = calculate_confusion_matrix_item(item=item,
    #                                                            scores=scores,
    #                                                            save_plot=True)


if __name__ == "__main__":
    unittest.main()
