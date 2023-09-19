import shutil
import unittest
import datetime
import logging
import json
import os

import dtlpy as dl
from dtlpymetrics.dtlpy_scores import ScoreType
from dtlpymetrics import calculate_task_score

logger = logging.getLogger()


class TestRunner(unittest.TestCase):
    def setUp(self):
        dl.setenv("rc")
        self.project = dl.projects.get(project_id='1c5f0cae-f2f8-48da-9429-9de875721759')  # Quality Task Scores Testing

        self.qualification_task = self.project.tasks.get(task_name='qualification testing task')  # c92
        self.honeypot_task = self.project.tasks.get(task_name='honeypot testing task')  # 7f7
        self.consensus_task_classification = self.project.tasks.get(
            # task_name='consensus testing task - classification')  # b47
            task_name='consensus testing task - classification 2.0 (Score-task-5)')  # 27f
        self.consensus_task_bbox = self.project.tasks.get(task_name='consensus testing task - bbox')  # 855
        self.label_confusion_task = self.project.tasks.get(task_name='qualification testing - confusion matrix')  # e14

        logger.info('[SETUP] - done getting entities')
        now = datetime.datetime.now().isoformat(sep='.', timespec='minutes').replace(':', '.').replace('-', '_')
        self.assets_path = os.path.join(os.getcwd(), 'assets')  # './tests/assets'
        self.test_dump_path = os.path.join(os.getcwd(), 'assets', now)
        os.environ['SCORES_DEBUG_PATH'] = self.test_dump_path

    def tearDown(self) -> None:
        if os.path.isdir(self.test_dump_path):
            shutil.rmtree(self.test_dump_path)

    def test_qualification_task(self):
        logger.info(f'Starting qualification testing task with dataset: {self.qualification_task.dataset}')
        self.qualification_task = calculate_task_score(task=self.qualification_task,
                                                       score_types=[ScoreType.ANNOTATION_LABEL,
                                                                    ScoreType.ANNOTATION_IOU])

        qualification_items = self.qualification_task.get_items().all()
        for item in qualification_items:
            logger.info(f'Comparing calculated scores with reference scores for item: {item.id}')
            with open(os.path.join(self.assets_path, self.qualification_task.id, f'{item.id}.json'), 'r') as f:
                ref_scores = json.load(f)
            with open(os.path.join(self.test_dump_path, self.qualification_task.id, f'{item.id}.json'), 'r') as f:
                test_scores = json.load(f)
            self.assertListEqual(test_scores, ref_scores)

    def test_honeypot_task(self):
        logger.info(f'Starting honeypot testing task with dataset: {self.honeypot_task.dataset}')
        self.honeypot_task = calculate_task_score(task=self.honeypot_task,
                                                  score_types=[ScoreType.ANNOTATION_LABEL])

        filters = dl.Filters()
        filters.add(field='hidden', values=True)
        honeypot_items = list(self.honeypot_task.get_items(filters=filters).all())
        for item in honeypot_items:
            logger.info(f'Comparing calculated scores with reference scores for item: {item.id}')
            with open(os.path.join(self.assets_path, self.honeypot_task.id, f'{item.id}.json'), 'r') as f:
                ref_scores = json.load(f)
            with open(os.path.join(self.test_dump_path, self.honeypot_task.id, f'{item.id}.json'), 'r') as f:
                test_scores = json.load(f)
            self.assertListEqual(test_scores, ref_scores)

    def test_consensus_tasks(self):
        logger.info(f'consensus testing task dataset: {self.consensus_task_classification.dataset}')
        ###########################
        # for classification task #
        ###########################
        logger.info('calculating scores for consensus classification task')
        self.consensus_task_classification = calculate_task_score(task=self.consensus_task_classification,
                                                                  score_types=[ScoreType.ANNOTATION_LABEL])

        consensus_assignment = self.consensus_task_classification.metadata['system']['consensusAssignmentId']
        consensus_class_items = self.consensus_task_classification.get_items(get_consensus_items=True).all()

        for item in consensus_class_items:
            orig_consensus = False
            for ref_obj in item.metadata['system']['refs']:
                if ref_obj['id'] == consensus_assignment:
                    orig_consensus = True
                    break
            if orig_consensus is True:
                logger.info(f'Comparing calculated scores with reference scores for item: {item.id}')
                with open(os.path.join(self.assets_path, self.consensus_task_classification.id, f'{item.id}.json'),
                          'r') as f:
                    ref_scores = json.load(f)
                with open(os.path.join(self.test_dump_path, self.consensus_task_classification.id, f'{item.id}.json'),
                          'r') as f:
                    test_scores = json.load(f)
                self.assertListEqual(test_scores, ref_scores)

        #################
        # for bbox task #
        #################
        logger.info('calculating scores for consensus object detection task')
        self.consensus_task_bbox = calculate_task_score(task=self.consensus_task_bbox,
                                                        score_types=[ScoreType.ANNOTATION_LABEL,
                                                                     ScoreType.ANNOTATION_IOU])
        consensus_assignment = self.consensus_task_bbox.metadata['system']['consensusAssignmentId']
        consensus_bbox_items = self.consensus_task_bbox.get_items(get_consensus_items=False).all()

        for item in consensus_bbox_items:
            orig_consensus = False
            for ref_obj in item.metadata['system']['refs']:
                if ref_obj['id'] == consensus_assignment:
                    orig_consensus = True
                    break
            if orig_consensus is True:
                logger.info(f'Comparing calculated scores with reference scores for item: {item.id}')
                with open(os.path.join(self.assets_path, self.consensus_task_bbox.id, f'{item.id}.json'), 'r') as f:
                    ref_scores = json.load(f)
                with open(os.path.join(self.test_dump_path, self.consensus_task_bbox.id, f'{item.id}.json'), 'r') as f:
                    test_scores = json.load(f)
                self.assertListEqual(test_scores, ref_scores)

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
