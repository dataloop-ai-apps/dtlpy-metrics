import unittest
import datetime
import logging
import json
import os

import dtlpy as dl
from dtlpymetrics import ScoreType, calc_task_score, get_model_agreement

logger = logging.getLogger()

PATH = os.path.dirname(os.path.abspath(__file__))


class TestRunner(unittest.TestCase):
    def setUp(self):
        dl.setenv("rc")
        self.project = dl.projects.get(project_id='1c5f0cae-f2f8-48da-9429-9de875721759')  # Quality Task Scores Testing

        self.qualification_task = self.project.tasks.get(task_name='qualification testing task')  # c92
        self.honeypot_task = self.project.tasks.get(task_name='honeypot testing task')  # 7f7
        self.consensus_task_classification = self.project.tasks.get(
            # task_name='consensus testing task - classification')  # b47
            # task_name='consensus testing task - classification 2.0 (Score-task-5)')  # 27f
            task_name='scoring test - consensus classification (TEST consensus classification)'
        )  # a29
        self.consensus_task_bbox = self.project.tasks.get(task_name='consensus testing task - bbox')  # 855
        # self.label_confusion_task = self.project.tasks.get(task_name='qualification testing - confusion matrix') # e14

        logger.info('[SETUP] - done getting entities')
        now = datetime.datetime.now().isoformat(sep='.', timespec='minutes').replace('.', '_').replace(':', '.')
        self.assets_path = os.path.join(PATH, 'assets')

    def tearDown(self) -> None:
        pass

    def test_qualification_task(self):
        logger.info(f'Starting qualification testing task with dataset: {self.qualification_task.dataset}')
        qualification_scores = calc_task_score(
            task=self.qualification_task,
            score_types=[ScoreType.ANNOTATION_LABEL, ScoreType.ANNOTATION_IOU],
            upload=False,
        )

        qualification_items = self.qualification_task.get_items().all()
        for item in qualification_items:
            logger.info(f'Comparing calculated scores with reference scores for item: {item.id}')
            with open(os.path.join(self.assets_path, self.qualification_task.id, f'{item.id}.json'), 'r') as f:
                ref_scores = json.load(f)
            test_scores = [score.to_json() for score in qualification_scores[item.id]]
            self.assertListEqual(test_scores, ref_scores)

    def test_honeypot_task(self):
        logger.info(f'Starting honeypot testing task with dataset: {self.honeypot_task.dataset}')
        honeypot_scores = calc_task_score(
            task=self.honeypot_task, score_types=[ScoreType.ANNOTATION_LABEL], upload=False
        )

        filters = dl.Filters()
        filters.add(field='hidden', values=True)
        honeypot_items = list(self.honeypot_task.get_items(filters=filters).all())
        for item in honeypot_items:
            logger.info(f'Comparing calculated scores with reference scores for item: {item.id}')
            with open(os.path.join(self.assets_path, self.honeypot_task.id, f'{item.id}.json'), 'r') as f:
                ref_scores = json.load(f)
            test_scores = [score.to_json() for score in honeypot_scores[item.id]]
            self.assertListEqual(test_scores, ref_scores)

    def test_consensus_tasks(self):
        logger.info(f'consensus testing task dataset: {self.consensus_task_classification.dataset}')
        ###########################
        # for classification task #
        ###########################
        logger.info('calculating scores for consensus classification task')
        consensus_classification_scores = calc_task_score(
            task=self.consensus_task_classification, score_types=[ScoreType.ANNOTATION_LABEL], upload=False
        )

        consensus_assignment = self.consensus_task_classification.metadata['system']['consensusAssignmentId']
        consensus_class_items = self.consensus_task_classification.get_items(get_consensus_items=True).all()

        num_consensus = 0
        for item in consensus_class_items:
            consensus_task_found = False
            for ref_obj in item.metadata['system']['refs']:
                if ref_obj['id'] == consensus_assignment:
                    consensus_task_found = True
                    break
            if consensus_task_found is True:
                num_consensus += 1
                logger.info(f'Comparing calculated scores with reference scores for item: {item.id}')
                with open(
                    os.path.join(self.assets_path, self.consensus_task_classification.id, f'{item.id}.json'), 'r'
                ) as f:
                    ref_scores = json.load(f)
                test_scores = [score.to_json() for score in consensus_classification_scores[item.id]]
                self.assertListEqual(test_scores, ref_scores)
        logger.info(f'Compared scores for {num_consensus} consensus classification items')
        print(f'Compared scores for {num_consensus} consensus classification items')

        #################
        # for bbox task #
        #################
        logger.info('calculating scores for consensus object detection task')
        consensus_bbox_scores = calc_task_score(
            task=self.consensus_task_bbox,
            score_types=[ScoreType.ANNOTATION_LABEL, ScoreType.ANNOTATION_IOU],
            upload=False,
        )
        consensus_assignment = self.consensus_task_bbox.metadata['system']['consensusAssignmentId']
        consensus_bbox_items = self.consensus_task_bbox.get_items(get_consensus_items=True).all()

        num_consensus = 0
        for item in consensus_bbox_items:
            consensus_task_found = False
            for ref_obj in item.metadata['system']['refs']:
                if ref_obj['id'] == consensus_assignment:
                    consensus_task_found = True
                    break
            if consensus_task_found is True:
                num_consensus += 1
                logger.info(f'Comparing calculated scores with reference scores for item: {item.id}')
                with open(os.path.join(self.assets_path, self.consensus_task_bbox.id, f'{item.id}.json'), 'r') as f:
                    ref_scores = json.load(f)
                test_scores = [score.to_json() for score in consensus_bbox_scores[item.id]]
                self.assertListEqual(test_scores, ref_scores)
        logger.info(f'Compared scores for {num_consensus} consensus items')
        print(f'Compared scores for {num_consensus} consensus bbox items')

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

    def test_model_agreement(self):
        dl.setenv("prod")
        model = dl.models.get(model_id='67076a13859b460370cfd24a')
        item_agree = dl.items.get(item_id='67076a10fb04409b488e570c')
        agreement = get_model_agreement(
            item=item_agree, model=model, agreement_config={'agreement_threshold': 0.5, 'fail_keep_all': True}
        )
        self.assertTrue(agreement)
        item_disagree = dl.items.get(item_id='67076a10fb044082498e5709')
        disagreement = get_model_agreement(
            item=item_disagree, model=model, agreement_config={'agreement_threshold': 0.5, 'fail_keep_all': False}
        )
        self.assertFalse(disagreement)


if __name__ == "__main__":
    unittest.main()
