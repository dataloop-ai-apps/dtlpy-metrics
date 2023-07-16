import unittest
import dtlpy as dl
from dtlpymetrics.scoring import create_task_item_score


class TestGetTaskItemScore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dl.setenv('prod')
        if dl.token_expired():
            dl.login()

            cls.project = dl.projects.get(project_name='Quality Task Scoring')
            cls.dataset = cls.project.datasets.get(dataset_name='qualification')
            cls.task = cls.dataset.tasks.get(task_name='quality task test')
            cls.item = cls.dataset.items.get(item_name='blank_space_0.png')
            cls.model = cls.project.models.get(model_name='yolov8-2023')


    def test_normal_case(self):
        pass


