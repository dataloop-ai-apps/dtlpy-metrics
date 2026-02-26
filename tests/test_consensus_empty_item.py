import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("DATALOOP_PATH", os.path.join(os.getcwd(), ".dataloop_test"))

from dtlpymetrics.evaluating.tasks import check_annotator_agreement, get_consensus_agreement


class TestConsensusEmptyItem(unittest.TestCase):
    def test_empty_user_confusion_scores_return_agreement(self):
        self.assertTrue(check_annotator_agreement(scores=[], threshold=0.5))

    @patch("dtlpymetrics.evaluating.tasks.cleanup_annots_by_score")
    @patch("dtlpymetrics.evaluating.tasks.get_scores_by_annotator")
    @patch("dtlpymetrics.evaluating.tasks.calc_task_item_score")
    def test_keep_only_best_on_blank_consensus_item_returns_agreement(
        self,
        mock_calc_task_item_score,
        mock_get_scores_by_annotator,
        mock_cleanup_annots_by_score,
    ):
        mock_calc_task_item_score.return_value = []
        mock_get_scores_by_annotator.return_value = {}

        item = MagicMock()
        item.id = "item-1"
        task = MagicMock()
        task.id = "task-1"
        task.name = "consensus-task"

        agreement = get_consensus_agreement(
            item=item,
            task=task,
            agreement_config={
                "agreement_threshold": 0.5,
                "keep_only_best": True,
                "fail_keep_all": True,
            },
        )

        self.assertTrue(agreement)
        mock_get_scores_by_annotator.assert_called_once()
        mock_cleanup_annots_by_score.assert_not_called()


if __name__ == "__main__":
    unittest.main()
