# import pytest
import dtlpy as dl
from unittest.mock import Mock, patch
from dtlpymetrics.evaluating.tasks import get_consensus_agreement, ScoreType


# @pytest.fixture
# def mock_item():
#     item = Mock(spec=dl.Item)
#     item.id = "test_item_id"
#     return item


# @pytest.fixture
# def mock_task():
#     task = Mock(spec=dl.Task)
#     task.id = "test_task_id"
#     task.name = "test_task"
#     return task


# @pytest.fixture
# def mock_progress():
#     progress = Mock(spec=dl.Progress)
#     return progress


# def test_get_consensus_agreement_with_agreement(mock_item, mock_task, mock_progress):
#     # Mock the calc_task_item_score to return scores indicating agreement
#     mock_scores = [
#         Mock(type=ScoreType.USER_CONFUSION, value=0.8, entity_id="ann1"),
#         Mock(type=ScoreType.USER_CONFUSION, value=0.9, entity_id="ann2"),
#         Mock(
#             type=ScoreType.ANNOTATION_OVERALL,
#             value=0.85,
#             entity_id="ann1",
#             context={"assignmentId": "user1"},
#         ),
#         Mock(
#             type=ScoreType.ANNOTATION_OVERALL,
#             value=0.95,
#             entity_id="ann2",
#             context={"assignmentId": "user2"},
#         ),
#     ]

#     agreement_config = {
#         "agree_threshold": 0.5,
#         "keep_only_best": True,
#         "fail_keep_all": True,
#     }

#     with patch(
#         "dtlpymetrics.evaluating.tasks.calc_task_item_score", return_value=mock_scores
#     ), patch("dtlpymetrics.evaluating.tasks.cleanup_annots_by_score") as mock_cleanup:

#         result = get_consensus_agreement(
#             item=mock_item,
#             task=mock_task,
#             agreement_config=agreement_config,
#             progress=mock_progress,
#         )

#         # Verify progress was updated correctly
#         mock_progress.update.assert_called_once_with(action="consensus passed")

#         # Verify cleanup was called with correct parameters
#         mock_cleanup.assert_called_once()

#         # Verify the returned item
#         assert result == mock_item


# def test_get_consensus_agreement_without_agreement(mock_item, mock_task, mock_progress):
#     # Mock the calc_task_item_score to return scores indicating disagreement
#     mock_scores = [
#         Mock(type=ScoreType.USER_CONFUSION, value=0.3, entity_id="ann1"),
#         Mock(type=ScoreType.USER_CONFUSION, value=0.4, entity_id="ann2"),
#         Mock(
#             type=ScoreType.ANNOTATION_OVERALL,
#             value=0.35,
#             entity_id="ann1",
#             context={"assignmentId": "user1"},
#         ),
#         Mock(
#             type=ScoreType.ANNOTATION_OVERALL,
#             value=0.45,
#             entity_id="ann2",
#             context={"assignmentId": "user2"},
#         ),
#     ]

#     agreement_config = {
#         "agree_threshold": 0.5,
#         "keep_only_best": True,
#         "fail_keep_all": False,
#     }

#     with patch(
#         "dtlpymetrics.evaluating.tasks.calc_task_item_score", return_value=mock_scores
#     ), patch("dtlpymetrics.evaluating.tasks.cleanup_annots_by_score") as mock_cleanup:

#         result = get_consensus_agreement(
#             item=mock_item,
#             task=mock_task,
#             agreement_config=agreement_config,
#             progress=mock_progress,
#         )

#         # Verify progress was updated correctly
#         mock_progress.update.assert_called_once_with(action="consensus failed")

#         # Verify cleanup was called with correct parameters
#         mock_cleanup.assert_called_once()

#         # Verify the returned item
#         assert result == mock_item


# def test_get_consensus_agreement_without_progress(mock_item, mock_task):
#     # Test the function without providing a progress object
#     mock_scores = [
#         Mock(type=ScoreType.USER_CONFUSION, value=0.8, entity_id="ann1"),
#         Mock(type=ScoreType.USER_CONFUSION, value=0.9, entity_id="ann2"),
#     ]

#     agreement_config = {
#         "agree_threshold": 0.5,
#         "keep_only_best": True,
#         "fail_keep_all": True,
#     }

#     with patch(
#         "dtlpymetrics.evaluating.tasks.calc_task_item_score", return_value=mock_scores
#     ):
#         result = get_consensus_agreement(
#             item=mock_item, task=mock_task, agreement_config=agreement_config
#         )

#         # Verify the returned item
#         assert result == mock_item


if __name__ == "__main__":
    dl.setenv("rc")
    item = dl.items.get(item_id='681b65b61b60b46ee7878bb0')
    task = dl.tasks.get(task_id='6805e555f641550c9c7e306f')
    agreement_config = {
        "agree_threshold": 0.2,
        "keep_only_best": True,
        "fail_keep_all": False,
    }
    progress = dl.Progress()
    context = dl.Context()
    _ = get_consensus_agreement(item=item, task=task, agreement_config=agreement_config, progress=progress)