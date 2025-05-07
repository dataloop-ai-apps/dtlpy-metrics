import unittest
import logging
import dtlpy as dl

from unittest.mock import Mock, patch
from dtlpymetrics.dtlpy_scores import Score, ScoreType
from dtlpymetrics.utils.dl_helpers import (
    check_if_video,
    add_score_context,
    cleanup_annots_by_score,
    get_scores_by_annotator,
    get_best_annotator_by_score,
)


class TestDLHelpers(unittest.TestCase):
    def setUp(self):
        # Create mock dl.Item
        self.mock_item = Mock(spec=dl.Item)
        self.mock_item.metadata = {"system": {"mimetype": "image/jpeg"}}

        # Create mock annotations
        self.mock_ground_truth_annot = Mock(spec=dl.Annotation)
        self.mock_ground_truth_annot.label = "cat"
        self.mock_ground_truth_annot.id = "ground_truth_1"

        self.mock_assignee_annot = Mock(spec=dl.Annotation)
        self.mock_assignee_annot.label = "dog"
        self.mock_assignee_annot.id = "assignee_1"

        # Create mock Scores for different types
        self.mock_raw_label_score = Mock(spec=Score)
        self.mock_raw_label_score.type = ScoreType.ANNOTATION_LABEL
        self.mock_raw_label_score.value = 0.9

        self.mock_raw_attribute_score = Mock(spec=Score)
        self.mock_raw_attribute_score.type = ScoreType.ANNOTATION_ATTRIBUTE
        self.mock_raw_attribute_score.value = 0.7

        self.mock_annotation_overall = Mock(spec=Score)
        self.mock_annotation_overall.type = ScoreType.ANNOTATION_OVERALL
        self.mock_annotation_overall.value = 0.8  # Mean of raw scores
        self.mock_annotation_overall.entity_id = "assignee_1"
        self.mock_annotation_overall.context = {
            "assignmentId": "assignee_assignment_id"
        }

        self.mock_user_confusion = Mock(spec=Score)
        self.mock_user_confusion.type = ScoreType.USER_CONFUSION
        self.mock_user_confusion.value = 0.85
        self.mock_user_confusion.context = {"assignmentId": "assignee_assignment_id"}

        # Create label confusion scores based on README example
        self.mock_cat_cat_confusion = Mock(spec=Score)
        self.mock_cat_cat_confusion.type = ScoreType.LABEL_CONFUSION
        self.mock_cat_cat_confusion.value = 1
        self.mock_cat_cat_confusion.entity_id = "cat"
        self.mock_cat_cat_confusion.context = {"relative": "cat"}

        self.mock_dog_dog_confusion = Mock(spec=Score)
        self.mock_dog_dog_confusion.type = ScoreType.LABEL_CONFUSION
        self.mock_dog_dog_confusion.value = 3
        self.mock_dog_dog_confusion.entity_id = "dog"
        self.mock_dog_dog_confusion.context = {"relative": "dog"}

        self.mock_dog_cat_confusion = Mock(spec=Score)
        self.mock_dog_cat_confusion.type = ScoreType.LABEL_CONFUSION
        self.mock_dog_cat_confusion.value = 2
        self.mock_dog_cat_confusion.entity_id = "dog"
        self.mock_dog_cat_confusion.context = {"relative": "cat"}

        self.mock_item_overall = Mock(spec=Score)
        self.mock_item_overall.type = ScoreType.ITEM_OVERALL
        self.mock_item_overall.value = 0.82
        self.mock_item_overall.context = {"itemId": "test_item_id"}

    def test_check_if_video(self):
        # Test non-video item
        self.assertFalse(check_if_video(self.mock_item))

        # Test video item
        self.mock_item.metadata["system"]["mimetype"] = "video/mp4"
        self.assertTrue(check_if_video(self.mock_item))

        # Test item without system metadata
        self.mock_item.metadata = {}
        self.assertFalse(check_if_video(self.mock_item))

        # Test with different video mimetypes
        video_mimetypes = [
            "video/webm",
            "video/mp4",
        ]
        for mimetype in video_mimetypes:
            self.mock_item.metadata = {"system": {"mimetype": mimetype}}
            self.assertTrue(check_if_video(self.mock_item))

        # Test with None mimetype
        self.mock_item.metadata = {"system": {"mimetype": None}}
        self.assertFalse(check_if_video(self.mock_item))

    def test_add_score_context(self):
        # Test adding partial context
        # Create a mock annotation with no context
        mock_no_context = Mock(spec=Score)
        mock_no_context.type = ScoreType.ANNOTATION_OVERALL
        mock_no_context.value = 0.8

        score = add_score_context(
            mock_no_context, user_id="test_user_id", task_id="test_task_id"
        )

        self.assertEqual(score.user_id, "test_user_id")
        self.assertEqual(score.task_id, "test_task_id")
        self.assertIsNone(score.relative)
        self.assertIsNone(score.entity_id)
        self.assertIsNone(score.assignment_id)
        self.assertIsNone(score.item_id)
        self.assertIsNone(score.dataset_id)

        # Test adding all context fields, add the rest
        score = add_score_context(
            self.mock_annotation_overall,
            relative="test_relative",
            entity_id="test_entity_id",
            assignment_id="test_assignment_id",
            item_id="test_item_id",
            dataset_id="test_dataset_id",
        )

        self.assertEqual(score.relative, "test_relative")
        self.assertEqual(score.user_id, "test_user_id")
        self.assertEqual(score.entity_id, "test_entity_id")
        self.assertEqual(score.assignment_id, "test_assignment_id")
        self.assertEqual(score.task_id, "test_task_id")
        self.assertEqual(score.item_id, "test_item_id")
        self.assertEqual(score.dataset_id, "test_dataset_id")

        # Test with empty string values
        score = add_score_context(
            mock_no_context,
            relative="",
            user_id="",
            entity_id="",
            assignment_id="",
            task_id="",
            item_id="",
            dataset_id="",
        )
        self.assertEqual(score.relative, "")
        self.assertEqual(score.user_id, "")
        self.assertEqual(score.entity_id, "")
        self.assertEqual(score.assignment_id, "")
        self.assertEqual(score.task_id, "")
        self.assertEqual(score.item_id, "")
        self.assertEqual(score.dataset_id, "")

    @patch("dtlpy.annotations.delete")
    def test_cleanup_annots_by_score(self, mock_delete):
        # Create test scores with different types
        scores = [
            self.mock_annotation_overall,
            self.mock_user_confusion,
            self.mock_label_confusion,
            self.mock_item_overall,
        ]

        # Test with annots_to_keep
        cleanup_annots_by_score(scores, annots_to_keep=["test_entity_id"])
        mock_delete.assert_called_once()
        call_args = mock_delete.call_args[1]["filters"].prepare()
        self.assertEqual(call_args["id"], "test_entity_id")

        # Test with logger
        mock_logger = Mock(spec=logging.Logger)
        cleanup_annots_by_score(
            scores, annots_to_keep=["test_entity_id"], logger=mock_logger
        )
        mock_logger.info.assert_called_once()

        # Test with empty scores list
        mock_delete.reset_mock()
        cleanup_annots_by_score([], annots_to_keep=["test_entity_id"])
        mock_delete.assert_not_called()

        # Test with empty annots_to_keep list
        mock_delete.reset_mock()
        cleanup_annots_by_score(scores, annots_to_keep=[])
        mock_delete.assert_called_once()
        call_args = mock_delete.call_args[1]["filters"].prepare()
        self.assertEqual(call_args["id"], self.mock_annotation_overall.entity_id)

    def test_get_scores_by_annotator(self):
        # Create test scores with different types
        scores = [
            self.mock_annotation_overall,
            self.mock_user_confusion,
            self.mock_label_confusion,
            self.mock_item_overall,
        ]

        result = get_scores_by_annotator(scores)
        self.assertEqual(len(result), 1)
        self.assertEqual(result["test_assignment_id"], [0.8])

    def test_get_best_annotator_by_score(self):
        # Create test scores with different types
        scores = [
            self.mock_annotation_overall,
            self.mock_user_confusion,
            self.mock_label_confusion,
            self.mock_item_overall,
        ]

        result = get_best_annotator_by_score(scores)
        self.assertEqual(
            result, 0.8
        )  # Should return the highest score for ANNOTATION_OVERALL


if __name__ == "__main__":
    unittest.main()
