import uuid
import logging
import dtlpy as dl
import numpy as np
import pandas as pd

from dtlpy import entities
from typing import Union, List

from dtlpymetrics.dtlpy_scores import Score, ScoreType

logger = logging.getLogger(name='scoring-and-metrics')

all_compare_types = [entities.AnnotationType.BOX,
                     entities.AnnotationType.CLASSIFICATION,
                     entities.AnnotationType.POLYGON,
                     entities.AnnotationType.POINT,
                     entities.AnnotationType.SEGMENTATION]

results_columns = {'annotation_iou': 'geometry_score',
                   'annotation_label': 'label_score',
                   'annotation_attribute': 'attribute_score',
                   'annotation_overall': 'annotation_score'}


def mean_or_nan(arr):
    if isinstance(arr, list) and len(arr) == 0:
        return np.nan
    else:
        return np.mean(arr)


def mean_or_default(arr, default):
    if isinstance(arr, list) and len(arr) == 0:
        return default
    else:
        return np.mean(arr)


def measure_annotations(
        annotations_set_one: Union[entities.AnnotationCollection, list],
        annotations_set_two: Union[entities.AnnotationCollection, list],
        match_threshold=0.5,
        ignore_labels=False,
        ignore_attributes=False,
        ignore_geometry=False,
        match_wrong_labels=True,
        compare_types=None):
    """
    Compares list (or collections) of annotations
    This will also return the precision and recall of the two sets, given that the first that is a GT and the second set
    is the detection (this affects the denominator of the calculation).

    :param annotations_set_one: dl.AnnotationCollection entity with a list of annotations to compare
    :param annotations_set_two: dl.AnnotationCollection entity with a list of annotations to compare
    :param match_threshold: IoU threshold to count as a match
    :param ignore_labels: ignore label when comparing - measure only geometry. if annotation type is classification,
    this will always be True (optional)
    :param ignore_attributes: ignore attribute score for final annotation score. if annotation type is classification,
    this will always be True (optional)
    :param ignore_geometry: only for classification (optional)
    :param match_wrong_labels: if True, will match geometric annotations even if they have different labels  (optional)
    :param compare_types: list of type to compare. enum dl.AnnotationType
    :return: dictionary of all the compared data
    """

    if compare_types is None:
        compare_types = all_compare_types
    if not isinstance(compare_types, list):
        compare_types = [compare_types]
    final_results = dict()
    all_scores = list()
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # for local annotations - set random id if None
    for annotation in annotations_set_one:
        if annotation.id is None:
            annotation.id = str(uuid.uuid1())
    for annotation in annotations_set_two:
        if annotation.id is None:
            annotation.id = str(uuid.uuid1())

    # start comparing
    for compare_type in compare_types:
        matches = Matches()
        annotation_subset_one = entities.AnnotationCollection()
        annotation_subset_two = entities.AnnotationCollection()
        annotation_subset_one.annotations = [a for a in annotations_set_one if
                                             a.type == compare_type and not a.metadata.get('system', dict()).get(
                                                 'system', False)]
        annotation_subset_two.annotations = [a for a in annotations_set_two if
                                             a.type == compare_type and not a.metadata.get('system', dict()).get(
                                                 'system', False)]
        # create 2d dataframe with annotation id as names and set all to -1 -> not calculated
        if match_wrong_labels is True:
            matches = Matchers.general_match(matches=matches,
                                             first_set=annotation_subset_one,
                                             second_set=annotation_subset_two,
                                             match_type=compare_type,
                                             match_threshold=match_threshold,
                                             ignore_labels=ignore_labels,
                                             ignore_geometry=ignore_geometry,
                                             ignore_attributes=ignore_attributes)
        else:
            unique_labels = np.unique([a.label for a in annotation_subset_one] +
                                      [a.label for a in annotation_subset_two])
            for label in unique_labels:
                first_set = [a for a in annotation_subset_one if a.label == label]
                second_set = [a for a in annotation_subset_two if a.label == label]
                if compare_type == entities.AnnotationType.CLASSIFICATION:
                    matches = Matchers.general_match(matches=matches,
                                                     first_set=first_set,
                                                     second_set=second_set,
                                                     match_type=compare_type,
                                                     match_threshold=match_threshold,
                                                     ignore_labels=ignore_labels,
                                                     ignore_attributes=ignore_attributes,
                                                     ignore_geometry=True
                                                     )
                else:
                    matches = Matchers.general_match(matches=matches,
                                                     first_set=first_set,
                                                     second_set=second_set,
                                                     match_type=compare_type,
                                                     match_threshold=match_threshold,
                                                     ignore_labels=ignore_labels,
                                                     ignore_geometry=ignore_geometry,
                                                     ignore_attributes=ignore_attributes
                                                     )

        if len(matches) == 0:
            continue
        all_scores.extend(matches.to_df()['annotation_score'])
        final_results[compare_type] = Results(matches=matches,
                                              annotation_type=compare_type)
        true_positives += final_results[compare_type].summary()['n_annotations_matched_total']
        false_positives += final_results[compare_type].summary()['n_annotations_unmatched_set_two']
        false_negatives += final_results[compare_type].summary()['n_annotations_unmatched_set_one']

    final_results['total_mean_score'] = mean_or_nan(all_scores)
    final_results['precision'] = \
        true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else 0
    final_results['recall'] = \
        true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0
    return final_results


def calculate_annotation_score(annot_collection_1: Union[dl.AnnotationCollection, List[dl.Annotation]],
                               annot_collection_2: Union[dl.AnnotationCollection, List[dl.Annotation]],
                               ignore_labels=False,  # for mean annotation calculation
                               ignore_geometry=False,  # for mean annotation calculation
                               ignore_attributes=False,  # for mean annotation calculation
                               match_wrong_labels=True,
                               include_confusion=True,
                               match_threshold=0.5,
                               compare_types=None,
                               score_types=None) -> List[Score]:
    """
    Creates Scores from comparing two annotation lists.

    The first annotation collection is considered the reference, and the second collection is the set for comparing.
    If we switch the order of the annotation collections, the scores remain the same but the user id context changes.

    :param annot_collection_1: dl.AnnotationCollection or list of annotations
    :param annot_collection_2: dl.AnnotationCollection or list of annotations
    :param ignore_labels: bool, True means every annotation will be cross-compared regardless of label classification (optional)
    :param ignore_geometry: bool, ignore iou score in mean. for classification (optional)
    :param ignore_attributes: bool, ignore attribute score in mean (optional)
    :param match_wrong_labels: bool, ignore attribute score in mean (optional)
    :param include_confusion: bool, True means label confusion scores will be calculated (optional)
    :param match_threshold: float, threshold for considering two annotations a "match" (optional)
    :param compare_types: dl.AnnotationType entity or string for the annotation types to be compared (optional)
    :param score_types: dl.ScoreType entity or string for the score types to be calculated (e.g. "annotation_iou") (optional)
    :return: list of Score entities
    """
    if score_types is None:
        score_types = [ScoreType.ANNOTATION_LABEL, ScoreType.ANNOTATION_IOU, ScoreType.ANNOTATION_ATTRIBUTE]
    if compare_types is None:
        compare_types = all_compare_types
    if not isinstance(score_types, list):
        score_types = [score_types]

    #######################
    # compare annotations #
    #######################
    results = measure_annotations(
        annotations_set_one=annot_collection_1,
        annotations_set_two=annot_collection_2,
        compare_types=compare_types,
        ignore_labels=ignore_labels,
        ignore_geometry=ignore_geometry,
        match_wrong_labels=match_wrong_labels,
        ignore_attributes=ignore_attributes,
        match_threshold=match_threshold)

    all_results = pd.DataFrame()
    for compare_type in compare_types:
        try:
            results_df = results[compare_type].to_df()
            all_results = pd.concat([all_results, results_df])
        except KeyError:
            continue

    #########################
    # create score entities #
    #########################
    logger.info(f'Creating scores for types: {score_types}')
    annotation_scores = []
    ###############################################
    # collect all scores for existing annotations #
    ###############################################
    for i, row in all_results.iterrows():
        if row['second_id'] is None:
            continue
        # add annotation overall
        annotation_scores.append(Score(type=ScoreType.ANNOTATION_OVERALL,
                                       value=row['annotation_score'],
                                       entity_id=row['second_id']))
        # add other types
        for score_type in score_types:
            annot_score = Score(type=score_type,
                                value=row[results_columns[score_type.value.lower()]],
                                entity_id=row['second_id'],
                                relative=row['first_id'])
            annotation_scores.append(annot_score)
    ##############################################
    # create user confusion from ALL annotations #
    ##############################################
    annotation_scores.append(Score(type=ScoreType.USER_CONFUSION,
                                   value=mean_or_default(arr=all_results.get('annotation_score', list()),
                                                         default=1)))
    ##############################################
    # create label confusion scores for this set #
    ##############################################
    if include_confusion is True:
        if all_results.shape[0] > 0:

            label_confusion_set = all_results[['first_label', 'second_label']]
            label_confusion_set = label_confusion_set.fillna('unmatched')

            label_confusion_summary = label_confusion_set.groupby(['first_label', 'second_label']).size().reset_index(
                name='counts')
            # print(label_confusion_summary)  # debug
            for i, row in label_confusion_summary.iterrows():
                confusion_score = Score(type=ScoreType.LABEL_CONFUSION,
                                        value=row['counts'],
                                        entity_id=row['second_label'],  # assignee label
                                        relative=row['first_label'])  # ground truth label
                annotation_scores.append(confusion_score)

    return annotation_scores


class Results:
    def __init__(self, matches, annotation_type):
        self.matches = matches
        self.annotation_type = annotation_type

    def to_df(self):
        return self.matches.to_df()

    def summary(self):
        df = self.matches.to_df()
        total_set_one = len(df['first_id'].dropna())
        total_set_two = len(df['second_id'].dropna())
        # each set unmatched is the number of Nones from the other set
        unmatched_set_one = df.shape[0] - total_set_two
        unmatched_set_two = df.shape[0] - total_set_one
        matched_set_one = total_set_one - unmatched_set_one
        matched_set_two = total_set_two - unmatched_set_two
        # sanity
        assert matched_set_one == matched_set_two, 'matched numbers are not the same'
        assert df['annotation_score'].shape[0] == (unmatched_set_one + unmatched_set_two + matched_set_one), \
            'mis-match number if scores and annotations'
        return {
            'annotation_type': self.annotation_type,
            'mean_annotations_scores': df['annotation_score'].mean(),
            'mean_geometries_scores': df['geometry_score'].mean(),
            'mean_attributes_scores': df['attribute_score'].mean(),
            'mean_labels_scores': df['label_score'].mean(),
            'n_annotations_set_one': total_set_one,
            'n_annotations_set_two': total_set_two,
            'n_annotations_total': total_set_one + total_set_two,
            'n_annotations_unmatched_set_one': unmatched_set_one,
            'n_annotations_unmatched_set_two': unmatched_set_two,
            'n_annotations_unmatched_total': unmatched_set_one + unmatched_set_two,
            'n_annotations_matched_set_one': matched_set_one,
            'n_annotations_matched_set_two': matched_set_two,
            'n_annotations_matched_total': matched_set_one + matched_set_two,
            'precision': matched_set_one / (matched_set_one + unmatched_set_two) if
            (matched_set_one + unmatched_set_two) != 0 else 0,
            'recall': matched_set_one / (matched_set_one + unmatched_set_one) if
            (matched_set_one + unmatched_set_one) != 0 else 0
        }


class Match:
    def __init__(self,
                 first_annotation_id, first_annotation_creator, first_annotation_label, first_annotation_confidence,
                 second_annotation_id, second_annotation_creator, second_annotation_label, second_annotation_confidence,
                 item_id,
                 # defaults
                 annotation_score: float = 0.,
                 attributes_score: float = 0.,
                 geometry_score: float = 0.,
                 label_score: float = 0.):
        """
        Save a match between two annotations with all relevant scores

        :param first_annotation_id: annotation ID of the annotation to compare to
        :param second_annotation_id: annotation ID of the annotation to compare
        :param annotation_score: score of the annotation match
        :param attributes_score: score of the attributes match
        :param geometry_score: score of the geometry match
        :param label_score: score of the label match
        """
        self.first_annotation_id = first_annotation_id
        self.first_annotation_creator = first_annotation_creator
        self.first_annotation_label = first_annotation_label
        self.first_annotation_confidence = first_annotation_confidence
        self.second_annotation_id = second_annotation_id
        self.second_annotation_creator = second_annotation_creator
        self.second_annotation_label = second_annotation_label
        self.second_annotation_confidence = second_annotation_confidence
        self.item_id = item_id
        self.annotation_score = annotation_score
        self.attributes_score = attributes_score
        # Replace the old annotation score
        self.geometry_score = geometry_score
        self.label_score = label_score

    def __repr__(self):
        return 'annotation: {:.2f}, attributes: {:.2f}, geometry: {:.2f}, label: {:.2f}'.format(
            self.annotation_score, self.attributes_score, self.geometry_score, self.label_score)


class Matches:
    def __init__(self):
        self.matches = list()
        self._annotations_raw_df = list()

    def __len__(self):
        return len(self.matches)

    def __repr__(self):
        return self.to_df().to_string()

    def to_df(self):
        results = list()
        for match in self.matches:
            results.append({
                'first_id': match.first_annotation_id,
                'first_creator': match.first_annotation_creator,
                'first_label': match.first_annotation_label,
                'first_confidence': match.first_annotation_confidence,
                'second_id': match.second_annotation_id,
                'second_creator': match.second_annotation_creator,
                'second_label': match.second_annotation_label,
                'second_confidence': match.second_annotation_confidence,
                'item_id': match.item_id,
                'annotation_score': match.annotation_score,
                'attribute_score': match.attributes_score,
                'geometry_score': match.geometry_score,
                'label_score': match.label_score,
            })
        df = pd.DataFrame(results)
        return df

    def add(self, match: Match):
        self.matches.append(match)

    def validate(self):
        first = list()
        second = list()
        for match in self.matches:
            if match.first_annotation_id in first:
                raise ValueError('duplication for annotation id {!r} in FIRST set'.format(match.first_annotation_id))
            if match.first_annotation_id is not None:
                first.append(match.first_annotation_id)
            if match.second_annotation_id in second:
                raise ValueError('duplication for annotation id {!r} in SECOND set'.format(match.second_annotation_id))
            if match.second_annotation_id is not None:
                second.append(match.second_annotation_id)
        return True

    def find(self, annotation_id, loc='first'):
        for match in self.matches:
            if loc == 'first':
                if match.first_annotation_id == annotation_id:
                    return match
            elif loc == 'second':
                if match.second_annotation_id == annotation_id:
                    return match
        raise ValueError('could not find annotation id {!r} in {}'.format(annotation_id, loc))


class Matchers:

    @staticmethod
    def calculate_iou_box(pts1, pts2, config):
        """
        Measures IOU for two lists of bounding box points
        :param pts1: ann.geo coordinates
        :param pts2: ann.geo coordinates
        :param config: arg added to match other functions
        :return: `float` how Intersection over Union of tho shapes
        """
        import shapely
        if int(shapely.__version__.split('.')[0]) < 2:
            from shapely.geometry import Polygon
        else:
            from shapely import Polygon

        if len(pts1) == 2:
            # regular box annotation (2 pts)
            pt1_left_top = [pts1[0][0], pts1[0][1]]
            pt1_right_top = [pts1[0][0], pts1[1][1]]
            pt1_right_bottom = [pts1[1][0], pts1[1][1]]
            pt1_left_bottom = [pts1[1][0], pts1[0][1]]
        else:
            # rotated box annotation (4 pts)
            pt1_left_top = pts1[0]
            pt1_right_top = pts1[3]
            pt1_left_bottom = pts1[1]
            pt1_right_bottom = pts1[2]

        poly_1 = Polygon([pt1_left_top,
                          pt1_right_top,
                          pt1_right_bottom,
                          pt1_left_bottom])

        if len(pts2) == 2:
            # regular box annotation (2 pts)
            pt2_left_top = [pts2[0][0], pts2[0][1]]
            pt2_right_top = [pts2[0][0], pts2[1][1]]
            pt2_right_bottom = [pts2[1][0], pts2[1][1]]
            pt2_left_bottom = [pts2[1][0], pts2[0][1]]
        else:
            # rotated box annotation (4 pts)
            pt2_left_top = pts2[0]
            pt2_right_top = pts2[3]
            pt2_left_bottom = pts2[1]
            pt2_right_bottom = pts2[2]

        poly_2 = Polygon([pt2_left_top,
                          pt2_right_top,
                          pt2_right_bottom,
                          pt2_left_bottom])
        if poly_1.intersects(poly_2):
            iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        else:
            iou = 0
        return iou

    @staticmethod
    def calculate_iou_classification(class1, class2, config):
        """
        Measure the accuracy of classification labels
        :param class1: `str` classification label
        :param class2: `str` classification label
        :param config: arg added to match other functions
        :return: `float` how Intersection over Union of tho shapes
        """
        return 1 if class1 == class2 else 0

    @staticmethod
    def calculate_iou_polygon(pts1, pts2, config):
        """
        Measures IOU for two lists of polygon points
        :param pts1: ann.geo coordinates
        :param pts2: ann.geo coordinates
        :param config: arg added to match other functions
        :return: `float` how Intersection over Union of tho shapes
        """
        # from shapely import Polygon
        import cv2

        if pts1.shape[0] == 0 or pts2.shape[0] == 0:
            # one of the polygons has not points
            return 0
        # # using shapley
        # poly_1 = Polygon(pts1)
        # poly_2 = Polygon(pts2)
        # iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area

        # # using opencv
        width = int(np.ceil(np.max(np.concatenate((pts1[:, 0], pts2[:, 0]))))) + 10
        height = int(np.ceil(np.max(np.concatenate((pts1[:, 1], pts2[:, 1]))))) + 10
        mask1 = np.zeros((height, width))
        mask2 = np.zeros((height, width))
        mask1 = cv2.drawContours(
            image=mask1,
            contours=[pts1.round().astype(int)],
            contourIdx=-1,
            color=1,
            thickness=-1,
        )
        mask2 = cv2.drawContours(
            image=mask2,
            contours=[pts2.round().astype(int)],
            contourIdx=-1,
            color=1,
            thickness=-1,
        )
        iou = np.sum((mask1 + mask2) == 2) / np.sum((mask1 + mask2) > 0)
        if np.sum((mask1 + mask2) > 2):
            assert False
        return iou

    @staticmethod
    def calculate_iou_semantic(mask1, mask2, config):
        joint_mask = mask1 + mask2
        return np.sum(np.sum(joint_mask == 2) / np.sum(joint_mask > 0))

    @staticmethod
    def calculate_iou_point(pt1, pt2, config):
        """
        pt is [x,y]
        normalizing to score between [0, 1] -> 1 is the exact match
        if same point score is 1
        at about 20 pix distance score is about 0.5, 100 goes to 0

        x = np.arange(int(diag))
        y = np.exp(-1 / diag * 20 * x)
        plt.figure()
        plt.plot(x, y)

        :param pt1: point 1
        :param pt2: point 2
        :param config: arg added to match other functions
        :return: calculated point iou

        """
        height = config.get('height', 500)
        width = config.get('width', 500)
        diag = np.sqrt(height ** 2 + width ** 2)
        # 20% of the image diagonal tolerance (empirically). need to
        return np.exp(-1 / diag * 20 * np.linalg.norm(np.asarray(pt1) - np.asarray(pt2)))

    @staticmethod
    def calculate_iou_cube():
        pass

    @staticmethod
    def match_attributes(attributes1, attributes2):
        """
        Returns IoU of the attributes. If both are empty, it's a perfect match (returns 1).
        0: no matching
        1: perfect attributes match
        """
        if type(attributes1) is not type(attributes2):
            logger.warning('attributes are not same type: {}, {}'.format(type(attributes1), type(attributes2)))
            return 0

        if attributes1 is None and attributes2 is None:
            return 1

        if isinstance(attributes1, dict) and isinstance(attributes2, dict):
            # convert to list
            attributes1 = ['{}-{}'.format(key, val) for key, val in attributes1.items()]
            attributes2 = ['{}-{}'.format(key, val) for key, val in attributes2.items()]

        intersection = set(attributes1).intersection(set(attributes2))
        union = set(attributes1).union(attributes2)
        if len(union) == 0:
            # if there is no union - there are no attributes at all
            return 1
        return len(intersection) / len(union)

    @staticmethod
    def match_labels(label1, label2):
        """
        Returns 1 in one of the labels in substring of the second
        """
        return int(label1 in label2 or label2 in label1)

    @staticmethod
    def general_match(matches: Matches,
                      first_set: entities.AnnotationCollection,
                      second_set: entities.AnnotationCollection,
                      match_type,
                      match_threshold: float,
                      ignore_attributes=False,
                      ignore_labels=False,
                      ignore_geometry=False):
        """
        Finds all matches between two sets of annotations
        :param matches: Matches object to populate
        :param first_set: `AnnotationCollection` or list
        :param second_set: `AnnotationCollection` or list
        :param match_type: type of annotation to match (e.g. box, semantic, etc.)
        :param match_threshold: threshold for including a match
        :param ignore_attributes: ignore attribute score for final annotation score (optional)
        :param ignore_labels: ignore label when comparing - take also wrong label as a match (optional)
        :param ignore_geometry: ignore geometry when comparing - for classification (optional)
        :return: Matches object
        """
        if ignore_geometry is True and ignore_labels is True and ignore_attributes is True:
            raise ValueError('Cant compare annotation with all ignore flags set to True, must choose at least one')
        annotation_type_to_func = {
            entities.AnnotationType.BOX: Matchers.calculate_iou_box,
            entities.AnnotationType.CLASSIFICATION: Matchers.calculate_iou_classification,
            entities.AnnotationType.SEGMENTATION: Matchers.calculate_iou_semantic,
            entities.AnnotationType.POLYGON: Matchers.calculate_iou_polygon,
            entities.AnnotationType.POINT: Matchers.calculate_iou_point,
        }
        df = pd.DataFrame(data=-1 * np.ones((len(second_set), len(first_set))),
                          columns=[a.id for a in first_set],
                          index=[a.id for a in second_set])
        for annotation_one in first_set:
            for annotation_two in second_set:
                if match_type not in annotation_type_to_func:
                    raise ValueError('unsupported type: {}'.format(match_type))
                if df[annotation_one.id][annotation_two.id] == -1:
                    try:
                        config = {'height': annotation_one._item.height if annotation_one._item is not None else 500,
                                  'width': annotation_one._item.width if annotation_one._item is not None else 500}
                        # df.loc[row, col]
                        df.loc[annotation_two.id, annotation_one.id] = annotation_type_to_func[match_type](
                            annotation_one.geo,
                            annotation_two.geo,
                            config)
                    except ZeroDivisionError:
                        logger.warning(
                            'Found annotations with area=0!: annotations ids: {!r}, {!r}'.format(annotation_one.id,
                                                                                                 annotation_two.id))
                        df[annotation_one.id][annotation_two.id] = 0
        # for debug - save the annotations scoring matrix
        matches._annotations_raw_df.append(df.copy())

        # go over all matches
        while True:
            # take max IoU score, list the match and remove annotations' ids from columns and rows
            # keep doing that until no more matches or lower than match threshold
            max_cell = df.max().max()
            if max_cell < match_threshold or np.isnan(max_cell):
                break
            row_index, col_index = np.where(df == max_cell)
            row_index = row_index[0]
            col_index = col_index[0]
            first_annotation_id = df.columns[col_index]
            second_annotation_id = df.index[row_index]
            first_annotation = [a for a in first_set if a.id == first_annotation_id][0]
            second_annotation = [a for a in second_set if a.id == second_annotation_id][0]
            geometry_score = df.iloc[row_index, col_index]
            attribute_score = Matchers.match_attributes(attributes1=first_annotation.attributes,
                                                        attributes2=second_annotation.attributes)
            labels_score = Matchers.match_labels(label1=first_annotation.label,
                                                 label2=second_annotation.label)
            match_scores = list()
            if ignore_geometry is False:
                match_scores.append(geometry_score)
            if ignore_attributes is False:
                match_scores.append(attribute_score)
            if ignore_labels is False:
                match_scores.append(labels_score)
            annotation_score = float(np.mean(match_scores))
            matches.add(match=Match(
                first_annotation_id=first_annotation_id,
                first_annotation_creator=first_annotation.creator,
                first_annotation_label=first_annotation.label,
                first_annotation_confidence=first_annotation.metadata.get('user', dict()).get('model', dict()).get(
                    'confidence', 1),
                second_annotation_id=second_annotation_id,
                second_annotation_creator=second_annotation.creator,
                second_annotation_label=second_annotation.label,
                second_annotation_confidence=second_annotation.metadata.get('user', dict()).get('model', dict()).get(
                    'confidence', 1),
                geometry_score=geometry_score,
                annotation_score=annotation_score,
                label_score=labels_score,
                attributes_score=attribute_score,
                item_id=second_annotation.item_id
            ))
            df.drop(index=second_annotation_id, inplace=True)
            df.drop(columns=first_annotation_id, inplace=True)
        # add un-matched
        for second_id in df.index:
            second_annotation = [a for a in second_set if a.id == second_id][0]
            matches.add(match=Match(first_annotation_id=None,
                                    first_annotation_creator=None,
                                    first_annotation_label=None,
                                    first_annotation_confidence=None,
                                    second_annotation_id=second_id,
                                    second_annotation_creator=second_annotation.creator,
                                    second_annotation_label=second_annotation.label,
                                    second_annotation_confidence=
                                    second_annotation.metadata.get('user', dict()).get('model', dict()).get(
                                        'confidence', 1),
                                    item_id=second_annotation.item_id
                                    ))
        for first_id in df.columns:
            first_annotation = [a for a in first_set if a.id == first_id][0]
            matches.add(match=Match(
                first_annotation_id=first_id,
                first_annotation_creator=first_annotation.creator,
                first_annotation_label=first_annotation.label,
                first_annotation_confidence=first_annotation.metadata.get('user', dict()).get('model', dict()).get(
                    'confidence', 1),
                second_annotation_id=None,
                second_annotation_creator=None,
                second_annotation_label=None,
                second_annotation_confidence=None,
                item_id=first_annotation.item_id
            ))

        return matches
