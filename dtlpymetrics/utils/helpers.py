import os
import dtlpy as dl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dtlpymetrics.dtlpy_scores import Score
from typing import List


def check_if_video(item: dl.Item):
    if item.metadata.get('system', dict()):
        item_mimetype = item.metadata['system'].get('mimetype', None)
        is_video = 'video' in item_mimetype
    else:
        is_video = False
    return is_video


# def get_annotations_from_frames(annotations: List[dl.Annotation]):
#     """
#     Split video annotations by frame
#     @param annotations: a list of annotations from a video item
#     @return: annotations_by_frame, dict with frame number as key and list of annotations as value
#     """
#     annotations_by_frame = dict()
#     for annotation in annotations:
#         # then, in each frame, split by assignment (dict in dict)
#         for frame, frame_annotation in annotation.frames.items():
#             frame_annotation.annotation.set_frame(frame=frame)
#             if frame not in annotations_by_frame:
#                 annotations_by_frame[frame] = [frame_annotation.annotation]
#             else:
#                 annotations_by_frame[frame].append(frame_annotation.annotation)
#     return annotations_by_frame


def add_score_context(score: Score, **kwargs):
    """
    Adds context to a score
    :param kwargs: context to add to score
    :return: score with context
    """
    score = kwargs.get('score')
    context = kwargs.get('context')
    if context is None:
        context = dict()
    score.context = context
    return score


# @scorer.add_function(display_name='Plot annotators confusion matrix')
def plot_matrix(item_title, filename, matrix_to_plot, axis_labels):
    # annotators matrix plot, per item
    mask = np.zeros_like(matrix_to_plot, dtype=bool)
    # mask[np.triu_indices_from(mask)] = True

    sns.set(rc={'figure.figsize': (20, 10),
                'axes.facecolor': 'white'},
            font_scale=2)
    sns_plot = sns.heatmap(matrix_to_plot,
                           annot=True,
                           mask=mask,
                           cmap='Blues',
                           xticklabels=axis_labels,
                           yticklabels=axis_labels,
                           vmin=0,
                           vmax=1)
    sns_plot.set(title=item_title)
    sns_plot.set_yticklabels(sns_plot.get_yticklabels(), rotation=270)

    fig = sns_plot.get_figure()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fig.savefig(filename)
    plt.close()

    return filename
