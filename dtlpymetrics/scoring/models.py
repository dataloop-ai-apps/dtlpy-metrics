from typing import List
import datetime
import logging
import json
import os
import pathlib
import tqdm

import dtlpy as dl
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from matplotlib import pyplot as plt

from ..utils import (
    add_score_context,
    all_compare_types,
    measure_annotations,
    mean_or_default,
    check_if_video,
    calculate_annotation_score,
)
from ..dtlpy_scores import Score, ScoreType, Scores

logger = logging.getLogger("scoring-and-metrics")


def create_model_score(
    dataset: dl.Dataset,
    model: dl.Model,
    filters: dl.Filters,
    ignore_labels=False,
    match_threshold=0.01,
    compare_types=None,
) -> dl.Model:
    """
    Creates scores for a set of model predictions compared against ground truth annotations in a given dataset.

    :param dataset: Dataset associated with the ground truth annotations
    :param model: Model for evaluating predictions
    :param filters: DQL Filter for retrieving the test items
    :param ignore_labels: bool, True means every annotation will be cross-compared regardless of label (optinal)
    :param match_threshold: float, threshold for matching annotations together (optional)
    :param compare_types: annotation types to compare (optional)
    :return: dl.Model
    """
    if dataset is None:
        raise ValueError("No dataset provided, please provide a dataset.")
    if model is None:
        raise ValueError("No model provided, please provide a model.")
    if model.name is None:
        raise ValueError("No model name found for the second set of annotations, please provide model name.")
    if compare_types is None:
        compare_types = all_compare_types
    if not isinstance(compare_types, list):
        if compare_types not in model.output_type:
            raise ValueError(f"Annotation type {compare_types} does not match model output type {model.output_type}")
        compare_types = [compare_types]

    # TODO use export to download the zip and take the annotation from there
    logger.info("Downloading dataset annotations...")
    json_path = dataset.download_annotations(
        filters=filters, annotation_options=[dl.VIEW_ANNOTATION_OPTIONS_JSON], overwrite=True
    )
    item_json_files = list(pathlib.Path(json_path).rglob("*.json"))
    if len(item_json_files) == 0:
        raise KeyError("No items found in the dataset, please check the dataset and filters.")

    ########################################
    # Create list of item annotation lists #
    ########################################
    annotation_sets_by_item = dict()
    n_gt_annotations = 0
    n_md_annotations = 0
    pbar = tqdm.tqdm(item_json_files)
    pbar.set_description("Loading annotations from items... ")
    for item_file in pbar:
        with open(item_file, "r") as f:
            data = json.load(f)
        item = dl.Item.from_json(_json=data, client_api=dataset._client_api, dataset=dataset)
        item_id = data["id"]
        collection: dl.AnnotationCollection = dl.AnnotationCollection.from_json(_json=data["annotations"], item=item)
        item_annots_1 = []
        item_annots_2 = []
        for annotation in collection:
            if annotation.metadata.get("user", {}).get("model") is None:
                # GT annotation (no model in metadata)
                item_annots_1.append(annotation)
            elif annotation.metadata.get("user", {}).get("model", {}).get("name", "") == model.name:
                # annotation came from the evaluated model
                item_annots_2.append(annotation)
        annotation_sets_by_item[item_id] = {"gt": item_annots_1, "model": item_annots_2}
        n_gt_annotations += len(item_annots_1)
        n_md_annotations += len(item_annots_2)
        pbar.update()

    logger.info(f"Found {len(annotation_sets_by_item)} GT item.")
    logger.info(f"Found {n_gt_annotations} GT annotations.")
    logger.info(f"Found {n_md_annotations} model annotations.")
    #########################################################
    # Compare annotations and return concatenated dataframe #
    #########################################################
    all_results = pd.DataFrame()
    pbar = tqdm.tqdm(total=len(annotation_sets_by_item), desc="Calculating metrics...")
    pool = ThreadPoolExecutor(max_workers=32)

    def calc_single(w_item_id, w_annotation_sets):
        try:
            set_1_item_annotations = w_annotation_sets["gt"]
            set_2_item_annotations = w_annotation_sets["model"]
            if not (len(set_1_item_annotations) == 0 and len(set_2_item_annotations) == 0):
                results = measure_annotations(
                    annotations_set_one=set_1_item_annotations,
                    annotations_set_two=set_2_item_annotations,
                    match_threshold=match_threshold,
                    # default 0.01 to get all possible matches
                    ignore_labels=ignore_labels,
                    compare_types=compare_types,
                )
                for compare_type in compare_types:
                    try:
                        results_df = results[compare_type].to_df()
                    except KeyError:
                        continue
                    results_df["item_id"] = [w_item_id] * results_df.shape[0]
                    results_df["annotation_type"] = [compare_type] * results_df.shape[0]
                    all_results[w_item_id] = results_df
        finally:
            pbar.update()

    all_results = dict()
    for item_id, annotation_sets in annotation_sets_by_item.items():
        pool.submit(calc_single, w_item_id=item_id, w_annotation_sets=annotation_sets)
    pool.shutdown()
    all_results = pd.concat(list(all_results.values()), ignore_index=True)

    ###############################################
    # Save results to csv for IOU/label/attribute #
    ###############################################
    # TODO save via scoring when ready
    # file format "/.modelscores/modelId.csv"
    all_results["model_id"] = [model.id] * all_results.shape[0]
    all_results["dataset_id"] = [dataset.id] * all_results.shape[0]

    if not os.path.isdir(os.path.join(os.getcwd(), "../.dataloop")):
        os.mkdir(os.path.join(os.getcwd(), "../.dataloop"))
    scores_filepath = os.path.join(os.getcwd(), "../.dataloop", f"{model.id}.csv")

    all_results.to_csv(scores_filepath, index=False)
    item = dataset.items.upload(local_path=scores_filepath, remote_path=f"/.modelscores", overwrite=True)
    logger.info(f"Successfully created model scores and saved as item {item.id}.")

    # This is a workaround for uploading interpolated precision-recall for 10 iou levels
    calc_and_upload_interpolation(model=model, dataset=dataset)
    return model


def calc_item_model_score(item: dl.Item, model: dl.Model, score_types=None, upload=True) -> List[Score]:
    """
    Creates scores for a set of model predictions compared against ground truth annotations in a given dataset.

    :param item: Item entity
    :param model: Model entity
    :param score_types: annotation types to compare (optional)
    :param upload: bool, default False means scores will be saved locally (optional)
    :return: list of Scores
    """
    logger.info(f"Starting scoring for item: {item.id} with model: {model.id}")

    ###################################
    # collect annotations for sorting #
    ###################################
    annotations = item.annotations.list()

    # Sort annotations into ground truth and model predictions
    gt_annotations = []
    model_annotations = []

    for annotation in annotations:
        if annotation.metadata.get("user", {}).get("model") is None:
            # GT annotation (no model in metadata)
            gt_annotations.append(annotation)
        elif annotation.metadata.get("user", {}).get("model", {}).get("name", "") == model.name:
            # annotation came from the evaluated model
            model_annotations.append(annotation)

    #########################################
    # sort annotations and calculate scores #
    #########################################

    is_video = check_if_video(item=item)

    if is_video is True:  # video items
        annotations_by_frame = _split_video_to_frames(annotations=annotations, item=item, model=model)

        all_scores = get_video_scores(
            annotations_by_frame=annotations_by_frame,
            grouped_annotations={"gt": gt_annotations, "model": model_annotations},
            item=item,
            model=model,
            score_types=score_types,
        )
    else:  # image items
        all_scores = get_image_scores(
            grouped_annotations={"gt": gt_annotations, "model": model_annotations},
            item=item,
            model=model,
            score_types=score_types,
        )

    # overall item score is an average of all overall annotation scores
    item_overall = [score.value for score in all_scores if score.type == ScoreType.ANNOTATION_OVERALL.value]

    item_score = Score(
        type=ScoreType.ITEM_OVERALL,
        value=mean_or_default(arr=item_overall, default=1),
        entity_id=item.id,
        item_id=item.id,
        dataset_id=item.dataset.id,
    )
    all_scores.append(item_score)

    #############################
    # upload scores to platform #
    #############################
    if upload is True:
        logger.info(f"Deleting all scores with context item ID: {item.id} and model ID: {model.id}")
        dl_scores = Scores(client_api=dl.client_api)
        dl_scores.delete(context={"itemId": item.id, "modelId": model.id})
        dl_scores = dl_scores.create(all_scores)
        logger.info(f"Uploaded {len(dl_scores)} scores to platform.")

    return all_scores


def _split_video_to_frames(annotations: dl.AnnotationCollection, item: dl.Item, model: dl.Model) -> dict:
    """
    Split video annotations frame by frame and sort by type (gt/model)

    :param annotations: Collection of annotations to split
    :param item: dl.Item entity
    :param model: dl.Model entity
    :return: dict of annotations by frame and type
    """
    # get max frames for all annotations
    try:
        num_frames = int(item.metadata["system"]["ffmpeg"]["nb_read_frames"])
    except KeyError:
        end_frames = [ann.end_frame for ann in annotations]
        num_frames = np.max(end_frames) + 1

    # Get all annotation slices for each frame
    all_annotation_slices = dict()
    for f in range(num_frames):
        all_annotation_slices[f] = annotations.get_frame(frame_num=f)

    # Sort annotations by frame and type (gt/model)
    annotations_by_frame = {}
    for frame, annotation_slices in all_annotation_slices.items():
        frame_gt = []
        frame_model = []
        for annotation_slice in annotation_slices:
            if annotation_slice.metadata.get("user", {}).get("model") is None:
                # GT annotation (no model in metadata)
                frame_gt.append(annotation_slice)
            elif annotation_slice.metadata.get("user", {}).get("model", {}).get("name", "") == model.name:
                # annotation came from the evaluated model
                frame_model.append(annotation_slice)
        annotations_by_frame[frame] = {"gt": frame_gt, "model": frame_model}

    return annotations_by_frame


def calc_precision_recall(
    dataset_id: str, model_id: str, iou_threshold=0.01, method_type=None, each_label=True, n_points=None
) -> pd.DataFrame:
    """
    Internal function for calculating  precision recall values for model predictions, for a given metric threshold.
    :param dataset_id: str dataset ID
    :param model_id: str model ID
    :param iou_threshold: float Threshold for accepting matched annotations as a true positive (optional)
    :param method_type: str method for calculating precision and recall. Options are: every_point and n_point_interpolated (optional)
    :param each_label: bool calculate precision recall for each one of the labels (optional)
    :param n_points: int number of points to interpolate in case of n point interpolation (optional)
    :return: dataframe with all the points to plot for the dataset and individual labels
    """
    if method_type is None:
        method_type = "n_point_interpolation"
    elif method_type not in ["every_point", "n_point_interpolation"]:
        raise ValueError(f"Unknown method type {method_type}. Options are: every_point and n_point_interpolated")

    ################################
    # get matched annotations data #
    ################################
    # TODO use scoring once available
    model_filename = f"{model_id}.csv"
    items_filters = dl.Filters(field="hidden", values=True)
    items_filters.add(field="name", values=model_filename)
    dataset = dl.datasets.get(dataset_id=dataset_id)
    items = list(dataset.items.list(filters=items_filters).all())
    if len(items) == 0:
        raise ValueError(f"No scores found for model ID {model_id}. Please evaluate model on the dataset first.")
    elif len(items) > 1:
        raise ValueError(f"Found {len(items)} items with name {model_id}.")
    else:
        scores_file = items[0].download(overwrite=True)
        logger.info(f"Downloaded scores file to {scores_file}")

    scores = pd.read_csv(scores_file)
    labels = dataset.labels
    label_names = [label.tag for label in labels]
    if len(label_names) == 0:
        label_names = list(pd.concat([scores.first_label, scores.second_label]).dropna().drop_duplicates())

    ##############################
    # calculate precision/recall #
    ##############################
    logger.info("Calculating precision/recall")

    dataset_points = {
        "iou_threshold": {},
        "data": {},  # "dataset" or "label"
        "label_name": {},  # label name or NA,
        "precision": {},
        "recall": {},
        "confidence": {},
    }

    # first set is GT
    num_gts = sum(scores.first_id.notna())

    detections = scores[scores.second_id.notna()].copy()

    detections.sort_values("second_confidence", inplace=True, ascending=False, ignore_index=True)
    detections["true_positives"] = detections["geometry_score"] >= iou_threshold
    detections["false_positives"] = detections["geometry_score"] < iou_threshold

    # get dataset-level precision/recall
    dataset_fps = np.cumsum(detections["false_positives"])
    dataset_tps = np.cumsum(detections["true_positives"])
    dataset_recall = dataset_tps / num_gts
    dataset_precision = np.divide(dataset_tps, (dataset_fps + dataset_tps))

    # detections.to_csv(index=False)  # DEBUG

    if method_type == "every_point":
        [_, dataset_plot_precision, dataset_plot_recall, dataset_plot_confidence] = _every_point_curve(
            recall=list(dataset_recall),
            precision=list(dataset_precision),
            confidence=list(detections["second_confidence"]),
        )
    else:
        [_, dataset_plot_precision, dataset_plot_recall, dataset_plot_confidence] = _n_point_interpolated_curve(
            recall=list(dataset_recall),
            precision=list(dataset_precision),
            confidence=list(detections["second_confidence"]),
            n_points=n_points,
        )

    dataset_points["iou_threshold"] = [iou_threshold] * len(dataset_plot_precision)
    dataset_points["data"] = ["dataset"] * len(dataset_plot_precision)
    dataset_points["label_name"] = ["_NA"] * len(dataset_plot_precision)
    dataset_points["precision"] = dataset_plot_precision
    dataset_points["recall"] = dataset_plot_recall
    dataset_points["confidence"] = dataset_plot_confidence
    dataset_points["dataset_name"] = [dataset.name] * len(dataset_plot_precision)

    dataset_df = pd.DataFrame(dataset_points).drop_duplicates()

    ##########################################
    # calculate label-level precision/recall #
    ##########################################
    if each_label is True:
        all_labels = pd.DataFrame(columns=dataset_df.columns)
        label_points = {key: {} for key in dataset_points}
        for label_name in list(set(label_names)):
            label_detections = detections.loc[
                (detections.first_label == label_name) | (detections.second_label == label_name)
            ].copy()
            if label_detections.shape[0] == 0:
                label_plot_precision = [0]
                label_plot_recall = [0]
                label_plot_confidence = [0]
            else:
                label_detections.sort_values("second_confidence", inplace=True, ascending=False, ignore_index=True)

                label_fps = np.cumsum(label_detections["false_positives"])
                label_tps = np.cumsum(label_detections["true_positives"])
                label_recall = label_tps / num_gts
                label_precision = np.divide(label_tps, (label_fps + label_tps))

                if method_type == "every_point":
                    [_, label_plot_precision, label_plot_recall, label_plot_confidence] = _every_point_curve(
                        recall=list(label_recall),
                        precision=list(label_precision),
                        confidence=list(label_detections["second_confidence"]),
                    )
                else:
                    [_, label_plot_precision, label_plot_recall, label_plot_confidence] = _n_point_interpolated_curve(
                        recall=list(label_recall),
                        precision=list(label_precision),
                        confidence=list(label_detections["second_confidence"]),
                        n_points=n_points,
                    )

            label_points["iou_threshold"] = [iou_threshold] * len(label_plot_precision)
            label_points["data"] = ["label"] * len(label_plot_precision)
            label_points["label_name"] = [label_name] * len(label_plot_precision)
            label_points["precision"] = label_plot_precision
            label_points["recall"] = label_plot_recall
            label_points["confidence"] = label_plot_confidence
            label_points["dataset_name"] = [dataset.name] * len(label_plot_precision)

            label_df = pd.DataFrame(label_points).drop_duplicates()
            all_labels = pd.concat([all_labels, label_df])
        ####################
        # combine all data #
        ####################
        plot_points = pd.concat([dataset_df, all_labels])
    else:
        plot_points = dataset_df
    logger.info("Done")
    # plot_points.to_csv(os.path.join(os.getcwd(), 'plot_points.csv'), index=False)     # DEBUG

    return plot_points


def plot_precision_recall(plot_points: pd.DataFrame, dataset_name=None, label_names=None, local_path=None):
    """
    Plot precision recall curve for a given metric threshold

    :param plot_points: dict generated from calculate_precision_recall with all the points to plot by label and
     the entire dataset. keys include: confidence threshold, iou threshold, dataset levels precision, recall, and
     confidence, and label-level precision, recall and confidence
    :param dataset_name: name of dataset to plot in legend (optional)
    :param label_names: list of label names to plot (optional)
    :param local_path: path to save plot (optional)
    :return: directory path where plots are saved
    """
    if local_path is None:
        root_dir = os.getcwd().split("dtlpymetrics")[0]
        save_dir = os.path.join(root_dir, "dtlpymetrics", "../.dataloop")
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))
    else:
        save_dir = os.path.join(local_path)

    ###################
    # plot by dataset #
    ###################
    logger.info("Plotting precision recall")

    plt.figure()
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    # plot each label separately
    dataset_points = plot_points[plot_points["data"] == "dataset"]
    dataset_legend = f"{dataset_points['dataset_id'].iloc[0]}" if dataset_name is None else dataset_name

    plt.plot(dataset_points["recall"], dataset_points["precision"], label=dataset_legend)

    plt.legend(loc="upper right")

    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.grid()

    # plot the dataset level
    plot_filename = f"dataset_precision_recall_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png"
    save_path = os.path.join(save_dir, plot_filename)
    plt.savefig(save_path)
    # plt.close()
    logger.info(f"Saved dataset precision recall plot to {save_path}")

    #################
    # plot by label #
    #################
    all_labels = plot_points[plot_points["data"] == "label"]

    if (label_names is None) or (bool(label_names) is False):
        label_names = all_labels["label_name"].copy().drop_duplicates()

    plt.figure()
    plt.xlim(0, 1.1)
    plt.ylim(0, 1.1)

    # plot each label separately
    for label_name in label_names:
        label_points = all_labels[all_labels["label_name"] == label_name].copy()

        plt.plot(label_points["recall"], label_points["precision"], label=label_name)

    plt.legend(loc="upper right")
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.grid()

    # plot the dataset level
    plot_filename = f"label_precision_recall_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.png"
    save_path = os.path.join(save_dir, plot_filename)
    plt.savefig(save_path)
    # plt.close()
    logger.info(f"Saved labels precision recall plot to {save_path}")

    return save_dir


def get_false_negatives(model: dl.Model, dataset: dl.Dataset) -> pd.DataFrame:
    """
    Retrieves the dataframe for all the scores for a given model on a dataset via a hidden csv file,
    and returns a dataframe with the properties of all the false negatives.
    :param model: Model entity
    :param dataset: Dataset where the model was evaluated
    :return: DataFrame with all the false negatives
    """
    file_name = f"{model.id}.csv"
    local_path = os.path.join(os.getcwd(), "../.dataloop", file_name)
    filters = dl.Filters(field="name", values=file_name)
    filters.add(field="hidden", values=True)
    pages = dataset.items.list(filters=filters)

    if pages.items_count > 0:
        for item in pages.all():
            item.download(local_path=local_path)
    else:
        raise ValueError(
            f"No scores file found for model {model.id} on dataset {dataset.id}. Please evaluate model on the dataset first."
        )

    scores_df = pd.read_csv(local_path)

    ########################
    # list false negatives #
    ########################
    model_fns = dict()
    annotation_to_item_map = {ann_id: item_id for ann_id, item_id in zip(scores_df.first_id, scores_df.itemId)}
    fn_annotation_ids = scores_df[scores_df.second_id.isna()].first_id
    print(f"model: {model.name} with {len(fn_annotation_ids)} false negative")
    fn_items_ids = np.unique([annotation_to_item_map[ann_id] for ann_id in fn_annotation_ids])
    for i_id in fn_items_ids:
        if i_id not in model_fns:
            i_id: dl.Item
            url = dl.client_api._get_resource_url(
                "projects/{}/datasets/{}/items/{}".format(dataset.project.id, dataset.id, i_id)
            )
            model_fns[i_id] = {"itemId": i_id, "url": url}
        model_fns[i_id].update({model.name: True})

    model_fn_df = pd.DataFrame(model_fns.values()).fillna(False)
    model_fn_df.to_csv(os.path.join(os.getcwd(), f"{model.name}_false_negatives.csv"))

    return model_fn_df


def calc_and_upload_interpolation(model: dl.Model, dataset: dl.Dataset):
    """
    Calculate precision recall for a model and dataset, and upload the interpolated points to the dataset
    :param model: dl.Model
    :param dataset: dl.Dataset
    :return: True
    """
    figures = dict()
    for iou_th in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        df = calc_precision_recall(
            model_id=model.id,
            dataset_id=dataset.id,
            method_type="n_point_interpolation",
            n_points=201,
            iou_threshold=iou_th,
        )
        dataset_points = df[df["label_name"] == "_NA"]
        # TODO plot each label separately
        recall = dataset_points["recall"]
        precision = dataset_points["precision"]
        confidence = dataset_points["confidence"]
        figures[iou_th] = {
            "recall": recall.to_list(),
            "precision": precision.to_list(),
            "confidence": confidence.to_list(),
        }
    filepath = os.path.join(os.getcwd(), "../.dataloop", f"{model.id}-interpolated.json")
    with open(filepath, "w") as f:
        json.dump(figures, f)
    item = dataset.items.upload(local_path=filepath, remote_path=f"/.modelscores", overwrite=True)

    return True


def _every_point_curve(recall: list, precision: list, confidence: list):
    """
    Calculate precision-recall curve from a list of precision & recall values
    :param recall: list of recall values
    :param precision: list of precision values
    :param confidence: list of confidence values
    :return: list of average precision all values, precision points, recall points, confidence points
    """
    recall_points = np.concatenate([[0], recall, [recall[-1]]])
    precision_points = np.concatenate([[0], precision, [0]])
    confidence_points = np.concatenate([[confidence[0]], confidence, [confidence[-1]]])

    # find the maximum precision between each recall value, backwards
    for i in range(len(precision_points) - 1, 0, -1):
        precision_points[i - 1] = max(precision_points[i - 1], precision_points[i])
        # print(precision_points[i-1])  # DEBUG

    # build the simplified recall list, removing values when the precision doesn't change
    recall_intervals = []
    for i in range(len(recall_points) - 1):
        if recall_points[1 + i] != recall_points[i]:
            recall_intervals.append(i + 1)
    # plt.plot(recall_points, precision_points)  # DEBUG

    # use the recall intervals to calculate the average precision / area under the curve
    avg_precis = 0
    for i in recall_intervals:
        avg_precis = avg_precis + np.sum((recall_points[i] - recall_points[i - 1]) * precision_points[i])

    return [
        avg_precis,
        precision_points[0 : len(precision_points)],
        recall_points[0 : len(precision_points)],
        confidence_points[0 : len(precision_points)],
    ]


def _n_point_interpolated_curve(recall: list, precision: list, confidence: list, n_points=201):
    """
    Calculate precision-recall curve from a list of precision & recall values, using n-points interpolation

    :param recall: list of recall values
    :param precision: list of precision values
    :param confidence: list of confidence values
    :param n_points: number of points to interpolate, default = 201
    :return: list of average precision all values, precision points, recall points, confidence points
    """
    recall_all = recall
    precision_all = precision
    confidence_all = confidence

    recall_intervals = np.linspace(1, 0, n_points)
    # recall_intervals = list(reversed(recall_intervals))

    rho_interpol = []  # the interpolated precision values for each interval range
    recall_valid = []
    conf_valid = []

    for recall_interval in recall_intervals:
        larger_recall = np.argwhere(recall_all[:] >= recall_interval).squeeze(axis=1)
        precision_max = 0
        confidence_min = 0
        if larger_recall.size != 0:
            precision_max = max([precision_all[i] for i in larger_recall])
            # print(f'precis max: {precision_max}, up to recall interval {recall_interval}')  # DEBUG
            confidence_min = confidence_all[list(precision_all).index(precision_max)]
        conf_valid.append(confidence_min)
        recall_valid.append(recall_interval)
        rho_interpol.append(precision_max)

    avg_precis = sum(rho_interpol) / n_points

    # make points plot-ready
    recall_points = np.concatenate([[recall_valid[0]], recall_valid, [recall_valid[-1]]])  # 1 to 0
    precision_points = np.concatenate([[rho_interpol[0]], rho_interpol, [rho_interpol[-1]]])  # 0 to 1
    confidence_points = np.concatenate([[conf_valid[0]], conf_valid, [conf_valid[-1]]])  # conf min to max

    cc = []
    for i in range(1, len(recall_points)):
        point_1 = (recall_points[i], precision_points[i - 1], confidence_points[i - 1])
        point_2 = (recall_points[i], precision_points[i], confidence_points[i])
        if point_1 not in cc:
            cc.append(point_1)
        if point_2 not in cc:
            cc.append(point_2)

    recall_plot = [i[0] for i in cc]
    precision_plot = [i[1] for i in cc]
    confidence_plot = [i[2] for i in cc]

    return [avg_precis, precision_plot, recall_plot, confidence_plot]


def get_image_scores(
    grouped_annotations: dict, item: dl.Item, model: dl.Model, score_types: list = None, match_threshold: float = 0.01
) -> list:
    """
    Calculate scores for an image item by comparing ground truth annotations with model predictions

    :param grouped_annotations: dict of annotations grouped by type (gt/model)
    :param item: dl.Item
    :param score_types: list of score types to be calculated (optional)
    :return all_scores: list of all annotation and item scores
    """
    ####################
    # calculate scores #
    ####################
    # compare between GT and model annotations and create Score entities
    all_scores = list()

    # Get GT and model annotations
    gt_annotations = grouped_annotations.get("gt", [])
    model_annotations = grouped_annotations.get("model", [])

    logger.info("Comparing GT annotations with model predictions")

    # Calculate pairwise scores between GT and model annotations
    pairwise_scores = calculate_annotation_score(
        annot_collection_1=gt_annotations,
        annot_collection_2=model_annotations,
        ignore_labels=False,
        ignore_attributes=True,
        ignore_geometry=True,
        match_threshold=match_threshold,
        score_types=score_types,
    )

    for score in pairwise_scores:
        updated_score = add_score_context(score=score, entity_id=score.entity_id, model_id=model.id, item_id=item.id)
        all_scores.append(updated_score)

    # accumulate label confusion
    confusion_scores = list()
    for i_score, score in reversed(list(enumerate(all_scores))):
        if score.type == ScoreType.LABEL_CONFUSION:
            confusion_scores.append(score)
            all_scores.pop(i_score)

    confusion_dict = dict()
    for score in confusion_scores:
        if score.entity_id not in confusion_dict:
            confusion_dict[score.entity_id] = dict()
        if score.relative not in confusion_dict[score.entity_id]:
            confusion_dict[score.entity_id][score.relative] = 0
        confusion_dict[score.entity_id][score.relative] += 1

    for entity_id, v in confusion_dict.items():
        for relative, count in v.items():
            all_scores.append(
                Score(
                    type=ScoreType.LABEL_CONFUSION,
                    value=count,
                    entity_id=entity_id,  # TODO check
                    relative=relative,
                    model_id=model.id,
                    item_id=item.id,
                )
            )

    # Calculate average overall score for each annotation id
    annotation_overalls = list()
    for i_score, score in reversed(list(enumerate(all_scores))):
        if score.type == ScoreType.ANNOTATION_OVERALL:
            annotation_overalls.append(score)
            all_scores.pop(i_score)

    unique_annotation_ids = np.unique([score.entity_id for score in annotation_overalls])
    for annotation_id in unique_annotation_ids:
        overalls = [score for score in annotation_overalls if score.entity_id == annotation_id]
        # this is a matching score between annotations
        # to make it a probability we will add the current self match as 1
        # for instance, if we had [A,A,B], and the current is A, the overall probability is 2/3
        overalls_values = [s.value for s in overalls]
        overalls_values.append(1)  # the match to the current annotation, this will it the probability

        # add joint overall (single one for each annotation
        all_scores.append(
            Score(
                type=ScoreType.ANNOTATION_OVERALL,
                value=mean_or_default(arr=overalls_values, default=0),
                entity_id=annotation_id,
                model_id=model.id,
                item_id=item.id,
            )
        )

    return all_scores


def get_video_scores(
    annotations_by_frame: dict, grouped_annotations: dict, item: dl.Item, model: dl.Model, score_types: list = None
):
    """
    Create scores for a video item by comparing ground truth annotations with model predictions

    :param annotations_by_frame: dict of annotations by frame, grouped by type (gt/model)
    :param grouped_annotations: dict of annotations grouped by type (gt/model)
    :param item: dl.Item
    :param model: dl.Model
    :param score_types: list of scores to calculate
    :return all_scores: list of all annotation and item scores
    """
    ####################
    # calculate scores #
    ####################
    all_scores_by_frame = dict()
    ann_ids = list()

    for frame, annots_by_type in annotations_by_frame.items():
        # Get GT and model annotations for this frame
        gt_annotations = grouped_annotations.get("gt", [])
        model_annotations = grouped_annotations.get("model", [])

        frame_scores = list()

        logger.info(f"Comparing GT annotations with model predictions for frame {frame}")

        # Calculate pairwise scores between GT and model annotations
        pairwise_scores = calculate_annotation_score(
            annot_collection_1=gt_annotations,
            annot_collection_2=model_annotations,
            ignore_labels=False,
            ignore_attributes=True,
            ignore_geometry=True,
            match_threshold=0.01,
            score_types=score_types,
        )

        for score in pairwise_scores:
            updated_score = add_score_context(
                score=score, entity_id=score.entity_id, model_id=model.id, item_id=item.id
            )
            frame_scores.append(updated_score)
            ann_ids.extend([ann.id for ann in gt_annotations])
            ann_ids.extend([ann.id for ann in model_annotations])

        all_scores_by_frame[frame] = frame_scores

    # After each frame's score is calculated, calculate the mean score across all frames
    all_scores = list()
    unique_annotation_ids = np.unique(ann_ids)

    # Handle confusion scores
    confusion_scores = list()
    for frame, scores in all_scores_by_frame.items():
        for score in scores:
            if score.type == ScoreType.LABEL_CONFUSION:
                confusion_scores.append(score)

    confusion_dict = dict()
    for score in confusion_scores:
        if score.entity_id not in confusion_dict:
            confusion_dict[score.entity_id] = dict()
        if score.relative not in confusion_dict[score.entity_id]:
            confusion_dict[score.entity_id][score.relative] = 0
        confusion_dict[score.entity_id][score.relative] += 1

    for entity_id, v in confusion_dict.items():
        for relative, count in v.items():
            all_scores.append(
                Score(
                    type=ScoreType.LABEL_CONFUSION,
                    value=count,
                    entity_id=entity_id,  # TODO check that it's a model label
                    relative=relative,
                    model_id=model.id,
                    item_id=item.id,
                )
            )

    # Calculate overall scores for each annotation across frames
    for annotation_id in unique_annotation_ids:
        annotation_frame_scores = [
            frame_score
            for frame_scores in all_scores_by_frame.values()
            for frame_score in frame_scores
            if frame_score.entity_id == annotation_id
        ]

        # Calculate mean scores across frames
        all_scores.append(
            Score(
                type=ScoreType.ANNOTATION_OVERALL,
                value=mean_or_default(
                    arr=[
                        score.value
                        for score in annotation_frame_scores
                        if score.type == ScoreType.ANNOTATION_OVERALL.value
                    ],
                    default=1,
                ),
                entity_id=annotation_id,
                model_id=model.id,
                item_id=item.id,
            )
        )
        all_scores.append(
            Score(
                type=ScoreType.ANNOTATION_LABEL,
                value=mean_or_default(
                    arr=[
                        score.value
                        for score in annotation_frame_scores
                        if score.type == ScoreType.ANNOTATION_LABEL.value
                    ],
                    default=1,
                ),
                entity_id=annotation_id,
                model_id=model.id,
                item_id=item.id,
            )
        )
        all_scores.append(
            Score(
                type=ScoreType.ANNOTATION_IOU,
                value=mean_or_default(
                    arr=[
                        score.value for score in annotation_frame_scores if score.type == ScoreType.ANNOTATION_IOU.value
                    ],
                    default=1,
                ),
                entity_id=annotation_id,
                model_id=model.id,
                item_id=item.id,
            )
        )
        all_scores.append(
            Score(
                type=ScoreType.ANNOTATION_ATTRIBUTE,
                value=mean_or_default(
                    arr=[
                        score.value
                        for score in annotation_frame_scores
                        if score.type == ScoreType.ANNOTATION_ATTRIBUTE.value
                    ],
                    default=1,
                ),
                entity_id=annotation_id,
                model_id=model.id,
                item_id=item.id,
            )
        )

    return all_scores
