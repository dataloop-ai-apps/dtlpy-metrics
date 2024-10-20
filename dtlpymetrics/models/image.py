import os
import dtlpy as dl
import pandas as pd


def get_model_scores_df(dataset: dl.Dataset, model: dl.Model) -> pd.DataFrame:
    """
    Retrieves the dataframe for all the scores for a given model on a dataset via a hidden csv file.
    :param dataset: Dataset where the model was evaluated
    :param model: Model entity
    :return: matched_annots_df: dataframe of all annotations in ground truth and model predictions
    """
    file_name = f'{model.id}.csv'
    local_path = os.path.join(os.getcwd(), '.dataloop', file_name)
    filters = dl.Filters(field='name', values=file_name)
    filters.add(field='hidden', values=True)
    pages = dataset.items.list(filters=filters)

    if pages.items_count > 0:
        for item in pages.all():
            item.download(local_path=local_path)
    else:
        raise ValueError(
            f'No matched annotations file found for model {model.id} on dataset {dataset.id}. Please evaluate model on the dataset first.')

    model_scores_df = pd.read_csv(local_path)
    return model_scores_df


def calculate_model_item_score(model_scores: pd.DataFrame):
    """
    Calculate scores for each item that a model predicts on

    :param model_scores: DataFrame of model scores
    :return:
    """

    pass
