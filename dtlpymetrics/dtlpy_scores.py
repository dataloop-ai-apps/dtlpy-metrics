import dtlpy as dl
from enum import Enum

from dtlpy import exceptions, entities, miscellaneous, _api_reference, repositories
from dtlpy.services.api_client import ApiClient


class ScoreType(Enum):
    ANNOTATION_IOU = 'annotation_iou'
    ANNOTATION_LABEL = 'annotation_label'
    ANNOTATION_OVERALL = 'annotation_overall'
    ITEM_OVERALL = 'item_overall'
    USER_OVERALL = 'user_overall'


class Score(entities.DlEntity):
    # platform
    id: str = entities.DlProperty(location=['id'], _type=str)
    url: str = entities.DlProperty(location=['url'], _type=str)
    created_at: str = entities.DlProperty(location=['createdAt'], _type=str)
    updated_at: str = entities.DlProperty(location=['updatedAt'], _type=str)
    creator: str = entities.DlProperty(location=['creator'], _type=str)
    # score
    entity_id: str = entities.DlProperty(location=['entityId'], _type=str)
    value: str = entities.DlProperty(location=['value'], _type=float)
    type: str = entities.DlProperty(location=['type'], _type=str)
    # context
    context: str = entities.DlProperty(location=['context'], _type=dict)
    project_id: str = entities.DlProperty(location=['context', 'projectId'], _type=str)
    dataset_id: str = entities.DlProperty(location=['context', 'datasetId'], _type=str)
    task_id: str = entities.DlProperty(location=['context', 'taskId'], _type=str)
    user_id: str = entities.DlProperty(location=['context', 'userId'], _type=str)
    assignment_id: str = entities.DlProperty(location=['context', 'assignmentId'], _type=str)
    item_id: str = entities.DlProperty(location=['context', 'itemId'], _type=str)
    model_id: str = entities.DlProperty(location=['context', 'modelId'], _type=str)
    relative: str = entities.DlProperty(location=['context', 'relative'], _type=str)

    def to_json(self):
        _json = self._dict.copy()
        return _json

    @classmethod
    def from_json(cls, _json: dict):
        return cls(**_json)


class Scores:
    URL = '/scores'

    def __init__(self,
                 client_api: dl.ApiClient,
                 project: entities.Project = None,
                 project_id: str = None):
        self._project = project
        self._project_id = project_id
        self._client_api = client_api

    @property
    def project(self) -> entities.Project:
        if self._project is None and self._project_id is not None:
            # get from id
            self._project = repositories.Projects(client_api=self._client_api).get(project_id=self._project_id)
        if self._project is None:
            # try get checkout
            project = self._client_api.state_io.get('project')
            if project is not None:
                self._project = entities.Project.from_json(_json=project, client_api=self._client_api)
        if self._project is None:
            raise exceptions.PlatformException(
                error='2001',
                message='Cannot perform action WITHOUT Project entity in Datasets repository.'
                        ' Please checkout or set a project')
        assert isinstance(self._project, entities.Project)
        return self._project

    ###########
    # methods #
    ###########
    def get(self, score_id: str) -> Score:
        success, response = self._client_api.gen_request(req_type="GET",
                                                         path="{}/{}".format(self.URL, score_id))

        # exception handling
        if not success:
            raise exceptions.PlatformException(response)

        # return entity
        return Score.from_json(_json=response.json())

    def create(self, scores):
        if not isinstance(scores, list):
            raise ValueError(f'score input must be a list of dl.Score')
        payload = {'scores': [score.to_json() for score in scores]}
        success, response = self._client_api.gen_request(req_type="post",
                                                         json_req=payload,
                                                         path=self.URL)

        # exception handling
        if not success:
            raise exceptions.PlatformException(response)

        outputs = [Score.from_json(_json=r) for r in response.json()]
        # return entity
        return outputs

    def delete(self, context: dict):

        success, response = self._client_api.gen_request(req_type="delete",
                                                         path=self.URL,
                                                         json_req=context)

        # check response
        if success:
            return success
        else:
            raise exceptions.PlatformException(response)

    # def _list(self, filters: entities.Filters):
    #     """
    #     Get dataset items list This is a browsing endpoint, for any given path item count will be returned,
    #     user is expected to perform another request then for every folder item to actually get the its item list.
    #
    #     :param dtlpy.entities.filters.Filters filters: Filters entity or a dictionary containing filters parameters
    #     :return: json response
    #     """
    #     # prepare request
    #     success, response = self._client_api.gen_request(req_type="POST",
    #                                                      path="{}/query".format(self.URL),
    #                                                      json_req=filters.prepare(),
    #                                                      headers={'user_query': filters.user_query}
    #                                                      )
    #     if not success:
    #         raise exceptions.PlatformException(response)
    #     return response.json()
    #
    # def list(self, filters: entities.Filters = None) -> entities.PagedEntities:
    #     """
    #     List of features
    #
    #     :param dtlpy.entities.filters.Filters filters: Filters to query the features data
    #     :return: Pages object
    #     :rtype: dtlpy.entities.paged_entities.PagedEntities
    #     """
    #     # default filters
    #     if filters is None:
    #         filters = entities.Filters(resource=entities.FiltersResource.FEATURE, user_query=False)
    #     # assert type filters
    #     if not isinstance(filters, entities.Filters):
    #         raise exceptions.PlatformException(error='400',
    #                                            message='Unknown filters type: {!r}'.format(type(filters)))
    #     if filters.resource != entities.FiltersResource.FEATURE:
    #         raise exceptions.PlatformException(
    #             error='400',
    #             message='Filters resource must to be FiltersResource.FEATURE. Got: {!r}'.format(filters.resource))
    #     paged = entities.PagedEntities(items_repository=self,
    #                                    filters=filters,
    #                                    page_offset=filters.page,
    #                                    page_size=filters.page_size,
    #                                    client_api=self._client_api)
    #     paged.get_page()
    #     return paged
    # def _build_entities_from_response(self, response_items) -> miscellaneous.List[entities.Item]:
    #     pool = self._client_api.thread_pools(pool_name='entity.create')
    #     jobs = [None for _ in range(len(response_items))]
    #     # return triggers list
    #     for i_item, item in enumerate(response_items):
    #         jobs[i_item] = pool.submit(entities.Feature._protected_from_json,
    #                                    **{'client_api': self._client_api,
    #                                       '_json': item})
    #     # get all results
    #     results = [j.result() for j in jobs]
    #     # log errors
    #     _ = [logger.warning(r[1]) for r in results if r[0] is False]
    #     # return good jobs
    #     items = miscellaneous.List([r[1] for r in results if r[0] is True])
    #     return items


def tests():
    dl.setenv('rc')
    project = dl.projects.get(project_id='2cb9ae90-b6e8-4d15-9016-17bacc9b7bdf')
    task = dl.tasks.get(task_id='63bffffa8cac97275a31fd17')
    annotation = dl.annotations.get(annotation_id='6499302b6563931044046cbc')

    score = Score(type=ScoreType.ANNOTATION_IOU.value,
                  value=0.9,
                  entity_id=annotation.id,
                  task_id=task.id)
    print(score.to_json())

    scores = Scores(client_api=dl.client_api,
                    project=project)

    dl_scores = scores.create([score])
    print([d.id for d in dl_scores])

    scores.delete({'taskId': task.id,
                   'itemId': item.id})
