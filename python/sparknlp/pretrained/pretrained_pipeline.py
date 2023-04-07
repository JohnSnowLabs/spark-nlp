#  Copyright 2017-2022 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Contains classes for the PretrainedPipeline."""

from pyspark.ml import PipelineModel
from pyspark.sql import DataFrame

from sparknlp.base import LightPipeline
from sparknlp.pretrained.resource_downloader import ResourceDownloader


class PretrainedPipeline:
    """Loads a Represents a fully constructed and trained Spark NLP pipeline,
    ready to be used.

    This way, a whole pipeline can be defined in 1 line. Additionally, the
    :class:`.LightPipeline` version of the model can be retrieved with member
    :attr:`.light_model`.

    For more extended examples see the `Pipelines page
    <https://sparknlp.org/docs/en/pipelines>`_ and our `Github Model
    Repository <https://github.com/JohnSnowLabs/spark-nlp-models>`_  for
    available pipeline models.

    Parameters
    ----------
    name : str
        Name of the PretrainedPipeline. These can be gathered from the Pipelines
        Page.
    lang : str, optional
        Langauge of the model, by default 'en'
    remote_loc : str, optional
        Link to the remote location of the model (if it was already downloaded),
        by default None
    parse_embeddings : bool, optional
        Whether to parse embeddings, by default False
    disk_location : str , optional
        Path to locally stored PretrainedPipeline, by default None
    """

    def __init__(self, name, lang='en', remote_loc=None, parse_embeddings=False, disk_location=None):
        if not disk_location:
            self.model = ResourceDownloader().downloadPipeline(name, lang, remote_loc)
        else:
            self.model = PipelineModel.load(disk_location)
        self.light_model = LightPipeline(self.model, parse_embeddings)

    @staticmethod
    def from_disk(path, parse_embeddings=False):
        return PretrainedPipeline(None, None, None, parse_embeddings, path)

    def annotate(self, target, column=None):
        """Annotates the data provided, extracting the results.

        The data should be either a list or a str.

        Parameters
        ----------
        target : list or str
            The data to be annotated

        Returns
        -------
        List[dict] or dict
            The result of the annotation

        Examples
        --------
        >>> from sparknlp.pretrained import PretrainedPipeline
        >>> explain_document_pipeline = PretrainedPipeline("explain_document_dl")
        >>> result = explain_document_pipeline.annotate('U.N. official Ekeus heads for Baghdad.')
        >>> result.keys()
        dict_keys(['entities', 'stem', 'checked', 'lemma', 'document', 'pos', 'token', 'ner', 'embeddings', 'sentence'])
        >>> result["ner"]
        ['B-ORG', 'O', 'O', 'B-PER', 'O', 'O', 'B-LOC', 'O']
        """

        annotations = self.light_model.annotate(target)
        return annotations

    def fullAnnotate(self, target, optional_target=""):
        """Annotates the data provided into `Annotation` type results.

        The data should be either a list or a str.

        Parameters
        ----------
        target : list or str
            The data to be annotated

        Returns
        -------
        List[Annotation]
            The result of the annotation

        Examples
        --------
        >>> from sparknlp.pretrained import PretrainedPipeline
        >>> explain_document_pipeline = PretrainedPipeline("explain_document_dl")
        >>> result = explain_document_pipeline.fullAnnotate('U.N. official Ekeus heads for Baghdad.')
        >>> result[0].keys()
        dict_keys(['entities', 'stem', 'checked', 'lemma', 'document', 'pos', 'token', 'ner', 'embeddings', 'sentence'])
        >>> result[0]["ner"]
        [Annotation(named_entity, 0, 2, B-ORG, {'word': 'U.N'}),
        Annotation(named_entity, 3, 3, O, {'word': '.'}),
        Annotation(named_entity, 5, 12, O, {'word': 'official'}),
        Annotation(named_entity, 14, 18, B-PER, {'word': 'Ekeus'}),
        Annotation(named_entity, 20, 24, O, {'word': 'heads'}),
        Annotation(named_entity, 26, 28, O, {'word': 'for'}),
        Annotation(named_entity, 30, 36, B-LOC, {'word': 'Baghdad'}),
        Annotation(named_entity, 37, 37, O, {'word': '.'})]
        """
        annotations = self.light_model.fullAnnotate(target, optional_target)
        return annotations

    def fullAnnotateImage(self, path_to_image):
        """Annotates the data provided into `Annotation` type results.

        The data should be either a list or a str.

        Parameters
        ----------
        path_to_image : list or str
            Source path of image, list of paths to images

        Returns
        -------
        List[AnnotationImage]
            The result of the annotation
        """
        pipeline = self.light_model
        return pipeline.fullAnnotateImage(path_to_image)

    def transform(self, data):
        """Transforms the input dataset with Spark.

        Parameters
        ----------
        data : :class:`pyspark.sql.DataFrame`
            input dataset

        Returns
        -------
        :class:`pyspark.sql.DataFrame`
            transformed dataset
        """
        return self.model.transform(data)
