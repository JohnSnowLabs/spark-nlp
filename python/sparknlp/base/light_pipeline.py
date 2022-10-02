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
"""Contains classes for the LightPipeline."""

from sparknlp.annotation import Annotation
import sparknlp.internal as _internal
from sparknlp.annotation_image import AnnotationImage


class LightPipeline:
    """Creates a LightPipeline from a Spark PipelineModel.

    LightPipeline is a Spark NLP specific Pipeline class equivalent to Spark
    ML Pipeline. The difference is that it’s execution does not hold to
    Spark principles, instead it computes everything locally (but in
    parallel) in order to achieve fast results when dealing with small
    amounts of data. This means, we do not input a Spark Dataframe, but a
    string or an Array of strings instead, to be annotated. To create Light
    Pipelines, you need to input an already trained (fit) Spark ML Pipeline.

    It’s :meth:`.transform` has now an alternative :meth:`.annotate`, which
    directly outputs the results.

    Parameters
    ----------
    pipelineModel : :class:`pyspark.ml.PipelineModel`
        The PipelineModel containing Spark NLP Annotators
    parse_embeddings : bool, optional
        Whether to parse embeddings, by default False

    Notes
    -----
    Use :meth:`.fullAnnotate` to also output the result as
    :class:`.Annotation`, with metadata.

    Examples
    --------
    >>> from sparknlp.base import LightPipeline
    >>> light = LightPipeline(pipeline.fit(data))
    >>> light.annotate("We are very happy about Spark NLP")
    {
        'document': ['We are very happy about Spark NLP'],
        'lemmas': ['We', 'be', 'very', 'happy', 'about', 'Spark', 'NLP'],
        'pos': ['PRP', 'VBP', 'RB', 'JJ', 'IN', 'NNP', 'NNP'],
        'sentence': ['We are very happy about Spark NLP'],
        'spell': ['We', 'are', 'very', 'happy', 'about', 'Spark', 'NLP'],
        'stems': ['we', 'ar', 'veri', 'happi', 'about', 'spark', 'nlp'],
        'token': ['We', 'are', 'very', 'happy', 'about', 'Spark', 'NLP']
    }
    """

    def __init__(self, pipelineModel, parse_embeddings=False):
        self.pipeline_model = pipelineModel
        self._lightPipeline = _internal._LightPipeline(pipelineModel, parse_embeddings).apply()

    @staticmethod
    def _annotation_from_java(java_annotations):
        annotations = []
        for annotation in java_annotations:

            index = annotation.toString().index("(")
            annotation_type = annotation.toString()[:index]

            if annotation_type == "AnnotationImage":
                annotations.append(
                    AnnotationImage(annotation.annotatorType(),
                                    annotation.origin(),
                                    annotation.height(),
                                    annotation.width(),
                                    annotation.nChannels(),
                                    annotation.mode(),
                                    list(annotation.result()),
                                    dict(annotation.metadata()))
                )
            else:
                annotations.append(
                    Annotation(annotation.annotatorType(),
                               annotation.begin(),
                               annotation.end(),
                               annotation.result(),
                               dict(annotation.metadata()),
                               [])
                )
        return annotations

    def fullAnnotate(self, target, optional_target=""):
        """Annotates the data provided into `Annotation` type results.

        The data should be either a list or a str.

        Parameters
        ----------
        target : list or str
            The data to be annotated
        optional_target: list or str
            Optional data to be annotated (currently used for Question Answering)

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
        result = []

        if optional_target == "":
            if type(target) is str:
               target = [target]
            elif type(target) is list and type(target[0]) is list:
                raise TypeError("target is a 1D list")
            else:
                raise TypeError("target for annotation must be 'str' or list")

            for annotations_result in self._lightPipeline.fullAnnotateJava(target):
                result.append(self.buildStages(annotations_result))

        else:
            if type(target) is str and type(optional_target) is str:
                annotations_dict = self._lightPipeline.fullAnnotateJava(target, optional_target)
                result.append(self.buildStages(annotations_dict))
            elif type(target) is list and type(optional_target) is list:
                if type(target[0]) is list or type(optional_target[0]) is list:
                    raise TypeError("target and optional_target is a 1D list")
                full_annotations = self._lightPipeline.fullAnnotateJava(target, optional_target)
                for annotations_dict in full_annotations:
                    result.append(self.buildStages(annotations_dict))
            else:
                raise TypeError("target and optional_target for annotation must be both 'str' or both lists")

        return result

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
        if type(path_to_image) is str:
            path_to_image = [path_to_image]

        if type(path_to_image) is list:
            result = []

            for image_result in self._lightPipeline.fullAnnotateImageJava(path_to_image):
                result.append(self.buildStages(image_result))

            return result
        else:
            raise TypeError("target for annotation may be 'str' or 'list'")

    def buildStages(self, annotations_result):
        stages = {}
        for annotator_type, annotations in annotations_result.items():
            stages[annotator_type] = self._annotation_from_java(annotations)
        return stages

    def annotate(self, target, optional_target=""):
        """Annotates the data provided, extracting the results.

        The data should be either a list or a str.

        Parameters
        ----------
        target : list or str
            The data to be annotated
        optional_target: list or str
            Optional data to be annotated (currently used for Question Answering)

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

        def reformat(annotations):
            return {k: list(v) for k, v in annotations.items()}

        if optional_target == "":
            if type(target) is str:
                annotations = self._lightPipeline.annotateJava(target)
                result = reformat(annotations)
            elif type(target) is list:
                if type(target[0]) is list:
                    raise TypeError("target is a 1D list")
                annotations = self._lightPipeline.annotateJava(target)
                result = list(map(lambda a: reformat(a), list(annotations)))
            else:
                raise TypeError("target for annotation must be 'str' or list")

        else:
            if type(target) is str and type(optional_target) is str:
                annotations = self._lightPipeline.annotateJava(target, optional_target)
                result = reformat(annotations)
            elif type(target) is list and type(optional_target) is list:
                if type(target[0]) is list or type(optional_target[0]) is list:
                    raise TypeError("target and optional_target is a 1D list")
                annotations = self._lightPipeline.annotateJava(target, optional_target)
                result = list(map(lambda a: reformat(a), list(annotations)))
            else:
                raise TypeError("target and optional_target for annotation must be both 'str' or both lists")

        return result

    def transform(self, dataframe):
        """Transforms a dataframe provided with the stages of the LightPipeline.

        Parameters
        ----------
        dataframe : :class:`pyspark.sql.DataFrame`
            The Dataframe to be transformed

        Returns
        -------
        :class:`pyspark.sql.DataFrame`
            The transformed DataFrame
        """
        return self.pipeline_model.transform(dataframe)

    def setIgnoreUnsupported(self, value):
        """Sets whether to ignore unsupported AnnotatorModels.

        Parameters
        ----------
        value : bool
            Whether to ignore unsupported AnnotatorModels.

        Returns
        -------
        LightPipeline
            The current LightPipeline
        """
        self._lightPipeline.setIgnoreUnsupported(value)
        return self

    def getIgnoreUnsupported(self):
        """Gets whether to ignore unsupported AnnotatorModels.

        Returns
        -------
        bool
            Whether to ignore unsupported AnnotatorModels.
        """
        return self._lightPipeline.getIgnoreUnsupported()
