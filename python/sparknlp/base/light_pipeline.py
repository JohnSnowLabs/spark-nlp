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
            annotations.append(Annotation(annotation.annotatorType(),
                                          annotation.begin(),
                                          annotation.end(),
                                          annotation.result(),
                                          annotation.metadata(),
                                          annotation.embeddings
                                          )
                               )
        return annotations

    def fullAnnotate(self, target, optional_target=""):
        """Annotates the data provided into `Annotation` type results.

        The data should be either a list or a str.

        Parameters
        ----------
        target : list or str
            The data to be annotated
        optional_target: str
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

            for row in self._lightPipeline.fullAnnotateJava(target):
                kas = {}
                for atype, annotations in row.items():
                    kas[atype] = self._annotation_from_java(annotations)
                result.append(kas)

        else:
            if type(target) is list or type(optional_target) is list:
                raise TypeError("target and optional_target for annotation must be 'str'")

            full_annotations = self._lightPipeline.fullAnnotateJava(target, optional_target)
            kas = {}
            for atype, annotations in full_annotations.items():
                kas[atype] = self._annotation_from_java(annotations)
            result.append(kas)

        return result

    def annotate(self, target, optional_target=""):
        """Annotates the data provided, extracting the results.

        The data should be either a list or a str.

        Parameters
        ----------
        target : list or str
            The data to be annotated
        optional_target: str
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
            annotations = self._lightPipeline.annotateJava(target)
        else:
            if type(target) is list or  type(optional_target) is list:
                raise TypeError("target and optional_target for annotation must be 'str'")
            annotations = self._lightPipeline.annotateJava(target, optional_target)

        if type(target) is str:
            result = reformat(annotations)
        elif type(target) is list:
            result = list(map(lambda a: reformat(a), list(annotations)))
        else:
            raise TypeError("target for annotation may be 'str' or 'list'")

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

