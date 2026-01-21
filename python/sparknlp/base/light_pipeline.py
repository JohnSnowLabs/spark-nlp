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

import sparknlp.internal as _internal
from sparknlp.annotation import Annotation
from sparknlp.annotation_audio import AnnotationAudio
from sparknlp.annotation_image import AnnotationImage
from sparknlp.common import AnnotatorApproach, AnnotatorModel
from sparknlp.internal import AnnotatorTransformer


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

    def __init__(self, pipelineModel, parse_embeddings=False, output_cols=None):
        """
        Parameters
        ----------
        pipelineModel : PipelineModel
            The fitted Spark NLP pipeline model.
        parse_embeddings : bool, optional
            Whether to parse embeddings.
        output_cols : list[str], optional
            List of output columns to return in results (optional).
        """
        if output_cols is None:
            output_cols = []

        self.pipeline_model = pipelineModel
        self.parse_embeddings = parse_embeddings
        self.output_cols = output_cols

        self._lightPipeline = _internal._LightPipeline(pipelineModel, parse_embeddings, output_cols).apply()

    def _validateStagesInputCols(self, stages):
        annotator_types = self._getAnnotatorTypes(stages)
        for stage in stages:
            if isinstance(stage, AnnotatorApproach) or isinstance(stage, AnnotatorModel):
                input_cols = stage.getInputCols()
                if type(input_cols) == str:
                    input_cols = [input_cols]
                input_annotator_types = stage.inputAnnotatorTypes + stage.optionalInputAnnotatorTypes
                for input_col in input_cols:
                    annotator_type = annotator_types.get(input_col)
                    if annotator_type is None or annotator_type not in input_annotator_types:
                        raise TypeError(f"Wrong or missing inputCols annotators in {stage.uid}"
                                        f" Make sure such annotator exist in your pipeline,"
                                        f" with the right output names and that they have following annotator types:"
                                        f" {input_annotator_types}")

    def _skipPipelineValidation(self, stages):
        exceptional_pipeline = [stage for stage in stages if self._skipStageValidation(stage)]
        if len(exceptional_pipeline) >= 1:
            return True
        else:
            return False

    def _skipStageValidation(self, stage):
        return hasattr(stage, 'skipLPInputColsValidation') and stage.skipLPInputColsValidation

    def _getAnnotatorTypes(self, stages):
        annotator_types = {}
        for stage in stages:
            if hasattr(stage, 'getOutputCols'):
                output_cols = stage.getOutputCols()
                for output_col in output_cols:
                    annotator_types[output_col] = stage.outputAnnotatorType
            elif isinstance(stage, AnnotatorApproach) or isinstance(stage, AnnotatorModel) or\
                    isinstance(stage, AnnotatorTransformer):
                if stage.outputAnnotatorType is not None:
                    annotator_types[stage.getOutputCol()] = stage.outputAnnotatorType
        return annotator_types

    def _annotationFromJava(self, java_annotations):
        annotations = []
        for annotation in java_annotations:

            index = annotation.toString().index("(")
            annotation_type = annotation.toString()[:index]

            if annotation_type == "AnnotationImage":
                result = self.__get_result(annotation)
                annotations.append(
                    AnnotationImage(annotation.annotatorType(),
                                    annotation.origin(),
                                    annotation.height(),
                                    annotation.width(),
                                    annotation.nChannels(),
                                    annotation.mode(),
                                    result,
                                    annotation.metadata())
                )
            elif annotation_type == "AnnotationAudio":
                result = self.__get_result(annotation)
                annotations.append(
                    AnnotationAudio(annotation.annotatorType(),
                                    result,
                                    annotation.metadata())
                )
            else:
                if self.parse_embeddings:
                    embeddings = list(annotation.embeddings())
                else:
                    embeddings = []
                annotations.append(
                    Annotation(annotation.annotatorType(),
                               annotation.begin(),
                               annotation.end(),
                               annotation.result(),
                               annotation.metadata(),
                               embeddings)
                )
        return annotations

    @staticmethod
    def __get_result(annotation):
        try:
            result = list(annotation.result())
        except TypeError:
            result = []

        return result

    def fullAnnotate(self, *args, **kwargs):
        """
        Annotate and return full Annotation objects.

        Supports both:
          - fullAnnotate(text: str)
          - fullAnnotate(texts: list[str])
          - fullAnnotate(ids: list[int], texts: list[str])

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
        if "target" in kwargs:
            args = (kwargs["target"],) + args
        if "optional_target" in kwargs:
            args = args + (kwargs["optional_target"],)

        stages = self.pipeline_model.stages
        if not self._skipPipelineValidation(stages):
            self._validateStagesInputCols(stages)

        input_type = self.__detectInputType(args)

        if input_type == "ids_texts":
            ids, texts = args
            results = self._lightPipeline.fullAnnotateWithIdsJava(ids, texts)
            return [self.__buildStages(r) for r in results]

        if input_type == "qa":
            question, context = args
            return self.__fullAnnotateQuestionAnswering(question, context)

        if input_type == "text":
            target = args[0]
            return self.__fullAnnotateText(target)

        if input_type == "audio":
            audios = args[0]
            return self.__fullAnnotateAudio(audios)

        if input_type == "image":
            images = args[0]
            return self.fullAnnotateImage(images)

        raise TypeError(
            "Unsupported input for fullAnnotate(). Expected: "
            "(text: str | list[str]), "
            "(ids: list[int], texts: list[str]), "
            "(question: str, context: str), "
            "(audio: list[float] | list[list[float]]), or "
            "(image_path: str | list[str])."
        )

    @staticmethod
    def __isTextInput(target):
        if type(target) is str:
            return True
        elif type(target) is list and type(target[0]) is str:
            return True
        else:
            return False

    @staticmethod
    def __isAudioInput(target):
        if type(target) is list and type(target[0]) is float:
            return True
        elif type(target) is list and type(target[0]) is list and type(target[0][0]) is float:
            return True
        else:
            return False

    def __fullAnnotateText(self, target):

        if self.__isPath(target):
            result = self.fullAnnotateImage(target)
            return result
        else:
            result = []
            if type(target) is str:
                target = [target]

            for annotations_result in self._lightPipeline.fullAnnotateJava(target):
                result.append(self.__buildStages(annotations_result))
            return result

    def __isPath(self, target):
        if type(target) is list:
            target = target[0]

        if target.find("/") < 0:
            return False
        else:
            is_valid_file = _internal._ResourceHelper_validFile(target).apply()
            return is_valid_file

    def __fullAnnotateAudio(self, audios):
        result = []
        if type(audios[0]) is float:
            annotations_dict = self._lightPipeline.fullAnnotateSingleAudioJava(audios)
            result.append(self.__buildStages(annotations_dict))
        else:
            full_annotations = self._lightPipeline.fullAnnotateAudiosJava(audios)
            for annotations_dict in full_annotations:
                result.append(self.__buildStages(annotations_dict))

        return result

    def __fullAnnotateQuestionAnswering(self, question, context):
        result = []
        if type(question) is str and type(context) is str:
            annotations_dict = self._lightPipeline.fullAnnotateJava(question, context)
            result.append(self.__buildStages(annotations_dict))
        else:
            full_annotations = self._lightPipeline.fullAnnotateJava(question, context)
            for annotations_dict in full_annotations:
                result.append(self.__buildStages(annotations_dict))

        return result

    def fullAnnotateImage(self, path_to_image, text=None):
        """Annotates the data provided into `Annotation` type results.

        The data should be either a list or a str.

        Parameters
        ----------
        path_to_image : list or str
            Source path of image, list of paths to images

        text: list or str, optional
           Optional list or str of texts. If None, defaults to empty list if path_to_image is a list, or empty string if path_to_image is a string.

        Returns
        -------
        List[AnnotationImage]
            The result of the annotation
        """
        if not isinstance(path_to_image, (str, list)):
            raise TypeError("argument for path_to_image must be 'str' or 'list[str]'")

        if text is None:
            text = "" if isinstance(path_to_image, str) else []

        if type(path_to_image) != type(text):
            raise ValueError("`path_to_image` and `text` must be of the same type")

        stages = self.pipeline_model.stages
        if not self._skipPipelineValidation(stages):
            self._validateStagesInputCols(stages)

        if isinstance(path_to_image, str):
            path_to_image = [path_to_image]
            text = [text]

        result = []

        for image_result in self._lightPipeline.fullAnnotateImageJava(path_to_image, text):
            result.append(self.__buildStages(image_result))

        return result


    def __buildStages(self, annotations_result):
        stages = {}
        for annotator_type, annotations in annotations_result.items():
            stages[annotator_type] = self._annotationFromJava(annotations)
        return stages


    def annotate(self, *args, **kwargs):
        """
        Annotate text(s) or text(s) with IDs using the LightPipeline.

        Supports both:
          - annotate(text: str)
          - annotate(texts: list[str])
          - annotate(ids: list[int], texts: list[str])

        Returns
        -------
        list[dict[str, list[str]]]

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

        if "target" in kwargs:
            args = (kwargs["target"],) + args
        if "optional_target" in kwargs:
            args = args + (kwargs["optional_target"],)

        stages = self.pipeline_model.stages
        if not self._skipPipelineValidation(stages):
            self._validateStagesInputCols(stages)

        input_type = self.__detectInputType(args)

        if input_type == "ids_texts":
            ids, texts = args
            annotations = self._lightPipeline.annotateWithIdsJava(ids, texts)
            results = list(map(lambda a: reformat(a), list(annotations)))
            return results

        if input_type == "qa":
            question, context = args
            if isinstance(question, list) and isinstance(context, list):
                annotations = self._lightPipeline.annotateJava(question, context)
                results = list(map(lambda a: reformat(a), list(annotations)))
                return results
            else:
                annotations = self._lightPipeline.annotateJava(question, context)
                results = reformat(annotations)
                return results

        if input_type == "text":
            target = args[0]
            if isinstance(target, str):
                annotations = self._lightPipeline.annotateJava(target)
                results = reformat(annotations)
                return results
            else:
                annotations = self._lightPipeline.annotateJava(target)
                results = list(map(lambda a: reformat(a), list(annotations)))
                return results

        raise TypeError(
            "Unsupported input for annotate(). Expected: "
            "(text: str | list[str]), "
            "(ids: list[int], texts: list[str]), "
            "or (question: str, context: str)."
        )

    def __detectInputType(self, args):
        """
        Determine the input type pattern for fullAnnotate().
        Returns one of: 'ids_texts', 'qa', 'text', 'audio', 'image', or 'unknown'.
        """
        if len(args) == 2:
            a1, a2 = args

            # (ids, texts)
            if (
                    isinstance(a1, list)
                    and all(isinstance(i, int) for i in a1)
                    and isinstance(a2, list)
                    and all(isinstance(t, str) for t in a2)
            ):
                return "ids_texts"

            # (question, context)
            if isinstance(a1, str) and isinstance(a2, str):
                return "qa"

            # (questions[], contexts[])
            if (
                    isinstance(a1, list)
                    and all(isinstance(q, str) for q in a1)
                    and isinstance(a2, list)
                    and all(isinstance(c, str) for c in a2)
            ):
                return "qa"

        elif len(args) == 1:
            a1 = args[0]

            if not isinstance(a1, (str, list)):
                return "unknown"

            # 🧩 Case 1: plain string
            if isinstance(a1, str):
                if self.__isPath(a1):
                    return "image"
                if self.__isTextInput(a1):
                    return "text"
                return "unknown"

            # 🧩 Case 2: list — ensure homogeneous types
            if isinstance(a1, list) and len(a1) > 0:

                # Guard clause — mixed or invalid types
                if not all(isinstance(x, (str, float, list)) for x in a1):
                    return "unknown"

                # Text list
                if all(isinstance(x, str) for x in a1) and self.__isTextInput(a1):
                    return "text"

                # Audio list
                if all(isinstance(x, float) for x in a1) or (
                        all(isinstance(x, list) for x in a1) and all(isinstance(i, float) for sub in a1 for i in sub)
                ):
                    return "audio"

                # Image list (only strings allowed)
                if all(isinstance(x, str) for x in a1) and all("/" in x for x in a1):
                    return "image"

            return "unknown"

        return "unknown"


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
        transformed_df = self.pipeline_model.transform(dataframe)

        if self.output_cols:
            original_cols = dataframe.columns
            filtered_cols = list(dict.fromkeys(original_cols + self.output_cols))
            transformed_df = transformed_df.select(*filtered_cols)

        return transformed_df

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
