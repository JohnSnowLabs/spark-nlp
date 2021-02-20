from abc import ABC

from pyspark import keyword_only
from pyspark.ml.wrapper import JavaEstimator
from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.ml.pipeline import Pipeline, PipelineModel, Estimator, Transformer
from sparknlp.common import AnnotatorProperties
from sparknlp.internal import AnnotatorTransformer, RecursiveEstimator, RecursiveTransformer

from sparknlp.annotation import Annotation
import sparknlp.internal as _internal


class LightPipeline:
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

    def fullAnnotate(self, target):
        result = []
        if type(target) is str:
            target = [target]
        for row in self._lightPipeline.fullAnnotateJava(target):
            kas = {}
            for atype, annotations in row.items():
                kas[atype] = self._annotation_from_java(annotations)
            result.append(kas)
        return result

    def annotate(self, target):

        def reformat(annotations):
            return {k: list(v) for k, v in annotations.items()}

        annotations = self._lightPipeline.annotateJava(target)

        if type(target) is str:
            result = reformat(annotations)
        elif type(target) is list:
            result = list(map(lambda a: reformat(a), list(annotations)))
        else:
            raise TypeError("target for annotation may be 'str' or 'list'")

        return result

    def transform(self, dataframe):
        return self.pipeline_model.transform(dataframe)

    def setIgnoreUnsupported(self, value):
        self._lightPipeline.setIgnoreUnsupported(value)
        return self

    def getIgnoreUnsupported(self):
        return self._lightPipeline.getIgnoreUnsupported()


class RecursivePipeline(Pipeline, JavaEstimator):
    @keyword_only
    def __init__(self, *args, **kwargs):
        super(RecursivePipeline, self).__init__(*args, **kwargs)
        self._java_obj = self._new_java_obj("com.johnsnowlabs.nlp.RecursivePipeline", self.uid)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    def _fit(self, dataset):
        stages = self.getStages()
        for stage in stages:
            if not (isinstance(stage, Estimator) or isinstance(stage, Transformer)):
                raise TypeError(
                    "Cannot recognize a pipeline stage of type %s." % type(stage))
        indexOfLastEstimator = -1
        for i, stage in enumerate(stages):
            if isinstance(stage, Estimator):
                indexOfLastEstimator = i
        transformers = []
        for i, stage in enumerate(stages):
            if i <= indexOfLastEstimator:
                if isinstance(stage, Transformer):
                    transformers.append(stage)
                    dataset = stage.transform(dataset)
                elif isinstance(stage, RecursiveEstimator):
                    model = stage.fit(dataset, pipeline=PipelineModel(transformers))
                    transformers.append(model)
                    if i < indexOfLastEstimator:
                        dataset = model.transform(dataset)
                else:
                    model = stage.fit(dataset)
                    transformers.append(model)
                    if i < indexOfLastEstimator:
                        dataset = model.transform(dataset)
            else:
                transformers.append(stage)
        return PipelineModel(transformers)


class RecursivePipelineModel(PipelineModel):

    def __init__(self, pipeline_model):
        super(PipelineModel, self).__init__()
        self.stages = pipeline_model.stages

    def _transform(self, dataset):
        for t in self.stages:
            if isinstance(t, HasRecursiveTransform):
                # drops current stage from the recursive pipeline within
                dataset = t.transform_recursive(dataset, PipelineModel(self.stages[:-1]))
            elif isinstance(t, AnnotatorProperties) and t.getLazyAnnotator():
                pass
            else:
                dataset = t.transform(dataset)
        return dataset


class HasRecursiveFit(RecursiveEstimator, ABC):
    pass


class HasRecursiveTransform(RecursiveTransformer):
    pass


class DocumentAssembler(AnnotatorTransformer):

    inputCol = Param(Params._dummy(), "inputCol", "input column name", typeConverter=TypeConverters.toString)
    outputCol = Param(Params._dummy(), "outputCol", "output column name", typeConverter=TypeConverters.toString)
    idCol = Param(Params._dummy(), "idCol", "column for setting an id to such string in row", typeConverter=TypeConverters.toString)
    metadataCol = Param(Params._dummy(), "metadataCol", "String to String map column to use as metadata", typeConverter=TypeConverters.toString)
    calculationsCol = Param(Params._dummy(), "calculationsCol", "String to Float vector map column to use as embeddigns and other representations", typeConverter=TypeConverters.toString)
    cleanupMode = Param(Params._dummy(), "cleanupMode", "possible values: disabled, inplace, inplace_full, shrink, shrink_full, each, each_full, delete_full", typeConverter=TypeConverters.toString)
    name = 'DocumentAssembler'

    @keyword_only
    def __init__(self):
        super(DocumentAssembler, self).__init__(classname="com.johnsnowlabs.nlp.DocumentAssembler")
        self._setDefault(outputCol="document", cleanupMode='disabled')

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCol(self, value):
        return self._set(inputCol=value)

    def setOutputCol(self, value):
        return self._set(outputCol=value)

    def setIdCol(self, value):
        return self._set(idCol=value)

    def setMetadataCol(self, value):
        return self._set(metadataCol=value)

    def setCalculationsCol(self, value):
        return self._set(metadataCol=value)

    def setCleanupMode(self, value):
        if value.strip().lower() not in ['disabled', 'inplace', 'inplace_full', 'shrink', 'shrink_full', 'each', 'each_full', 'delete_full']:
            raise Exception("Cleanup mode possible values: disabled, inplace, inplace_full, shrink, shrink_full, each, each_full, delete_full")
        return self._set(cleanupMode=value)


class TokenAssembler(AnnotatorTransformer, AnnotatorProperties):

    name = "TokenAssembler"
    preservePosition = Param(Params._dummy(), "preservePosition", "whether to preserve the actual position of the tokens or reduce them to one space", typeConverter=TypeConverters.toBoolean)

    @keyword_only
    def __init__(self):
        super(TokenAssembler, self).__init__(classname="com.johnsnowlabs.nlp.TokenAssembler")

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setPreservePosition(self, value):
        return self._set(preservePosition=value)


class Doc2Chunk(AnnotatorTransformer, AnnotatorProperties):

    chunkCol = Param(Params._dummy(), "chunkCol", "column that contains string. Must be part of DOCUMENT", typeConverter=TypeConverters.toString)
    startCol = Param(Params._dummy(), "startCol", "column that has a reference of where chunk begins", typeConverter=TypeConverters.toString)
    startColByTokenIndex = Param(Params._dummy(), "startColByTokenIndex", "whether start col is by whitespace tokens", typeConverter=TypeConverters.toBoolean)
    isArray = Param(Params._dummy(), "isArray", "whether the chunkCol is an array of strings", typeConverter=TypeConverters.toBoolean)
    failOnMissing = Param(Params._dummy(), "failOnMissing", "whether to fail the job if a chunk is not found within document. return empty otherwise", typeConverter=TypeConverters.toBoolean)
    lowerCase = Param(Params._dummy(), "lowerCase", "whether to lower case for matching case", typeConverter=TypeConverters.toBoolean)
    name = "Doc2Chunk"

    @keyword_only
    def __init__(self):
        super(Doc2Chunk, self).__init__(classname="com.johnsnowlabs.nlp.Doc2Chunk")
        self._setDefault(
            isArray=False
        )

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setChunkCol(self, value):
        return self._set(chunkCol=value)

    def setIsArray(self, value):
        return self._set(isArray=value)

    def setStartCol(self, value):
        return self._set(startCol=value)

    def setStartColByTokenIndex(self, value):
        return self._set(startColByTokenIndex=value)

    def setFailOnMissing(self, value):
        return self._set(failOnMissing=value)

    def setLowerCase(self, value):
        return self._set(lowerCase=value)


class Chunk2Doc(AnnotatorTransformer, AnnotatorProperties):

    name = "Chunk2Doc"

    @keyword_only
    def __init__(self):
        super(Chunk2Doc, self).__init__(classname="com.johnsnowlabs.nlp.Chunk2Doc")

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)


class Finisher(AnnotatorTransformer):

    inputCols = Param(Params._dummy(), "inputCols", "input annotations", typeConverter=TypeConverters.toListString)
    outputCols = Param(Params._dummy(), "outputCols", "output finished annotation cols", typeConverter=TypeConverters.toListString)
    valueSplitSymbol = Param(Params._dummy(), "valueSplitSymbol", "character separating annotations", typeConverter=TypeConverters.toString)
    annotationSplitSymbol = Param(Params._dummy(), "annotationSplitSymbol", "character separating annotations", typeConverter=TypeConverters.toString)
    cleanAnnotations = Param(Params._dummy(), "cleanAnnotations", "whether to remove annotation columns", typeConverter=TypeConverters.toBoolean)
    includeMetadata = Param(Params._dummy(), "includeMetadata", "annotation metadata format", typeConverter=TypeConverters.toBoolean)
    outputAsArray = Param(Params._dummy(), "outputAsArray", "finisher generates an Array with the results instead of string", typeConverter=TypeConverters.toBoolean)
    parseEmbeddingsVectors = Param(Params._dummy(), "parseEmbeddingsVectors", "whether to include embeddings vectors in the process", typeConverter=TypeConverters.toBoolean)

    name = "Finisher"

    @keyword_only
    def __init__(self):
        super(Finisher, self).__init__(classname="com.johnsnowlabs.nlp.Finisher")
        self._setDefault(
            cleanAnnotations=True,
            includeMetadata=False,
            outputAsArray=True,
            parseEmbeddingsVectors=False,
            valueSplitSymbol="#",
            annotationSplitSymbol="@"
        )

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCols(self, *value):
        if len(value) == 1 and type(value[0]) == list:
            return self._set(inputCols=value[0])
        else:
            return self._set(inputCols=list(value))

    def setOutputCols(self, *value):
        if len(value) == 1 and type(value[0]) == list:
            return self._set(outputCols=value[0])
        else:
            return self._set(outputCols=list(value))

    def setValueSplitSymbol(self, value):
        return self._set(valueSplitSymbol=value)

    def setAnnotationSplitSymbol(self, value):
        return self._set(annotationSplitSymbol=value)

    def setCleanAnnotations(self, value):
        return self._set(cleanAnnotations=value)

    def setIncludeMetadata(self, value):
        return self._set(includeMetadata=value)

    def setOutputAsArray(self, value):
        return self._set(outputAsArray=value)

    def setParseEmbeddingsVectors(self, value):
        return self._set(parseEmbeddingsVectors=value)


class EmbeddingsFinisher(AnnotatorTransformer):

    inputCols = Param(Params._dummy(), "inputCols", "name of input annotation cols containing embeddings", typeConverter=TypeConverters.toListString)
    outputCols = Param(Params._dummy(), "outputCols", "output EmbeddingsFinisher ouput cols", typeConverter=TypeConverters.toListString)
    cleanAnnotations = Param(Params._dummy(), "cleanAnnotations", "whether to remove all the existing annotation columns", typeConverter=TypeConverters.toBoolean)
    outputAsVector = Param(Params._dummy(), "outputAsVector", "if enabled it will output the embeddings as Vectors instead of arrays", typeConverter=TypeConverters.toBoolean)

    name = "EmbeddingsFinisher"

    @keyword_only
    def __init__(self):
        super(EmbeddingsFinisher, self).__init__(classname="com.johnsnowlabs.nlp.EmbeddingsFinisher")
        self._setDefault(
            cleanAnnotations=False,
            outputAsVector=False
        )

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCols(self, *value):
        if len(value) == 1 and type(value[0]) == list:
            return self._set(inputCols=value[0])
        else:
            return self._set(inputCols=list(value))

    def setOutputCols(self, *value):
        if len(value) == 1 and type(value[0]) == list:
            return self._set(outputCols=value[0])
        else:
            return self._set(outputCols=list(value))

    def setCleanAnnotations(self, value):
        return self._set(cleanAnnotations=value)

    def setOutputAsVector(self, value):
        return self._set(outputAsVector=value)
