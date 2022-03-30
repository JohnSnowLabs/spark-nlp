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

"""Contains Properties for the Annotator classes.
"""
import json
from abc import ABCMeta, abstractmethod

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param import Params
from pyspark.ml.param.shared import Param, TypeConverters, HasOutputCol, HasInputCols, HasInputCol
from pyspark.ml.util import JavaMLWritable
from pyspark.ml.wrapper import JavaModel, JavaEstimator
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, ArrayType, StructType

import sparknlp.internal as _internal
from sparknlp.annotation import Annotation


class AnnotatorProperties(Params):
    inputCols = Param(Params._dummy(),
                      "inputCols",
                      "previous annotations columns, if renamed",
                      typeConverter=TypeConverters.toListString)
    outputCol = Param(Params._dummy(),
                      "outputCol",
                      "output annotation column. can be left default.",
                      typeConverter=TypeConverters.toString)
    lazyAnnotator = Param(Params._dummy(),
                          "lazyAnnotator",
                          "Whether this AnnotatorModel acts as lazy in RecursivePipelines",
                          typeConverter=TypeConverters.toBoolean
                          )

    def setInputCols(self, *value):
        """Sets column names of input annotations.

        Parameters
        ----------
        *value : str
            Input columns for the annotator
        """
        if len(value) == 1 and type(value[0]) == list:
            return self._set(inputCols=value[0])
        else:
            return self._set(inputCols=list(value))

    def getInputCols(self):
        """Gets current column names of input annotations."""
        return self.getOrDefault(self.inputCols)

    def setOutputCol(self, value):
        """Sets output column name of annotations.

        Parameters
        ----------
        value : str
            Name of output column
        """
        return self._set(outputCol=value)

    def getOutputCol(self):
        """Gets output column name of annotations."""
        return self.getOrDefault(self.outputCol)

    def setLazyAnnotator(self, value):
        """Sets whether Annotator should be evaluated lazily in a
        RecursivePipeline.

        Parameters
        ----------
        value : bool
            Whether Annotator should be evaluated lazily in a
            RecursivePipeline
        """
        return self._set(lazyAnnotator=value)

    def getLazyAnnotator(self):
        """Gets whether Annotator should be evaluated lazily in a
        RecursivePipeline.
        """
        self.getOrDefault(self.lazyAnnotator)


class AnnotatorModel(JavaModel, _internal.AnnotatorJavaMLReadable, JavaMLWritable, AnnotatorProperties,
                     _internal.ParamsGettersSetters):

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    @keyword_only
    def __init__(self, classname, java_model=None):
        super(AnnotatorModel, self).__init__(java_model=java_model)
        if classname and not java_model:
            self.__class__._java_class_name = classname
            self._java_obj = self._new_java_obj(classname, self.uid)
        if java_model is not None:
            self._transfer_params_from_java()
        self._setDefault(lazyAnnotator=False)

    def apply(self):
        return self._java_obj


class AnnotateJava:

    def annotate(self, annotations, annotator):
        json_annotations = self.annotationsToJson(annotations)
        java_output_list = annotator.annotateJson(json_annotations)
        processed_annotations = self.javaOutputToAnnotation(java_output_list)

        return processed_annotations

    def annotationsToJson(self, annotations):

        json_annotations = []
        for annotation in annotations:
            json_annotation = json.dumps(annotation.__dict__)
            embeddings = annotation.embeddings
            json_annotations.append({json_annotation: embeddings})

        return json_annotations

    def javaOutputToAnnotation(self, java_output_list):
        annotations = []
        for java_output in java_output_list:
            for annotation_str, embeddings in java_output.items():
                annotation_dict = json.loads(annotation_str)
                annotation = Annotation(annotation_dict['annotatorType'], int(annotation_dict['begin']),
                                        int(annotation_dict['end']), annotation_dict['result'],
                                        annotation_dict['metadata'], list(embeddings))

                annotations.append(annotation)

        return annotations


class AnnotatorType(object):
    DOCUMENT = "document"
    TOKEN = "token"
    WORDPIECE = "wordpiece"
    WORD_EMBEDDINGS = "word_embeddings"
    SENTENCE_EMBEDDINGS = "sentence_embeddings"
    CATEGORY = "category"
    DATE = "date"
    ENTITY = "entity"
    SENTIMENT = "sentiment"
    POS = "pos"
    CHUNK = "chunk"
    NAMED_ENTITY = "named_entity"
    NEGEX = "negex"
    DEPENDENCY = "dependency"
    LABELED_DEPENDENCY = "labeled_dependency"
    LANGUAGE = "language"
    NODE = "node"
    DUMMY = "dummy"


class SparkNLPTransformer(Transformer, HasInputCols, HasOutputCol, AnnotatorProperties, metaclass=ABCMeta):

    def __init__(self, input_annotator_types=[AnnotatorType.DOCUMENT], output_annotator_type=AnnotatorType.DOCUMENT):
        super().__init__()
        self.input_annotator_types = input_annotator_types
        self.output_annotator_type = output_annotator_type

        if self.input_annotator_types is None or self.output_annotator_type is None:
            error = "Error creating SparkNLPTransformer inputAnnotatorTypes and outputAnnotatorType must be defined"
            raise Exception(error)

        if (type(self.input_annotator_types)) is not list:
            raise TypeError("inputAnnotatorTypes must be a 'list'")

    def _transform(self, dataset):
        processed_dataset = dataset

        for inputCol in self.getInputCols():
            data_type = dataset.schema[inputCol].dataType
            if isinstance(data_type, StringType):
                processed_dataset = dataset.withColumn(self.getOutputCol(),
                                                       self.castStringToAnnotation(col(inputCol)))
            if self.isAnnotationType(data_type):
                processed_dataset = self.castToAnnotation(processed_dataset, inputCol)

        return processed_dataset

    @staticmethod
    def isAnnotationType(data_type):

        if isinstance(data_type, ArrayType) and isinstance(data_type.elementType, StructType):
            fields = data_type.elementType.fields
            annotation_fields = Annotation.dataType().fields

            fields_structure = list(map(lambda field: (field.dataType, field.metadata, field.name), fields))
            annotation_structure = list(map(lambda field: (field.dataType, field.metadata, field.name),
                                            annotation_fields))

            return fields_structure[:-1] == annotation_structure[:-1]

        return False

    def castToAnnotation(self, processed_dataset, input_col):
        processed_dataset = processed_dataset.withColumn(self.getOutputCol(),
                                                         self.wrapColumnMetadata(self.annotateUDF(col(input_col))))
        return processed_dataset

    def wrapColumnMetadata(self, column):
        return column.alias(self.getOutputCol(), metadata={'annotatorType': self.output_annotator_type})

    @staticmethod
    @udf(returnType=Annotation.arrayType())
    def castStringToAnnotation(raw_text):
        annotation = Annotation(AnnotatorType.DOCUMENT, 0, len(raw_text) - 1, raw_text, {}, [])
        return [annotation]

    @staticmethod
    @abstractmethod
    def annotateUDF(annotations) -> [Annotation]:
        raise NotImplementedError()

    @abstractmethod
    def annotate(self, annotations) -> [Annotation]:
        raise NotImplementedError()


class HasEmbeddingsProperties(Params):
    dimension = Param(Params._dummy(),
                      "dimension",
                      "Number of embedding dimensions",
                      typeConverter=TypeConverters.toInt)

    def setDimension(self, value):
        """Sets embeddings dimension.

        Parameters
        ----------
        value : int
            Embeddings dimension
        """
        return self._set(dimension=value)

    def getDimension(self):
        """Gets embeddings dimension."""
        return self.getOrDefault(self.dimension)


class HasStorageRef:
    storageRef = Param(Params._dummy(), "storageRef",
                       "unique reference name for identification",
                       TypeConverters.toString)

    def setStorageRef(self, value):
        """Sets unique reference name for identification.

        Parameters
        ----------
        value : str
            Unique reference name for identification
        """
        return self._set(storageRef=value)

    def getStorageRef(self):
        """Gets unique reference name for identification.

        Returns
        -------
        str
            Unique reference name for identification
        """
        return self.getOrDefault("storageRef")


class HasBatchedAnnotate:
    batchSize = Param(Params._dummy(), "batchSize", "Size of every batch", TypeConverters.toInt)

    def setBatchSize(self, v):
        """Sets batch size.

        Parameters
        ----------
        v : int
            Batch size
        """
        return self._set(batchSize=v)

    def getBatchSize(self):
        """Gets current batch size.

        Returns
        -------
        int
            Current batch size
        """
        return self.getOrDefault("batchSize")


class HasCaseSensitiveProperties:
    caseSensitive = Param(Params._dummy(),
                          "caseSensitive",
                          "whether to ignore case in tokens for embeddings matching",
                          typeConverter=TypeConverters.toBoolean)

    def setCaseSensitive(self, value):
        """Sets whether to ignore case in tokens for embeddings matching.

        Parameters
        ----------
        value : bool
            Whether to ignore case in tokens for embeddings matching
        """
        return self._set(caseSensitive=value)

    def getCaseSensitive(self):
        """Gets whether to ignore case in tokens for embeddings matching.

        Returns
        -------
        bool
            Whether to ignore case in tokens for embeddings matching
        """
        return self.getOrDefault(self.caseSensitive)


class HasExcludableStorage:
    includeStorage = Param(Params._dummy(),
                           "includeStorage",
                           "whether to include indexed storage in trained model",
                           typeConverter=TypeConverters.toBoolean)

    def setIncludeStorage(self, value):
        """Sets whether to include indexed storage in trained model.

        Parameters
        ----------
        value : bool
            Whether to include indexed storage in trained model
        """
        return self._set(includeStorage=value)

    def getIncludeStorage(self):
        """Gets whether to include indexed storage in trained model.

        Returns
        -------
        bool
            Whether to include indexed storage in trained model
        """
        return self.getOrDefault("includeStorage")


class HasStorage(HasStorageRef, HasCaseSensitiveProperties, HasExcludableStorage):
    storagePath = Param(Params._dummy(),
                        "storagePath",
                        "path to file",
                        typeConverter=TypeConverters.identity)

    def setStoragePath(self, path, read_as):
        """Sets path to file.

        Parameters
        ----------
        path : str
            Path to file
        read_as : str
            How to interpret the file

        Notes
        -----
        See :class:`ReadAs <sparknlp.common.ReadAs>` for reading options.
        """
        return self._set(storagePath=ExternalResource(path, read_as, {}))

    def getStoragePath(self):
        """Gets path to file.

        Returns
        -------
        str
            path to file
        """
        return self.getOrDefault("storagePath")


class HasStorageModel(HasStorageRef, HasCaseSensitiveProperties, HasExcludableStorage):

    def saveStorage(self, path, spark):
        """Saves the current model to storage.

        Parameters
        ----------
        path : str
            Path for saving the model.
        spark : :class:`pyspark.sql.SparkSession`
            The current SparkSession
        """
        self._transfer_params_to_java()
        self._java_obj.saveStorage(path, spark._jsparkSession, False)

    @staticmethod
    def loadStorage(path, spark, storage_ref):
        raise NotImplementedError("AnnotatorModel with HasStorageModel did not implement 'loadStorage'")

    @staticmethod
    def loadStorages(path, spark, storage_ref, databases):
        for database in databases:
            _internal._StorageHelper(path, spark, database, storage_ref, within_storage=False)


class AnnotatorApproach(JavaEstimator, JavaMLWritable, _internal.AnnotatorJavaMLReadable, AnnotatorProperties,
                        _internal.ParamsGettersSetters, HasInputCol, HasInputCols, HasOutputCol):

    @keyword_only
    def __init__(self, classname):
        _internal.ParamsGettersSetters.__init__(self)
        self.__class__._java_class_name = classname
        self._java_obj = self._new_java_obj(classname, self.uid)
        self._setDefault(lazyAnnotator=False)

    def _create_model(self, java_model):
        raise NotImplementedError('Please implement _create_model in %s' % self)


class RecursiveAnnotatorApproach(_internal.RecursiveEstimator, JavaMLWritable, _internal.AnnotatorJavaMLReadable,
                                 AnnotatorProperties,
                                 _internal.ParamsGettersSetters):
    @keyword_only
    def __init__(self, classname):
        _internal.ParamsGettersSetters.__init__(self)
        self.__class__._java_class_name = classname
        self._java_obj = self._new_java_obj(classname, self.uid)
        self._setDefault(lazyAnnotator=False)

    def _create_model(self, java_model):
        raise NotImplementedError('Please implement _create_model in %s' % self)


def RegexRule(rule, identifier):
    return _internal._RegexRule(rule, identifier).apply()


class ReadAs(object):
    """Object that contains constants for how to read Spark Resources.

    Possible values are:

    ================= =======================================
    Value             Description
    ================= =======================================
    ``ReadAs.TEXT``   Read the resource as text.
    ``ReadAs.SPARK``  Read the resource as a Spark DataFrame.
    ``ReadAs.BINARY`` Read the resource as a binary file.
    ================= =======================================
    """
    TEXT = "TEXT"
    SPARK = "SPARK"
    BINARY = "BINARY"


def ExternalResource(path, read_as=ReadAs.TEXT, options={}):
    """Returns a representation fo an External Resource.

    How the resource is read can be set with `read_as`.

    Parameters
    ----------
    path : str
        Path to the resource
    read_as : str, optional
        How to read the resource, by default ReadAs.TEXT
    options : dict, optional
        Options to read the resource, by default {}
    """
    return _internal._ExternalResource(path, read_as, options).apply()


class CoverageResult:
    def __init__(self, cov_obj):
        self.covered = cov_obj.covered()
        self.total = cov_obj.total()
        self.percentage = cov_obj.percentage()


class HasEnableCachingProperties:
    enableCaching = Param(Params._dummy(),
                          "enableCaching",
                          "Whether to enable caching DataFrames or RDDs during the training",
                          typeConverter=TypeConverters.toBoolean)

    def setEnableCaching(self, value):
        """Sets whether to enable caching DataFrames or RDDs during the training

        Parameters
        ----------
        value : bool
            Whether to enable caching DataFrames or RDDs during the training
        """
        return self._set(enableCaching=value)

    def getEnableCaching(self):
        """Gets whether to enable caching DataFrames or RDDs during the training

        Returns
        -------
        bool
            Whether to enable caching DataFrames or RDDs during the training
        """
        return self.getOrDefault(self.enableCaching)
