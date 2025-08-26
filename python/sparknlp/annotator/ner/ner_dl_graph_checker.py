#  Copyright 2017-2025 John Snow Labs
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
"""Contains classes for NerDL."""

from sparknlp.common import *
import sparknlp.internal as _internal
from pyspark.ml.util import JavaMLWritable
from pyspark.ml.wrapper import JavaEstimator


class NerDLGraphChecker(
    JavaEstimator,
    JavaMLWritable,
    _internal.ParamsGettersSetters,
):
    """Checks whether a suitable NerDL graph is available for the given training dataset, before any
    computations/training is done. This annotator is useful for custom training cases, where
    specialized graphs are needed.

    Important: This annotator should be used or positioned before any embedding or NerDLApproach
    annotators in the pipeline and will process the whole dataset to extract the required graph parameters.

    This annotator requires a dataset with at least two columns: one with tokens and one with the
    labels. In addition, it requires the used embedding annotator in the pipeline to extract the
    suitable embedding dimension.

    For extended examples of usage, see the`Examples 
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master//home/ducha/Workspace/scala/spark-nlp-feature/examples/python/training/english/dl-ner/ner_dl_graph_checker.ipynb>`__ 
    and the `NerDLGraphCheckerTestSpec 
    <https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/ner/dl/NerDLGraphCheckerTestSpec.scala>`__.

    ==================================== ======================
    Input Annotation types               Output Annotation type
    ==================================== ======================
    ``DOCUMENT, TOKEN``                  `NONE`
    ==================================== ======================

    Parameters
    ----------
    inputCols
        Column names of input annotations
    labelColumn
        Column name for data labels
    embeddingsDim
        Dimensionality of embeddings

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline

    This CoNLL dataset already includes a sentence, token and label
    column with their respective annotator types. If a custom dataset is used,
    these need to be defined with for example:

    >>> conll = CoNLL()
    >>> trainingData = conll.readDataset(spark, "src/test/resources/conll2003/eng.train")
    >>> embeddings = BertEmbeddings \\
    ...     .pretrained() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setOutputCol("embeddings")
    
    This annotatorr requires the data for NerDLApproach graphs: text, tokens, labels and the embedding model

    >>> nerDLGraphChecker = NerDLGraphChecker() \\
    ...     .setInputCols(["sentence", "token"]) \\
    ...     .setLabelColumn("label") \\
    ...     .setEmbeddingsModel(embeddings)
    >>> nerTagger = NerDLApproach() \\
    ...     .setInputCols(["sentence", "token", "embeddings"]) \\
    ...     .setLabelColumn("label") \\
    ...     .setOutputCol("ner") \\
    ...     .setMaxEpochs(1) \\
    ...     .setRandomSeed(0) \\
    ...     .setVerbose(0)
    >>> pipeline = Pipeline().setStages([nerDLGraphChecker, embeddings, nerTagger])
    
    If we now fit the model with a graph missing, then an exception is raised.

    >>> pipelineModel = pipeline.fit(trainingData)
    """

    inputCols = Param(
        Params._dummy(),
        "inputCols",
        "Input columns",
        typeConverter=TypeConverters.toListString,
    )

    def setInputCols(self, *value):
        """Sets column names of input annotations.

        Parameters
        ----------
        *value : List[str]
            Input columns for the annotator
        """
        if type(value[0]) == str or type(value[0]) == list:
            # self.inputColsValidation(value)
            if len(value) == 1 and type(value[0]) == list:
                return self._set(inputCols=value[0])
            else:
                return self._set(inputCols=list(value))
        else:
            raise TypeError(
                "InputCols datatype not supported. It must be either str or list"
            )

    labelColumn = Param(
        Params._dummy(),
        "labelColumn",
        "Column with label per each token",
        typeConverter=TypeConverters.toString,
    )

    def setLabelColumn(self, value):
        """Sets name of column for data labels.

        Parameters
        ----------
        value : str
            Column for data labels
        """
        return self._set(labelColumn=value)

    embeddingsDim = Param(
        Params._dummy(),
        "embeddingsDim",
        "Dimensionality of embeddings",
        typeConverter=TypeConverters.toInt,
    )

    def setEmbeddingsDim(self, value: int):
        """Sets Dimensionality of embeddings

        Parameters
        ----------
        value : int
            Dimensionality of embeddings
        """
        return self._set(embeddingsDim=value)

    def setEmbeddingsModel(self, model: HasEmbeddingsProperties):
        """
        Get embeddingsDim from a given embeddings model, if possible.
        Falls back to setEmbeddingsDim if dimension cannot be obtained automatically.
        """
        # Try Python API first
        if hasattr(model, "getDimension"):
            dim = model.getDimension()
            return self.setEmbeddingsDim(int(dim))
        # Try JVM side if available
        if hasattr(model, "_java_obj") and hasattr(model._java_obj, "getDimension"):
            dim = int(model._java_obj.getDimension())
            return self.setEmbeddingsDim(dim)
        raise ValueError(
            "Could not infer embeddings dimension from provided model. "
            "Use setEmbeddingsDim(dim) explicitly."
        )

    inputAnnotatorTypes = [
        AnnotatorType.DOCUMENT,
        AnnotatorType.TOKEN,
    ]

    graphFolder = Param(
        Params._dummy(),
        "graphFolder",
        "Folder path that contain external graph files",
        TypeConverters.toString,
    )

    def setGraphFolder(self, p):
        """Sets folder path that contain external graph files.

        Parameters
        ----------
        p : str
            Folder path that contain external graph files
        """
        return self._set(graphFolder=p)

    @keyword_only
    def __init__(self):
        _internal.ParamsGettersSetters.__init__(self)
        classname = "com.johnsnowlabs.nlp.annotators.ner.dl.NerDLGraphChecker"
        self.__class__._java_class_name = classname
        self._java_obj = self._new_java_obj(classname, self.uid)
        # self._setDefault()

    def _create_model(self, java_model):
        return NerDLGraphCheckerModel()


class NerDLGraphCheckerModel(
    JavaModel,
    JavaMLWritable,
    _internal.ParamsGettersSetters,
):
    """
    Resulting model from NerDLGraphChecker, that does not perform any transformations, as the
    checks are done during the ``fit`` phase. It acts as the identity.

    This annotator should never be used directly.
    """

    inputAnnotatorTypes = [
        AnnotatorType.DOCUMENT,
        AnnotatorType.TOKEN,
    ]

    @keyword_only
    def __init__(
        self,
        classname="com.johnsnowlabs.nlp.annotators.ner.dl.NerDLGraphCheckerModel",
        java_model=None,
    ):
        super(NerDLGraphCheckerModel, self).__init__(java_model=java_model)
        if classname and not java_model:
            self.__class__._java_class_name = classname
            self._java_obj = self._new_java_obj(classname, self.uid)
        if java_model is not None:
            self._transfer_params_from_java()
        # self._setDefault(lazyAnnotator=False)
