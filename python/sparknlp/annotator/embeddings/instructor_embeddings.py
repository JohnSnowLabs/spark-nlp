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
"""Contains classes for BertEmbeddings."""

from sparknlp.common import *


class InstructorEmbeddings(AnnotatorModel,
                           HasEmbeddingsProperties,
                           HasCaseSensitiveProperties,
                           HasStorageRef,
                           HasBatchedAnnotate,
                           HasMaxSentenceLengthLimit):
    """Sentence embeddings using INSTRUCTOR.

    Instructor👨‍🏫, an instruction-finetuned text embedding model that can generate text embeddings tailored to any task (e.g., classification, retrieval, clustering, text evaluation, etc.) and domains (e.g., science, finance, etc.) by simply providing the task instruction, without any finetuning. Instructor👨‍ achieves sota on 70 diverse embedding tasks!

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> embeddings = InstructorEmbeddings.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setInstruction("Represent the Medicine sentence for clustering: ") \\
    ...     .setOutputCol("instructor_embeddings")


    The default model is ``"instructor_base"``, if no name is provided.

    For available pretrained models please see the
    `Models Hub <https://sparknlp.org/models?q=Instructor>`__.


    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``            ``SENTENCE_EMBEDDINGS``
    ====================== ======================

    Parameters
    ----------
    batchSize
        Size of every batch , by default 8
    dimension
        Number of embedding dimensions, by default 768
    caseSensitive
        Whether to ignore case in tokens for embeddings matching, by default False
    instruction
        Set transformer instruction, e.g. 'summarize:'
    maxSentenceLength
        Max sentence length to process, by default 128
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.

    References
    ----------
    `One Embedder, Any Task: Instruction-Finetuned Text Embeddings <https://arxiv.org/abs/2212.09741>`__

    https://github.com/HKUNLP/instructor-embedding/

    **Paper abstract**

    *We introduce INSTRUCTOR, a new method for computing text embeddings given task instructions:
    every text input is embedded together with instructions explaining the use case (e.g., task and
    domain descriptions). Unlike encoders from prior work that are more specialized, INSTRUCTOR is a
    single embedder that can generate text embeddings tailored to different downstream tasks and domains,
    without any further training. We first annotate instructions for 330 diverse tasks and train INSTRUCTOR
    on this multitask mixture with a contrastive loss. We evaluate INSTRUCTOR on 70 embedding evaluation tasks
    (66 of which are unseen during training), ranging from classification and information retrieval to semantic
    textual similarity and text generation evaluation. INSTRUCTOR, while having an order of magnitude fewer
    parameters than the previous best model, achieves state-of-the-art performance, with an average improvement
    of 3.4% compared to the previous best results on the 70 diverse datasets. Our analysis suggests that
    INSTRUCTOR is robust to changes in instructions, and that instruction finetuning mitigates the challenge of
    training a single model on diverse datasets. Our model, code, and data are available at this https
    URL <https://instructor-embedding.github.io/>.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("document")
    >>> embeddings = InstructorEmbeddings.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setInstruction("Represent the Medicine sentence for clustering: ") \\
    ...     .setOutputCol("instructor_embeddings")
    >>> embeddingsFinisher = EmbeddingsFinisher() \\
    ...     .setInputCols(["instructor_embeddings"]) \\
    ...     .setOutputCols("finished_embeddings") \\
    ...     .setOutputAsVector(True)
    >>> pipeline = Pipeline().setStages([
    ...     documentAssembler,
    ...     embeddings,
    ...     embeddingsFinisher
    ... ])
    >>> data = spark.createDataFrame([["Dynamical Scalar Degree of Freedom in Horava-Lifshitz Gravity"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
    +--------------------------------------------------------------------------------+
    |                                                                          result|
    +--------------------------------------------------------------------------------+
    |[-2.3497989177703857,0.480538547039032,-0.3238905668258667,-1.612930893898010...|
    +--------------------------------------------------------------------------------+
    """

    name = "InstructorEmbeddings"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS
    instruction = Param(Params._dummy(), "instruction", "Set transformer instruction, e.g. 'summarize:'",
                        typeConverter=TypeConverters.toString)
    configProtoBytes = Param(Params._dummy(),
                             "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    def setInstruction(self, value):
        """ Sets transformer instruction, e.g. 'summarize:'.

        Parameters
        ----------
        value : str
        """
        return self._set(instruction=value)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.embeddings.InstructorEmbeddings", java_model=None):
        super(InstructorEmbeddings, self).__init__(
            classname=classname,
            java_model=java_model
        )
        self._setDefault(
            dimension=768,
            batchSize=8,
            maxSentenceLength=128,
            caseSensitive=False,
            instruction="",
        )

    @staticmethod
    def loadSavedModel(folder, spark_session):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        InstructorEmbeddings
            The restored model
        """
        from sparknlp.internal import _InstructorLoader
        jModel = _InstructorLoader(folder, spark_session._jsparkSession)._java_obj
        return InstructorEmbeddings(java_model=jModel)

    @staticmethod
    def pretrained(name="instructor_base", lang="en", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "instructor_base"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        InstructorEmbeddings
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(InstructorEmbeddings, name, lang, remote_loc)
