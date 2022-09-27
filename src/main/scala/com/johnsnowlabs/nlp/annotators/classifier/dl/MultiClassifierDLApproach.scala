/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.AnnotatorType.{CATEGORY, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{AnnotatorApproach, ParamsAndFeaturesWritable}
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.explode
import org.apache.spark.sql.types.{ArrayType, StringType}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.util.Random

/** Trains a MultiClassifierDL for Multi-label Text Classification.
  *
  * MultiClassifierDL uses a Bidirectional GRU with a convolutional model that we have built
  * inside TensorFlow and supports up to 100 classes.
  *
  * For instantiated/pretrained models, see [[MultiClassifierDLModel]].
  *
  * The input to `MultiClassifierDL` are Sentence Embeddings such as the state-of-the-art
  * [[com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder UniversalSentenceEncoder]],
  * [[com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings BertSentenceEmbeddings]], or
  * [[com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings SentenceEmbeddings]].
  *
  * In machine learning, multi-label classification and the strongly related problem of
  * multi-output classification are variants of the classification problem where multiple labels
  * may be assigned to each instance. Multi-label classification is a generalization of multiclass
  * classification, which is the single-label problem of categorizing instances into precisely one
  * of more than two classes; in the multi-label problem there is no constraint on how many of the
  * classes the instance can be assigned to. Formally, multi-label classification is the problem
  * of finding a model that maps inputs x to binary vectors y (assigning a value of 0 or 1 for
  * each element (label) in y).
  *
  * '''Notes''':
  *   - This annotator requires an array of labels in type of String.
  *   - [[com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder UniversalSentenceEncoder]],
  *     [[com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings BertSentenceEmbeddings]], or
  *     [[com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings SentenceEmbeddings]] can be used for
  *     the `inputCol`.
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/classification/MultiClassifierDL_train_multi_label_E2E_challenge_classifier.ipynb Spark NLP Workshop]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/MultiClassifierDLTestSpec.scala MultiClassifierDLTestSpec]].
  *
  * ==Example==
  * In this example, the training data has the form (Note: labels can be arbitrary)
  * {{{
  * mr,ref
  * "name[Alimentum], area[city centre], familyFriendly[no], near[Burger King]",Alimentum is an adult establish found in the city centre area near Burger King.
  * "name[Alimentum], area[city centre], familyFriendly[yes]",Alimentum is a family-friendly place in the city centre.
  * ...
  * }}}
  * It needs some pre-processing first, so the labels are of type `Array[String]`. This can be
  * done like so:
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.annotators.classifier.dl.MultiClassifierDLApproach
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
  * import org.apache.spark.ml.Pipeline
  * import org.apache.spark.sql.functions.{col, udf}
  *
  * // Process training data to create text with associated array of labels
  * def splitAndTrim = udf { labels: String =>
  *   labels.split(", ").map(x=>x.trim)
  * }
  *
  * val smallCorpus = spark.read
  *   .option("header", true)
  *   .option("inferSchema", true)
  *   .option("mode", "DROPMALFORMED")
  *   .csv("src/test/resources/classifier/e2e.csv")
  *   .withColumn("labels", splitAndTrim(col("mr")))
  *   .withColumn("text", col("ref"))
  *   .drop("mr")
  *
  * smallCorpus.printSchema()
  * // root
  * // |-- ref: string (nullable = true)
  * // |-- labels: array (nullable = true)
  * // |    |-- element: string (containsNull = true)
  *
  * // Then create pipeline for training
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *   .setCleanupMode("shrink")
  *
  * val embeddings = UniversalSentenceEncoder.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("embeddings")
  *
  * val docClassifier = new MultiClassifierDLApproach()
  *   .setInputCols("embeddings")
  *   .setOutputCol("category")
  *   .setLabelColumn("labels")
  *   .setBatchSize(128)
  *   .setMaxEpochs(10)
  *   .setLr(1e-3f)
  *   .setThreshold(0.5f)
  *   .setValidationSplit(0.1f)
  *
  * val pipeline = new Pipeline()
  *   .setStages(
  *     Array(
  *       documentAssembler,
  *       embeddings,
  *       docClassifier
  *     )
  *   )
  *
  * val pipelineModel = pipeline.fit(smallCorpus)
  * }}}
  *
  * @see
  *   [[https://en.wikipedia.org/wiki/Multi-label_classification Multi-label classification on Wikipedia]]
  * @see
  *   [[ClassifierDLApproach]] for single-class classification
  * @see
  *   [[SentimentDLApproach]] for sentiment analysis
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class MultiClassifierDLApproach(override val uid: String)
    extends AnnotatorApproach[MultiClassifierDLModel]
    with ParamsAndFeaturesWritable
    with ClassifierEncoder {

  def this() = this(Identifiable.randomUID("MultiClassifierDLApproach"))

  /** Trains TensorFlow model for multi-class text classification */
  override val description = "Trains TensorFlow model for multi-class text classification"

  /** Input annotator type : SENTENCE_EMBEDDINGS
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(SENTENCE_EMBEDDINGS)

  /** Output annotator type : CATEGORY
    *
    * @group anno
    */
  override val outputAnnotatorType: String = CATEGORY

  /** The minimum threshold for each label to be accepted (Default: `0.5f`)
    *
    * @group param
    */
  val threshold = new FloatParam(
    this,
    "threshold",
    "The minimum threshold for each label to be accepted. Default is 0.5")

  /** Whether to shuffle the training data on each Epoch (Default: `false`)
    *
    * @group param
    */
  val shufflePerEpoch = new BooleanParam(
    this,
    "shufflePerEpoch",
    "whether to shuffle the training data on each Epoch")

  /** The minimum threshold for each label to be accepted (Default: `0.5f`)
    *
    * @group setParam
    */
  def setThreshold(threshold: Float): MultiClassifierDLApproach.this.type =
    set(this.threshold, threshold)

  /** shufflePerEpoch
    *
    * @group setParam
    */
  def setShufflePerEpoch(value: Boolean): MultiClassifierDLApproach.this.type =
    set(this.shufflePerEpoch, value)

  /** The minimum threshold for each label to be accepted (Default: `0.5f`)
    *
    * @group getParam
    */
  def getThreshold: Float = $(this.threshold)

  /** Max sequence length to feed into TensorFlow
    *
    * @group getParam
    */
  def getShufflePerEpoch: Boolean = $(shufflePerEpoch)

  setDefault(
    maxEpochs -> 10,
    lr -> 1e-3f,
    batchSize -> 64,
    threshold -> 0.5f,
    randomSeed -> 44,
    shufflePerEpoch -> false)

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): MultiClassifierDLModel = {

    val labelColType = dataset.schema($(labelColumn)).dataType
    require(
      labelColType == ArrayType(StringType),
      s"The label column $labelColumn type is $labelColType and it's not compatible. Compatible types are ArrayType(StringType).")

    val (trainDataset, trainLabels) = buildDatasetWithLabels(dataset, getInputCols(0))
    val settings = ClassifierDatasetEncoderParams(tags = trainLabels)
    val encoder = new ClassifierDatasetEncoder(settings)
    val trainInputs = extractInputsMultilabel(encoder, trainDataset)

    var testEncoder: Option[ClassifierDatasetEncoder] = None
    val testInputs =
      if (!isDefined(testDataset)) None
      else {
        val testDataFrame = ResourceHelper.readSparkDataFrame($(testDataset))
        val (test, testLabels) = buildDatasetWithLabels(testDataFrame, getInputCols(0))

        val settings = ClassifierDatasetEncoderParams(tags = testLabels)
        testEncoder = Some(new ClassifierDatasetEncoder(settings))
        Option(extractInputsMultilabel(testEncoder.get, test))
      }

    val tfWrapper: TensorflowWrapper = loadSavedModel()

    val classifier =
      try {
        val model =
          new TensorflowMultiClassifier(
            tensorflow = tfWrapper,
            encoder,
            testEncoder,
            Verbose($(verbose)))
        if (isDefined(randomSeed)) {
          Random.setSeed($(randomSeed))
        }

        model.train(
          trainInputs,
          testInputs,
          trainLabels.length,
          lr = $(lr),
          batchSize = $(batchSize),
          endEpoch = $(maxEpochs),
          configProtoBytes = getConfigProtoBytes,
          validationSplit = $(validationSplit),
          evaluationLogExtended = $(evaluationLogExtended),
          enableOutputLogs = $(enableOutputLogs),
          outputLogsPath = $(outputLogsPath),
          shuffleEpoch = $(shufflePerEpoch),
          uuid = this.uid)
        model
      } catch {
        case e: Exception =>
          throw e
      }

    val newWrapper = new TensorflowWrapper(
      TensorflowWrapper.extractVariablesSavedModel(
        tfWrapper.getTFSession(configProtoBytes = getConfigProtoBytes)),
      tfWrapper.graph)

    val embeddingsRef =
      HasStorageRef.getStorageRefFromInput(dataset, $(inputCols), SENTENCE_EMBEDDINGS)

    val model = new MultiClassifierDLModel()
      .setDatasetParams(classifier.encoder.params)
      .setModelIfNotSet(dataset.sparkSession, newWrapper)
      .setStorageRef(embeddingsRef)
      .setThreshold($(threshold))

    if (get(configProtoBytes).isDefined)
      model.setConfigProtoBytes($(configProtoBytes))

    model
  }

  override protected def buildDatasetWithLabels(
      dataset: Dataset[_],
      inputCols: String): (DataFrame, Array[String]) = {

    val embeddingsField: String = ".embeddings"
    val inputColumns = inputCols + embeddingsField

    val datasetWithLabels = dataset.select(dataset.col($(labelColumn)), dataset.col(inputColumns))
    val labels = datasetWithLabels
      .select(explode(dataset.col($(labelColumn))))
      .distinct
      .collect
      .map(x => x(0).toString)

    require(
      labels.length >= 2 && labels.length <= 100,
      s"The total unique number of classes must be more than 2 and less than 100. Currently is ${labels.length}")

    (datasetWithLabels, labels)
  }

  private def extractInputsMultilabel(
      encoder: ClassifierDatasetEncoder,
      dataset: DataFrame): (Array[Array[Array[Float]]], Array[Array[String]]) = {

    val embeddingsDim = encoder.calculateEmbeddingsDim(dataset)
    require(
      embeddingsDim > 1 && embeddingsDim <= 1024,
      s"The MultiClassifierDL only accepts embeddings larger than 1 and less than 1024 dimensions. Current dimension is ${embeddingsDim}. Please use embeddings" +
        s" with at max 1024 dimensions")

    val trainSet = encoder.collectTrainingInstancesMultiLabel(dataset, getLabelColumn)
    val inputEmbeddings = encoder.extractSentenceEmbeddingsMultiLabel(trainSet)
    val inputLabels = encoder.extractLabelsMultiLabel(trainSet)

    (inputEmbeddings, inputLabels)
  }

  def loadSavedModel(): TensorflowWrapper = {

    val wrapper =
      TensorflowWrapper.readZippedSavedModel(
        "/multi-classifier-dl",
        fileName = s"multi-label-bilstm-1024",
        tags = Array("serve"),
        initAllTables = true)
    wrapper.variables = Variables(Array.empty[Array[Byte]], Array.empty[Byte])
    wrapper
  }

}
