/*
 * Copyright 2017-2021 John Snow Labs
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
import com.johnsnowlabs.ml.tensorflow.wrap.Variables
import com.johnsnowlabs.nlp.{AnnotatorApproach, AnnotatorType, ParamsAndFeaturesWritable}
import com.johnsnowlabs.nlp.AnnotatorType.{CATEGORY, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, FloatParam, IntArrayParam, IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.types.{DoubleType, FloatType, IntegerType, StringType}
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.util.Random

/** Trains a SentimentDL, an annotator for multi-class sentiment analysis.
 *
 * In natural language processing, sentiment analysis is the task of classifying the affective state or subjective view
 * of a text. A common example is if either a product review or tweet can be interpreted positively or negatively.
 *
 * For the instantiated/pretrained models, see [[SentimentDLModel]].
 *
 * '''Notes''':
 *   - This annotator accepts a label column of a single item in either type of String, Int, Float, or Double.
 *     So positive sentiment can be expressed as either `"positive"` or `0`, negative sentiment as `"negative"` or `1`.
 *   - [[com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder UniversalSentenceEncoder]],
 *     [[com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings BertSentenceEmbeddings]], or
 *     [[com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings SentenceEmbeddings]] can be used for the `inputCol`.
 *
 * For extended examples of usage, see the [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/classification/SentimentDL_train_multiclass_sentiment_classifier.ipynb Spark NLP Workshop]]
 * and the [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/SentimentDLTestSpec.scala SentimentDLTestSpec]].
 *
 * ==Example==
 * In this example, `sentiment.csv` is in the form
 * {{{
 * text,label
 * This movie is the best movie I have watched ever! In my opinion this movie can win an award.,0
 * This was a terrible movie! The acting was bad really bad!,1
 * }}}
 * The model can then be trained with
 * {{{
 * import com.johnsnowlabs.nlp.base.DocumentAssembler
 * import com.johnsnowlabs.nlp.annotator.UniversalSentenceEncoder
 * import com.johnsnowlabs.nlp.annotators.classifier.dl.{SentimentDLApproach, SentimentDLModel}
 * import org.apache.spark.ml.Pipeline
 *
 * val smallCorpus = spark.read.option("header", "true").csv("src/test/resources/classifier/sentiment.csv")
 *
 * val documentAssembler = new DocumentAssembler()
 *   .setInputCol("text")
 *   .setOutputCol("document")
 *
 * val useEmbeddings = UniversalSentenceEncoder.pretrained()
 *   .setInputCols("document")
 *   .setOutputCol("sentence_embeddings")
 *
 * val docClassifier = new SentimentDLApproach()
 *   .setInputCols("sentence_embeddings")
 *   .setOutputCol("sentiment")
 *   .setLabelColumn("label")
 *   .setBatchSize(32)
 *   .setMaxEpochs(1)
 *   .setLr(5e-3f)
 *   .setDropout(0.5f)
 *
 * val pipeline = new Pipeline()
 *   .setStages(
 *     Array(
 *       documentAssembler,
 *       useEmbeddings,
 *       docClassifier
 *     )
 *   )
 *
 * val pipelineModel = pipeline.fit(smallCorpus)
 * }}}
 *
 * @see [[ClassifierDLApproach]] for general single-class classification
 * @see [[MultiClassifierDLApproach]] for general multi-class classification
 * @param uid required uid for storing annotator to disk
 * @groupname anno Annotator types
 * @groupdesc anno Required input and expected output annotator types
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
 * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
 */
class SentimentDLApproach(override val uid: String)
  extends AnnotatorApproach[SentimentDLModel]
    with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("SentimentDL"))

  override val description = "Trains TensorFlow model for Sentiment Classification"
  /** Input Annotator Types: SENTENCE_EMBEDDINGS
   *
   * @group anno
   */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(SENTENCE_EMBEDDINGS)
  /** Output Annotator Types: CATEGORY
   *
   * @group anno
   */
  override val outputAnnotatorType: String = CATEGORY


  /** Random seed for shuffling the dataset
   *
   * @group param
   */
  val randomSeed = new IntParam(this, "randomSeed", "Random seed")


  /** Column with label per each document
   *
   * @group param
   */
  val labelColumn = new Param[String](this, "labelColumn", "Column with label per each document")

  /** Learning Rate (Default: `5e-3f`)
   *
   * @group param
   */
  val lr = new FloatParam(this, "lr", "Learning Rate")

  /** Batch size (Default: `64`)
   *
   * @group param
   */
  val batchSize = new IntParam(this, "batchSize", "Batch size")

  /** Dropout coefficient (Default: `0.5f`)
   *
   * @group param
   */
  val dropout = new FloatParam(this, "dropout", "Dropout coefficient")

  /** The minimum threshold for the final result otherwise it will be either neutral or the value set in thresholdLabel (Default: `0.6f`)
   *
   * @group param
   */
  val threshold = new FloatParam(this, "threshold", "The minimum threshold for the final result otherwise it will be either neutral or the value set in thresholdLabel.")

  /** In case the score is less than threshold, what should be the label (Default: `"neutral"`)
   *
   * @group param
   */
  val thresholdLabel = new Param[String](this, "thresholdLabel", "In case the score is less than threshold, what should be the label. Default is neutral.")

  /** Maximum number of epochs to train (Default: `10`)
   *
   * @group param
   */
  val maxEpochs = new IntParam(this, "maxEpochs", "Maximum number of epochs to train")

  /** Whether to output to annotators log folder (Default: `false`)
   *
   * @group param
   */
  val enableOutputLogs = new BooleanParam(this, "enableOutputLogs", "Whether to output to annotators log folder")

  /** Folder path to save training logs (Default: `""`)
   *
   * @group param
   */
  val outputLogsPath = new Param[String](this, "outputLogsPath", "Folder path to save training logs")

  /** Choose the proportion of training dataset to be validated against the model on each Epoch (Default: `0.0f`).
   * The value should be between 0.0 and 1.0 and by default it is 0.0 and off.
   *
   * @group param
   */
  val validationSplit = new FloatParam(this, "validationSplit", "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.")

  /** Level of verbosity during training (Default: `Verbose.Silent.id`)
   *
   * @group param
   */
  val verbose = new IntParam(this, "verbose", "Level of verbosity during training")

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
   *
   * @group param
   * */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()"
  )

  /** @group setParam */
  def setLabelColumn(column: String): SentimentDLApproach.this.type = set(labelColumn, column)

  /** @group setParam */
  def setLr(lr: Float): SentimentDLApproach.this.type = set(this.lr, lr)

  /** @group setParam */
  def setBatchSize(batch: Int): SentimentDLApproach.this.type = set(this.batchSize, batch)

  /** @group setParam */
  def setDropout(dropout: Float): SentimentDLApproach.this.type = set(this.dropout, dropout)

  /** @group setParam */
  def setThreshold(threshold: Float): SentimentDLApproach.this.type = set(this.threshold, threshold)

  /** @group setParam */
  def setThresholdLabel(label: String): SentimentDLApproach.this.type = set(this.thresholdLabel, label)

  /** @group setParam */
  def setMaxEpochs(epochs: Int): SentimentDLApproach.this.type = set(maxEpochs, epochs)

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): SentimentDLApproach.this.type = set(this.configProtoBytes, bytes)

  /** @group setParam */
  def setEnableOutputLogs(enableOutputLogs: Boolean): SentimentDLApproach.this.type = set(this.enableOutputLogs, enableOutputLogs)

  /** @group setParam */
  def setOutputLogsPath(path: String): SentimentDLApproach.this.type = set(this.outputLogsPath, path)

  /** @group setParam */
  def setValidationSplit(validationSplit: Float): SentimentDLApproach.this.type = set(this.validationSplit, validationSplit)

  /** @group setParam */
  def setVerbose(verbose: Int): SentimentDLApproach.this.type = set(this.verbose, verbose)

  /** @group setParam */
  def setVerbose(verbose: Verbose.Level): SentimentDLApproach.this.type = set(this.verbose, verbose.id)

  /** Random seed
   *
   * @group setParam
   */
  def setRandomSeed(seed: Int): SentimentDLApproach.this.type = set(randomSeed, seed)

  /** @group getParam */
  def getLabelColumn: String = $(this.labelColumn)

  /** @group getParam */
  def getLr: Float = $(this.lr)

  /** @group getParam */
  def getBatchSize: Int = $(this.batchSize)

  /** @group getParam */
  def getDropout: Float = $(this.dropout)

  /** @group getParam */
  def getThreshold: Float = $(this.threshold)

  /** @group getParam */
  def getThresholdLabel: String = $(this.thresholdLabel)

  /** @group getParam */
  def getEnableOutputLogs: Boolean = $(enableOutputLogs)

  /** @group getParam */
  def getOutputLogsPath: String = $(outputLogsPath)

  /** @group getParam */
  def getValidationSplit: Float = $(this.validationSplit)

  /** @group getParam */
  def getMaxEpochs: Int = $(maxEpochs)

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  setDefault(
    maxEpochs -> 10,
    lr -> 5e-3f,
    dropout -> 0.5f,
    batchSize -> 64,
    enableOutputLogs -> false,
    verbose -> Verbose.Silent.id,
    validationSplit -> 0.0f,
    outputLogsPath -> "",
    threshold -> 0.6f,
    thresholdLabel -> "neutral"
  )

  override def beforeTraining(spark: SparkSession): Unit = {}

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): SentimentDLModel = {

    val labelColType = dataset.schema($(labelColumn)).dataType
    require(
      labelColType == StringType | labelColType == IntegerType | labelColType == DoubleType | labelColType == FloatType,
      s"The label column $labelColumn type is $labelColType and it's not compatible. " +
        s"Compatible types are StringType, IntegerType, DoubleType, or FloatType. "
    )

    val embeddingsRef = HasStorageRef.getStorageRefFromInput(dataset, $(inputCols), AnnotatorType.SENTENCE_EMBEDDINGS)

    val embeddingsField: String = ".embeddings"
    val inputColumns = getInputCols(0) + embeddingsField
    val train = dataset.select(dataset.col($(labelColumn)).cast("string"), dataset.col(inputColumns))
    val labels = train.select($(labelColumn)).distinct.collect.map(x => x(0).toString)

    require(
      labels.length == 2,
      s"The total unique number of classes must be 2. Currently is ${labels.length}. Please use ClassifierDLApproach" +
        s" if you have more than 2 classes/labels."
    )

    val tf = loadSavedModel()

    val settings = ClassifierDatasetEncoderParams(
      tags = labels
    )

    val encoder = new ClassifierDatasetEncoder(
      settings
    )

    val embeddingsDim = encoder.calculateEmbeddingsDim(train)
    require(
      embeddingsDim <= 1024,
      s"The SentimentDL only accepts embeddings less than 1024 dimensions. Current dimension is ${embeddingsDim}. Please use embeddings" +
        s" with less than "
    )
    val trainDataset = encoder.collectTrainingInstances(train, getLabelColumn)
    val inputEmbeddings = encoder.extractSentenceEmbeddings(trainDataset)
    val inputLabels = encoder.extractLabels(trainDataset)

    val classifier = try {
      val model = new TensorflowSentiment(
        tensorflow = tf,
        encoder,
        Verbose($(verbose))
      )
      if (isDefined(randomSeed)) {
        Random.setSeed($(randomSeed))
      }

      model.train(
        inputEmbeddings,
        inputLabels,
        lr = $(lr),
        batchSize = $(batchSize),
        dropout = $(dropout),
        endEpoch = $(maxEpochs),
        configProtoBytes = getConfigProtoBytes,
        validationSplit = $(validationSplit),
        enableOutputLogs = $(enableOutputLogs),
        outputLogsPath = $(outputLogsPath),
        uuid = this.uid
      )
      model
    } catch {
      case e: Exception =>
        throw e
    }

    val newWrapper = new TensorflowWrapper(TensorflowWrapper.extractVariablesSavedModel(tf.getSession(configProtoBytes = getConfigProtoBytes)), tf.graph)

    val model = new SentimentDLModel()
      .setDatasetParams(classifier.encoder.params)
      .setModelIfNotSet(dataset.sparkSession, newWrapper)
      .setStorageRef(embeddingsRef)
      .setThreshold($(threshold))
      .setThresholdLabel($(thresholdLabel))

    if (get(configProtoBytes).isDefined)
      model.setConfigProtoBytes($(configProtoBytes))

    model
  }

  def loadSavedModel(): TensorflowWrapper = {

    val wrapper =
      TensorflowWrapper.readZippedSavedModel("/sentiment-dl", tags = Array("serve"), initAllTables = true)
    wrapper.variables = Variables(Array.empty[Array[Byte]], Array.empty[Byte])
    wrapper
  }
}

/**
 * This is the companion object of [[SentimentApproach]]. Please refer to that class for the documentation.
 */
object SentimentApproach extends DefaultParamsReadable[SentimentDLApproach]
