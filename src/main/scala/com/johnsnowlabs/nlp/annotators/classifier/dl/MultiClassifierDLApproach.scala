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
import com.johnsnowlabs.nlp.AnnotatorType.{CATEGORY, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.{AnnotatorApproach, ParamsAndFeaturesWritable}
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types.{ArrayType, StringType}
import org.apache.spark.sql.{Dataset, SparkSession}
import org.apache.spark.sql.functions.explode

import scala.util.Random

/**
 * Trains a MultiClassifierDL for Multi-label Text Classification.
 *
 * MultiClassifierDL uses a Bidirectional GRU with a convolutional model that we have built inside TensorFlow and supports
 * up to 100 classes.
 *
 * For instantiated/pretrained models, see [[MultiClassifierDLModel]].
 *
 * The input to `MultiClassifierDL` are Sentence Embeddings such as the state-of-the-art
 * [[com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder UniversalSentenceEncoder]],
 * [[com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings BertSentenceEmbeddings]], or
 * [[com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings SentenceEmbeddings]].
 *
 * In machine learning, multi-label classification and the strongly related problem of multi-output classification are
 * variants of the classification problem where multiple labels may be assigned to each instance. Multi-label
 * classification is a generalization of multiclass classification, which is the single-label problem of categorizing
 * instances into precisely one of more than two classes; in the multi-label problem there is no constraint on how many
 * of the classes the instance can be assigned to.
 * Formally, multi-label classification is the problem of finding a model that maps inputs x to binary vectors y
 * (assigning a value of 0 or 1 for each element (label) in y).
 *
 * '''Notes''':
 *   - This annotator requires an array of labels in type of String.
 *   - [[com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder UniversalSentenceEncoder]],
 *     [[com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings BertSentenceEmbeddings]], or
 *     [[com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings SentenceEmbeddings]] can be used for the `inputCol`.
 *
 * For extended examples of usage, see the [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/training/english/classification/MultiClassifierDL_train_multi_label_E2E_challenge_classifier.ipynb Spark NLP Workshop]]
 * and the [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/MultiClassifierDLTestSpec.scala MultiClassifierDLTestSpec]].
 *
 * ==Example==
 * In this example, the training data has the form (Note: labels can be arbitrary)
 * {{{
 * mr,ref
 * "name[Alimentum], area[city centre], familyFriendly[no], near[Burger King]",Alimentum is an adult establish found in the city centre area near Burger King.
 * "name[Alimentum], area[city centre], familyFriendly[yes]",Alimentum is a family-friendly place in the city centre.
 * ...
 * }}}
 * It needs some pre-processing first, so the labels are of type `Array[String]`. This can be done like so:
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
 * @see [[https://en.wikipedia.org/wiki/Multi-label_classification Multi-label classification on Wikipedia]]
 * @see [[ClassifierDLApproach]] for single-class classification
 * @see [[SentimentDLApproach]] for sentiment analysis
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
 * */
class MultiClassifierDLApproach(override val uid: String)
  extends AnnotatorApproach[MultiClassifierDLModel]
    with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("MultiClassifierDLApproach"))

  /** Trains TensorFlow model for multi-class text classification */
  override val description = "Trains TensorFlow model for multi-class text classification"
  /** Input annotator type : SENTENCE_EMBEDDINGS
   *
   * @group anno
   * */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(SENTENCE_EMBEDDINGS)

  /** Output annotator type : CATEGORY
   *
   * @group anno
   * */
  override val outputAnnotatorType: String = CATEGORY

  /** Random seed for shuffling the dataset (Default: `44`)
   *
   * @group param
   * */
  val randomSeed = new IntParam(this, "randomSeed", "Random seed")

  /** Column with label per each document
   *
   * @group param
   * */
  val labelColumn = new Param[String](this, "labelColumn", "Column with label per each document")

  /** Learning Rate (Default: `1e-3f`)
   *
   * @group param
   * */
  val lr = new FloatParam(this, "lr", "Learning Rate")

  /** Batch size (Default: `64`)
   *
   * @group param
   * */
  val batchSize = new IntParam(this, "batchSize", "Batch size")

  /** Maximum number of epochs to train (Default: `10`)
   *
   * @group param
   * */
  val maxEpochs = new IntParam(this, "maxEpochs", "Maximum number of epochs to train")

  /** Whether to output to annotators log folder (Default: `false`)
   *
   * @group param
   * */
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
   * */
  val validationSplit = new FloatParam(this, "validationSplit", "Choose the proportion of training dataset to be validated against the model on each Epoch. The value should be between 0.0 and 1.0 and by default it is 0.0 and off.")

  /** Level of verbosity during training (Default: `Verbose.Silent.id`)
   *
   * @group param
   * */
  val verbose = new IntParam(this, "verbose", "Level of verbosity during training")

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
   *
   * @group param
   * */

  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** The minimum threshold for each label to be accepted (Default: `0.5f`)
   *
   * @group param
   * */
  val threshold = new FloatParam(this, "threshold", "The minimum threshold for each label to be accepted. Default is 0.5")

  /** Whether to shuffle the training data on each Epoch (Default: `false`)
   *
   * @group param
   * */
  val shufflePerEpoch = new BooleanParam(this, "shufflePerEpoch", "whether to shuffle the training data on each Epoch")

  /** Random seed
   *
   * @group setParam
   */
  def setRandomSeed(seed: Int): MultiClassifierDLApproach.this.type = set(randomSeed, seed)

  /** Column with label per each document
   *
   * @group setParam
   * */
  def setLabelColumn(column: String): MultiClassifierDLApproach.this.type = set(labelColumn, column)

  /** Learning Rate (Default: `1e-3f`)
   *
   * @group setParam
   * */
  def setLr(lr: Float): MultiClassifierDLApproach.this.type = set(this.lr, lr)

  /** Batch size (Default: `64`)
   *
   * @group setParam
   * */
  def setBatchSize(batch: Int): MultiClassifierDLApproach.this.type = set(this.batchSize, batch)

  /** Maximum number of epochs to train (Default: `10`)
   *
   * @group setParam
   * */
  def setMaxEpochs(epochs: Int): MultiClassifierDLApproach.this.type = set(maxEpochs, epochs)

  /** Tensorflow config Protobytes passed to the TF session
   *
   * @group setParam
   * */
  def setConfigProtoBytes(bytes: Array[Int]): MultiClassifierDLApproach.this.type = set(this.configProtoBytes, bytes)

  /** Whether to output to annotators log folder (Default: `false`)
   *
   * @group setParam
   * */
  def setEnableOutputLogs(enableOutputLogs: Boolean): MultiClassifierDLApproach.this.type = set(this.enableOutputLogs, enableOutputLogs)

  /** Choose the proportion of training dataset to be validated against the model on each Epoch (Default: `0.0f`).
   * The value should be between 0.0 and 1.0 and by default it is 0.0 and off.
   *
   * @group setParam
   * */
  def setValidationSplit(validationSplit: Float): MultiClassifierDLApproach.this.type = set(this.validationSplit, validationSplit)

  /** Level of verbosity during training (Default: `Verbose.Silent.id`)
   *
   * @group setParam
   * */
  def setVerbose(verbose: Int): MultiClassifierDLApproach.this.type = set(this.verbose, verbose)

  /** Folder path to save training logs (Default: `""`)
   *
   * @group setParam
   * */
  def setOutputLogsPath(path: String): MultiClassifierDLApproach.this.type = set(this.outputLogsPath, path)

  /** Level of verbosity during training (Default: `Verbose.Silent.id`)
   *
   * @group setParam
   * */
  def setVerbose(verbose: Verbose.Level): MultiClassifierDLApproach.this.type = set(this.verbose, verbose.id)

  /** The minimum threshold for each label to be accepted (Default: `0.5f`)
   *
   * @group setParam
   * */
  def setThreshold(threshold: Float): MultiClassifierDLApproach.this.type = set(this.threshold, threshold)

  /** shufflePerEpoch
   *
   * @group setParam
   * */
  def setShufflePerEpoch(value: Boolean): MultiClassifierDLApproach.this.type = set(this.shufflePerEpoch, value)

  /** Random seed
   *
   * @group getParam
   */
  def getRandomSeed: Int = $(this.randomSeed)

  /** Column with label per each document
   *
   * @group getParam
   * */
  def getLabelColumn: String = $(this.labelColumn)

  /** Learning Rate (Default: `1e-3f`)
   *
   * @group getParam
   * */
  def getLr: Float = $(this.lr)

  /** Batch size (Default: `64`)
   *
   * @group getParam
   * */
  def getBatchSize: Int = $(this.batchSize)

  /** Whether to output to annotators log folder (Default: `false`)
   *
   * @group getParam
   * */
  def getEnableOutputLogs: Boolean = $(enableOutputLogs)

  /** Choose the proportion of training dataset to be validated against the model on each Epoch (Default: `0.0f`).
   * The value should be between 0.0 and 1.0 and by default it is 0.0 and off.
   *
   * @group getParam
   * */
  def getValidationSplit: Float = $(this.validationSplit)

  /** Folder path to save training logs (Default: `""`)
   *
   * @group getParam
   */
  def getOutputLogsPath: String = $(outputLogsPath)

  /** Maximum number of epochs to train (Default: `10`)
   *
   * @group getParam
   * */
  def getMaxEpochs: Int = $(maxEpochs)

  /** Tensorflow config Protobytes passed to the TF session
   *
   * @group getParam
   * */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** The minimum threshold for each label to be accepted (Default: `0.5f`)
   *
   * @group getParam
   * */
  def getThreshold: Float = $(this.threshold)

  /** Max sequence length to feed into TensorFlow
   *
   * @group getParam
   * */
  def getShufflePerEpoch: Boolean = $(shufflePerEpoch)

  setDefault(
    maxEpochs -> 10,
    lr -> 1e-3f,
    batchSize -> 64,
    enableOutputLogs -> false,
    verbose -> Verbose.Silent.id,
    validationSplit -> 0.0f,
    outputLogsPath -> "",
    threshold -> 0.5f,
    randomSeed -> 44,
    shufflePerEpoch -> false
  )

  override def beforeTraining(spark: SparkSession): Unit = {}

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): MultiClassifierDLModel = {

    val labelColType = dataset.schema($(labelColumn)).dataType
    require(
      labelColType == ArrayType(StringType),
      s"The label column $labelColumn type is $labelColType and it's not compatible. Compatible types are ArrayType(StringType)."
    )

    val embeddingsRef = HasStorageRef.getStorageRefFromInput(dataset, $(inputCols), SENTENCE_EMBEDDINGS)

    val embeddingsField: String = ".embeddings"
    val inputColumns = getInputCols(0) + embeddingsField
    val labelColumnString = $(labelColumn)
    val train = dataset.select(dataset.col(labelColumnString), dataset.col(inputColumns))
    val labelsArray = train.select(explode(dataset.col(labelColumnString))).distinct.collect.map(x => x(0).toString)
    val labelsCount = labelsArray.length

    require(
      labelsCount >= 2 && labelsCount <= 100,
      s"The total unique number of classes must be more than 2 and less than 100. Currently is $labelsCount"
    )

    val settings = ClassifierDatasetEncoderParams(
      tags = labelsArray
    )

    val encoder = new ClassifierDatasetEncoder(
      settings
    )

    val embeddingsDim = encoder.calculateEmbeddingsDim(train)
    require(
      embeddingsDim > 1 && embeddingsDim <= 1024,
      s"The MultiClassifierDL only accepts embeddings larger than 1 and less than 1024 dimensions. Current dimension is ${embeddingsDim}. Please use embeddings" +
        s" with at max 1024 dimensions"
    )

    val trainDataset = encoder.collectTrainingInstancesMultiLabel(train, getLabelColumn)
    val inputEmbeddings = encoder.extractSentenceEmbeddingsMultiLabel(trainDataset)
    val inputLabels = encoder.extractLabelsMultiLabel(trainDataset)

    val tf = loadSavedModel()

    val classifier = try {
      val model = new TensorflowMultiClassifier(
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
        labelsCount,
        lr = $(lr),
        batchSize = $(batchSize),
        endEpoch = $(maxEpochs),
        configProtoBytes = getConfigProtoBytes,
        validationSplit = $(validationSplit),
        enableOutputLogs = $(enableOutputLogs),
        outputLogsPath = $(outputLogsPath),
        shuffleEpoch = $(shufflePerEpoch),
        uuid = this.uid
      )
      model
    } catch {
      case e: Exception =>
        throw e
    }

    val newWrapper = new TensorflowWrapper(TensorflowWrapper.extractVariablesSavedModel(tf.getSession(configProtoBytes = getConfigProtoBytes)), tf.graph)

    val model = new MultiClassifierDLModel()
      .setDatasetParams(classifier.encoder.params)
      .setModelIfNotSet(dataset.sparkSession, newWrapper)
      .setStorageRef(embeddingsRef)
      .setThreshold($(threshold))

    if (get(configProtoBytes).isDefined)
      model.setConfigProtoBytes($(configProtoBytes))

    model
  }

  def loadSavedModel(): TensorflowWrapper = {

    val wrapper =
      TensorflowWrapper.readZippedSavedModel("/multi-classifier-dl", fileName = s"multi-label-bilstm-1024", tags = Array("serve"), initAllTables = true)
    wrapper.variables = Variables(Array.empty[Array[Byte]], Array.empty[Byte])
    wrapper
  }
}


