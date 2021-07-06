package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.AnnotatorType.{CATEGORY, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.{AnnotatorApproach, AnnotatorType, ParamsAndFeaturesWritable}
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.types.{DoubleType, FloatType, IntegerType, StringType}
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.util.Random

/**
 * Trains a ClassifierDL for generic Multi-class Text Classification.
 *
 * ClassifierDL uses the state-of-the-art Universal Sentence Encoder as an input for text classifications.
 * The ClassifierDL annotator uses a deep learning model (DNNs) we have built inside TensorFlow and supports up to
 * 100 classes.
 *
 * For instantiated/pretrained models, see [[ClassifierDLModel]].
 *
 * '''Notes''':
 *   - This annotator accepts a label column of a single item in either type of String, Int, Float, or Double.
 *   - [[com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder UniversalSentenceEncoder]],
 *     [[com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings BertSentenceEmbeddings]], or
 *     [[com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings SentenceEmbeddings]] can be used for the `inputCol`.
 *
 * For extended examples of usage, see the Spark NLP Workshop
 * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/scala/training/Train%20Multi-Class%20Text%20Classification%20on%20News%20Articles.scala [1] ]]
 * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.Text_Classification_with_ClassifierDL.ipynb [2] ]]
 * and the [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/classifier/dl/ClassifierDLTestSpec.scala ClassifierDLTestSpec]].
 *
 * ==Example==
 * In this example, the training data `"sentiment.csv"` has the form of
 * {{{
 * text,label
 * This movie is the best movie I have wached ever! In my opinion this movie can win an award.,0
 * This was a terrible movie! The acting was bad really bad!,1
 * ...
 * }}}
 * Then traning can be done like so:
 * {{{
 * import com.johnsnowlabs.nlp.base.DocumentAssembler
 * import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
 * import com.johnsnowlabs.nlp.annotators.classifier.dl.ClassifierDLApproach
 * import org.apache.spark.ml.Pipeline
 *
 * val smallCorpus = spark.read.option("header","true").csv("src/test/resources/classifier/sentiment.csv")
 *
 * val documentAssembler = new DocumentAssembler()
 *   .setInputCol("text")
 *   .setOutputCol("document")
 *
 * val useEmbeddings = UniversalSentenceEncoder.pretrained()
 *   .setInputCols("document")
 *   .setOutputCol("sentence_embeddings")
 *
 * val docClassifier = new ClassifierDLApproach()
 *   .setInputCols("sentence_embeddings")
 *   .setOutputCol("category")
 *   .setLabelColumn("label")
 *   .setBatchSize(64)
 *   .setMaxEpochs(20)
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
 * @see [[MultiClassifierDLApproach]] for multi-class classification
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
class ClassifierDLApproach(override val uid: String)
  extends AnnotatorApproach[ClassifierDLModel]
    with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("ClassifierDL"))

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

  /** Random seed for shuffling the dataset
   *
   * @group param
   * */
  val randomSeed = new IntParam(this, "randomSeed", "Random seed")
  /** Column with label per each document
   *
   * @group param
   * */
  val labelColumn = new Param[String](this, "labelColumn", "Column with label per each document")
  /** Learning Rate (Default: `5e-3f`)
   *
   * @group param
   * */
  val lr = new FloatParam(this, "lr", "Learning Rate")
  /** Batch size (Default: `64`)
   *
   * @group param
   * */
  val batchSize = new IntParam(this, "batchSize", "Batch size")
  /** Dropout coefficient (Default: `0.5f`)
   *
   * @group param
   * */
  val dropout = new FloatParam(this, "dropout", "Dropout coefficient")
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

  /** Random seed
   *
   * @group setParam
   */
  def setRandomSeed(seed: Int): ClassifierDLApproach.this.type = set(randomSeed, seed)

  /** Column with label per each document
   *
   * @group setParam
   * */
  def setLabelColumn(column: String): ClassifierDLApproach.this.type = set(labelColumn, column)

  /** Learning Rate (Default: `5e-3f`)
   *
   * @group setParam
   * */
  def setLr(lr: Float): ClassifierDLApproach.this.type = set(this.lr, lr)

  /** Batch size (Default: `64`)
   *
   * @group setParam
   * */
  def setBatchSize(batch: Int): ClassifierDLApproach.this.type = set(this.batchSize, batch)

  /** Dropout coefficient (Default: `0.5f`)
   *
   * @group setParam
   * */
  def setDropout(dropout: Float): ClassifierDLApproach.this.type = set(this.dropout, dropout)

  /** Maximum number of epochs to train (Default: `10`)
   *
   * @group setParam
   * */
  def setMaxEpochs(epochs: Int): ClassifierDLApproach.this.type = set(maxEpochs, epochs)

  /** Tensorflow config Protobytes passed to the TF session
   *
   * @group setParam
   * */
  def setConfigProtoBytes(bytes: Array[Int]): ClassifierDLApproach.this.type = set(this.configProtoBytes, bytes)

  /** Whether to output to annotators log folder (Default: `false`)
   *
   * @group setParam
   * */
  def setEnableOutputLogs(enableOutputLogs: Boolean): ClassifierDLApproach.this.type = set(this.enableOutputLogs, enableOutputLogs)


  /** Choose the proportion of training dataset to be validated against the model on each Epoch (Default: `0.0f`).
   * The value should be between 0.0 and 1.0 and by default it is 0.0 and off.
   *
   * @group setParam
   * */
  def setValidationSplit(validationSplit: Float): ClassifierDLApproach.this.type = set(this.validationSplit, validationSplit)

  /** Level of verbosity during training (Default: `Verbose.Silent.id`)
   *
   * @group setParam
   * */
  def setVerbose(verbose: Int): ClassifierDLApproach.this.type = set(this.verbose, verbose)

  /** Folder path to save training logs (Default: `""`)
   *
   * @group setParam
   */
  def setOutputLogsPath(path: String): ClassifierDLApproach.this.type = set(this.outputLogsPath, path)

  /** Level of verbosity during training (Default: `Verbose.Silent.id`)
   *
   * @group setParam
   * */
  def setVerbose(verbose: Verbose.Level): ClassifierDLApproach.this.type = set(this.verbose, verbose.id)

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

  /** Learning Rate (Default: `5e-3f`)
   *
   * @group getParam
   * */
  def getLr: Float = $(this.lr)

  /** Batch size (Default: `64`)
   *
   * @group getParam
   * */
  def getBatchSize: Int = $(this.batchSize)

  /** Dropout coefficient (Default: `0.5f`)
   *
   * @group getParam
   * */
  def getDropout: Float = $(this.dropout)

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

  setDefault(
    maxEpochs -> 10,
    lr -> 5e-3f,
    dropout -> 0.5f,
    batchSize -> 64,
    enableOutputLogs -> false,
    verbose -> Verbose.Silent.id,
    validationSplit -> 0.0f,
    outputLogsPath -> ""
  )

  override def beforeTraining(spark: SparkSession): Unit = {}

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): ClassifierDLModel = {

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
    val labelsCount = labels.length

    require(
      labels.length >= 2 && labels.length <= 100,
      s"The total unique number of classes must be more than 2 and less than 100. Currently is ${labels.length}"
    )

    val tfWrapper: TensorflowWrapper = loadSavedModel()

    val settings = ClassifierDatasetEncoderParams(tags = labels)

    val encoder = new ClassifierDatasetEncoder(settings)

    val embeddingsDim = encoder.calculateEmbeddingsDim(train)
    require(
      embeddingsDim <= 1024,
      s"The ClassifierDL only accepts embeddings less than 1024 dimensions. Current dimension is ${embeddingsDim}. Please use embeddings" +
        s" with less than "
    )

    val trainDataset = encoder.collectTrainingInstances(train, getLabelColumn)
    val inputEmbeddings = encoder.extractSentenceEmbeddings(trainDataset)
    val inputLabels = encoder.extractLabels(trainDataset)

    val classifier = try {
      val model = new TensorflowClassifier(
        tensorflow = tfWrapper,
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

    val newWrapper = new TensorflowWrapper(TensorflowWrapper.extractVariablesSavedModel(tfWrapper.getSession(configProtoBytes = getConfigProtoBytes)), tfWrapper.graph)

    val model = new ClassifierDLModel()
      .setDatasetParams(classifier.encoder.params)
      .setModelIfNotSet(dataset.sparkSession, newWrapper)
      .setStorageRef(embeddingsRef)

    if (get(configProtoBytes).isDefined)
      model.setConfigProtoBytes($(configProtoBytes))

    model
  }

  def loadSavedModel(): TensorflowWrapper = {

    val wrapper =
      TensorflowWrapper
        .readZippedSavedModel("/classifier-dl", tags = Array("serve"), initAllTables = true)

    wrapper.variables = Variables(Array.empty[Byte], Array.empty[Byte])
    wrapper
  }
}

/**
 * This is the companion object of [[ClassifierDLApproach]]. Please refer to that class for the documentation.
 */
object ClassifierDLApproach extends DefaultParamsReadable[ClassifierDLApproach]
