package com.johnsnowlabs.nlp.annotators.ner.dl


import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.Annotated.NerTaggedSentence
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, FloatParam, IntArrayParam, IntParam, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, SparkSession}

/**
  * This Named Entity recognition annotator allows to train generic NER model based on Neural Networks. Its train data (train_ner) is either a labeled or an external CoNLL 2003 IOB based spark dataset with Annotations columns. Also the user has to provide word embeddings annotation column.
  * Neural Network architecture is Char CNNs - BiLSTM - CRF that achieves state-of-the-art in most datasets.
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/ner/dl]] for further reference on how to use this API.
  **/
class NerDLModel(override val uid: String)
  extends AnnotatorModel[NerDLModel]
    with WriteTensorflowModel
    with HasStorageRef
    with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("NerDLModel"))

  /** Required input Annotators coulumns, expects DOCUMENT, TOKEN, WORD_EMBEDDINGS
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN, WORD_EMBEDDINGS)
  /** Output Annnotator type : NAMED_ENTITY
    *
    * @group anno
    **/
  override val outputAnnotatorType: String = NAMED_ENTITY

  /** Minimum probability. Used only if there is no CRF on top of LSTM layer.
    *
    * @group param
    **/
  val minProba = new FloatParam(this, "minProbe", "Minimum probability. Used only if there is no CRF on top of LSTM layer.")
  /** Size of every batch.
    *
    * @group param
    **/
  val batchSize = new IntParam(this, "batchSize", "Size of every batch.")
  /** datasetParams
    *
    * @group param
    **/
  val datasetParams = new StructFeature[DatasetEncoderParams](this, "datasetParams")
  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
    *
    * @group param
    **/
  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")
  /** whether to include confidence scores in annotation metadata
    *
    * @group param
    **/
  val includeConfidence = new BooleanParam(this, "includeConfidence", "whether to include confidence scores in annotation metadata")

  val classes = new StringArrayParam(this, "classes", "keep an internal copy of classes for Python")

  setDefault(
    includeConfidence -> false
  )

  /** Minimum probability. Used only if there is no CRF on top of LSTM layer.
    *
    * @group setParam
    **/
  def setMinProbability(minProba: Float): this.type = set(this.minProba, minProba)

  /** Size of every batch.
    *
    * @group setParam
    **/
  def setBatchSize(size: Int): this.type = set(this.batchSize, size)

  /** datasetParams
    *
    * @group setParam
    **/
  def setDatasetParams(params: DatasetEncoderParams): this.type = set(this.datasetParams, params)

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
    *
    * @group setParam
    **/
  def setConfigProtoBytes(bytes: Array[Int]): this.type = set(this.configProtoBytes, bytes)

  /** whether to include confidence scores in annotation metadata
    *
    * @group setParam
    **/
  def setIncludeConfidence(value: Boolean): this.type = set(this.includeConfidence, value)

  /** Minimum probability. Used only if there is no CRF on top of LSTM layer.
    *
    * @group getParam
    **/
  def getMinProba: Float = $(this.minProba)

  /** Size of every batch.
    *
    * @group getParam
    **/
  def getBatchSize: Int = $(this.batchSize)

  /** datasetParams
    *
    * @group getParam
    **/
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
    *
    * @group getParam
    **/
  def getModelIfNotSet: TensorflowNer = _model.get.value

  /** whether to include confidence scores in annotation metadata
    *
    * @group getParam
    **/
  def getIncludeConfidence: Boolean = $(includeConfidence)

  /** get the tags used to trained this NerDLModel
    *
    * @group getParam
    **/
  def getClasses: Array[String] = {
    val encoder = new NerDatasetEncoder(datasetParams.get.get)
    set(classes, encoder.tags)
    encoder.tags
  }

  def tag(tokenized: Array[WordpieceEmbeddingsSentence]): Array[NerTaggedSentence] = {
    // Predict
    val labels = getModelIfNotSet.predict(tokenized, getConfigProtoBytes, includeConfidence = $(includeConfidence))

    // Combine labels with sentences tokens
    tokenized.indices.map { i =>
      val sentence = tokenized(i)

      val tokens = sentence.tokens.indices.flatMap { j =>
        val token = sentence.tokens(j)
        val label = labels(i)(j)
        if (token.isWordStart) {
          Some(IndexedTaggedWord(token.token, label._1, token.begin, token.end, label._2.map(_.toFloat)))
        }
        else {
          None
        }
      }.toArray

      new TaggedSentence(tokens)
    }.toArray
  }

  def setModelIfNotSet(spark: SparkSession, tf: TensorflowWrapper): this.type = {
    if (_model.isEmpty) {
      require(datasetParams.isSet, "datasetParams must be set before usage")

      val encoder = new NerDatasetEncoder(datasetParams.get.get)
      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowNer(
            tf,
            encoder,
            1, // Tensorflow doesn't clear state in batch
            Verbose.Silent
          )
        )
      )
    }
    this
  }

  private var _model: Option[Broadcast[TensorflowNer]] = None

  override protected def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    validateStorageRef(dataset, $(inputCols), AnnotatorType.WORD_EMBEDDINGS)
    dataset
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    // Parse
    val tokenized = WordpieceEmbeddingsSentence.unpack(annotations).toArray

    // Predict
    val tagged = tag(tokenized)

    // Pack
    NerTagged.pack(tagged)
  }


  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(path, spark, getModelIfNotSet.tensorflow, "_nerdl", NerDLModel.tfFile, configProtoBytes = getConfigProtoBytes)
  }

}

trait ReadsNERGraph extends ParamsAndFeaturesReadable[NerDLModel] with ReadTensorflowModel {

  override val tfFile = "tensorflow"

  def readNerGraph(instance: NerDLModel, path: String, spark: SparkSession): Unit = {
    val tf = readTensorflowModel(path, spark, "_nerdl")
    instance.setModelIfNotSet(spark: SparkSession, tf)
    // This allows for Python to access getClasses function
    val encoder = new NerDatasetEncoder(instance.datasetParams.get.get)
    instance.set(instance.classes, encoder.tags)
  }

  addReader(readNerGraph)
}

trait ReadablePretrainedNerDL extends ParamsAndFeaturesReadable[NerDLModel] with HasPretrained[NerDLModel] {
  override val defaultModelName: Some[String] = Some("ner_dl")

  override def pretrained(name: String, lang: String, remoteLoc: String): NerDLModel = {
    ResourceDownloader.downloadModel(NerDLModel, name, Option(lang), remoteLoc)
  }
  /** Java compliant-overrides */
  override def pretrained(): NerDLModel = pretrained(defaultModelName.get, defaultLang, defaultLoc)
  override def pretrained(name: String): NerDLModel = pretrained(name, defaultLang, defaultLoc)
  override def pretrained(name: String, lang: String): NerDLModel = pretrained(name, lang, defaultLoc)
}


object NerDLModel extends ReadablePretrainedNerDL with ReadsNERGraph
