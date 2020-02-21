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
import org.apache.spark.ml.param.{BooleanParam, FloatParam, IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, SparkSession}


class NerDLModel(override val uid: String)
  extends AnnotatorModel[NerDLModel]
    with WriteTensorflowModel
    with HasStorageRef
    with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("NerDLModel"))

  override val inputAnnotatorTypes = Array(DOCUMENT, TOKEN, WORD_EMBEDDINGS)
  override val outputAnnotatorType = NAMED_ENTITY

  val minProba = new FloatParam(this, "minProbe", "Minimum probability. Used only if there is no CRF on top of LSTM layer.")
  val batchSize = new IntParam(this, "batchSize", "Size of every batch.")
  val datasetParams = new StructFeature[DatasetEncoderParams](this, "datasetParams")
  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")
  val includeConfidence = new BooleanParam(this, "includeConfidence", "whether to include confidence scores in annotation metadata")

  setDefault(includeConfidence, false)

  def setMinProbability(minProba: Float) = set(this.minProba, minProba)
  def setBatchSize(size: Int) = set(this.batchSize, size)
  def setDatasetParams(params: DatasetEncoderParams) = set(this.datasetParams, params)
  def setConfigProtoBytes(bytes: Array[Int]) = set(this.configProtoBytes, bytes)
  def setIncludeConfidence(value: Boolean) = set(this.includeConfidence, value)

  def getMinProba: Float = $(this.minProba)
  def getBatchSize: Int = $(this.batchSize)
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))
  def getModelIfNotSet: TensorflowNer = _model.get.value
  def getIncludeConfidence: Boolean = $(includeConfidence)

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
