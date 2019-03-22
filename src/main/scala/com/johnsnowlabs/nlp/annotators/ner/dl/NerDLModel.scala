package com.johnsnowlabs.nlp.annotators.ner.dl


import com.johnsnowlabs.ml.tensorflow.{DatasetEncoderParams, NerDatasetEncoder, TensorflowNer, TensorflowWrapper}
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.Annotated.NerTaggedSentence
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.ml.tensorflow.{ReadTensorflowModel, WriteTensorflowModel}
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.serialization.StructFeature
import org.apache.spark.ml.param.{FloatParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession


class NerDLModel(override val uid: String)
  extends AnnotatorModel[NerDLModel]
    with WriteTensorflowModel
    with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("NerDLModel"))

  override val inputAnnotatorTypes = Array(DOCUMENT, TOKEN, WORD_EMBEDDINGS)
  override val outputAnnotatorType = NAMED_ENTITY

  val minProba = new FloatParam(this, "minProbe", "Minimum probability. Used only if there is no CRF on top of LSTM layer.")
  def setMinProbability(minProba: Float) = set(this.minProba, minProba)

  val batchSize = new IntParam(this, "batchSize", "Size of every batch.")
  def setBatchSize(size: Int) = set(this.batchSize, size)

  val datasetParams = new StructFeature[DatasetEncoderParams](this, "datasetParams")
  def setDatasetParams(params: DatasetEncoderParams) = set(this.datasetParams, params)

  private var tensorflow: TensorflowWrapper = null

  def setTensorflow(tf: TensorflowWrapper): NerDLModel = {
    this.tensorflow = tf
    this
  }

  @transient private var _model: TensorflowNer = null

  def getModelIfNotSet: TensorflowNer = {
    if (_model == null) {
      require(tensorflow != null, "Tensorflow must be set before usage. Use method setTensorflow() for it.")
      require(datasetParams.isSet, "datasetParams must be set before usage")

      val encoder = new NerDatasetEncoder(datasetParams.get.get)
      _model = new TensorflowNer(
        tensorflow,
        encoder,
        1, // Tensorflow doesn't clear state in batch
        Verbose.Silent)
    }

    _model
  }

  def tag(tokenized: Array[WordpieceEmbeddingsSentence]): Array[NerTaggedSentence] = {
    // Predict
    val labels = getModelIfNotSet.predict(tokenized)

    // Combine labels with sentences tokens
    tokenized.indices.map { i =>
      val sentence = tokenized(i)

      val tokens = sentence.tokens.indices.flatMap { j =>
        val token = sentence.tokens(j)
        val label = labels(i)(j)
        if (token.isWordStart) {
          Some(IndexedTaggedWord(token.token, label, token.begin, token.end))
        }
        else {
          None
        }
      }.toArray

      new TaggedSentence(tokens)
    }.toArray
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
    writeTensorflowModel(path, spark, tensorflow, "_nerdl", NerDLModel.tfFile)
  }
}

trait ReadsNERGraph extends ParamsAndFeaturesReadable[NerDLModel] with ReadTensorflowModel {

  override val tfFile = "tensorflow"

  def readNerGraph(instance: NerDLModel, path: String, spark: SparkSession): Unit = {
    val tf = readTensorflowModel(path, spark, "_nerdl")
    instance.setTensorflow(tf)
  }

  addReader(readNerGraph)
}

trait PretrainedNerDL {
  def pretrained(name: String = "ner_precise", language: Option[String] = Some("en"), remoteLoc: String = ResourceDownloader.publicLoc): NerDLModel =
    ResourceDownloader.downloadModel(NerDLModel, name, language, remoteLoc)
}


object NerDLModel extends ParamsAndFeaturesReadable[NerDLModel] with ReadsNERGraph with PretrainedNerDL
