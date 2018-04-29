package com.johnsnowlabs.nlp.annotators.ner.dl


import com.johnsnowlabs.ml.tensorflow.{DatasetEncoderParams, NerDatasetEncoder, TensorflowNer, TensorflowWrapper}
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, NAMED_ENTITY, TOKEN}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.assertion.dl.{ReadTensorflowModel, WriteTensorflowModel}
import com.johnsnowlabs.nlp.annotators.common.Annotated.NerTaggedSentence
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.embeddings.EmbeddingsReadable
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.serialization.StructFeature
import org.apache.spark.ml.param.{FloatParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession


class NerDLModel(override val uid: String)
  extends AnnotatorModel[NerDLModel]
    with HasWordEmbeddings
    with WriteTensorflowModel
    with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("NerDLModel"))

  override val requiredAnnotatorTypes = Array(DOCUMENT, TOKEN)
  override val annotatorType = NAMED_ENTITY


  val minProba = new FloatParam(this, "minProbe", "Minimum probability. Used only if there is no CRF on top of LSTM layer.")
  def setMinProbability(minProba: Float) = set(this.minProba, minProba)

  val batchSize = new IntParam(this, "batchSize", "Size of every batch.")
  def setBatchSize(size: Int) = set(this.batchSize, size)

  val datasetParams = new StructFeature[DatasetEncoderParams](this, "datasetParams")
  def setDatasetParams(params: DatasetEncoderParams) = set(this.datasetParams, params)

  var tensorflow: TensorflowWrapper = null

  def setTensorflow(tf: TensorflowWrapper): NerDLModel = {
    this.tensorflow = tf
    this
  }

  @transient
  private var _model: TensorflowNer = null

  def model: TensorflowNer = {
    if (_model == null) {
      require(tensorflow != null, "Tensorflow must be set before usage. Use method setTensorflow() for it.")
      require(embeddings.isDefined, "Embeddings must be defined before usage")
      require(datasetParams.isSet, "datasetParams must be set before usage")

      val encoder = new NerDatasetEncoder(embeddings.get.getEmbeddings, datasetParams.get.get)
      _model = new TensorflowNer(
        tensorflow,
        encoder,
        1, // Tensorflow doesn't clear state in batch
        Verbose.Silent)
    }

    _model
  }

  def tag(tokenized: Array[TokenizedSentence]): Array[NerTaggedSentence] = {
    // Predict
    val labels = model.predict(tokenized)

    // Combine labels with sentences tokens
    (0 until tokenized.length).map { i =>
      val sentence = tokenized(i)

      val tokens = (0 until sentence.tokens.length).map { j =>
        val token = sentence.indexedTokens(j)
        val label = labels(i)(j)
        IndexedTaggedWord(token.token, label, token.begin, token.end)
      }.toArray

      new TaggedSentence(tokens)
    }.toArray
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    require(model != null, "call setModel before usage")

    // Parse
    val tokenized = TokenizedWithSentence.unpack(annotations).toArray

    // Predict
    val tagged = tag(tokenized)

    // Pack
    NerTagged.pack(tagged)
  }


  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(path, spark, tensorflow, "_nerdl")
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


object NerDLModel extends EmbeddingsReadable[NerDLModel] with ReadsNERGraph
