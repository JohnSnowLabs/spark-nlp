package com.johnsnowlabs.nlp.annotators.ner.dl


import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.Annotated.NerTaggedSentence
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.serialization.StructFeature
import org.apache.commons.lang.SystemUtils
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{FloatParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{Dataset, SparkSession}


class NerDLModel(override val uid: String)
  extends AnnotatorModel[NerDLModel]
    with WriteTensorflowModel
    with ParamsAndFeaturesWritable
    with LoadsContrib {

  def this() = this(Identifiable.randomUID("NerDLModel"))

  override val inputAnnotatorTypes = Array(DOCUMENT, TOKEN, WORD_EMBEDDINGS)
  override val outputAnnotatorType = NAMED_ENTITY

  val minProba = new FloatParam(this, "minProbe", "Minimum probability. Used only if there is no CRF on top of LSTM layer.")
  def setMinProbability(minProba: Float) = set(this.minProba, minProba)

  val batchSize = new IntParam(this, "batchSize", "Size of every batch.")
  def setBatchSize(size: Int) = set(this.batchSize, size)

  val datasetParams = new StructFeature[DatasetEncoderParams](this, "datasetParams")
  def setDatasetParams(params: DatasetEncoderParams) = set(this.datasetParams, params)

  def getModelIfNotSet: TensorflowNer = _model.get.value

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

  override def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {

    require(_model.isDefined, "Tensorflow model has not been initialized")

    loadContribToCluster(dataset.sparkSession)

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
    writeTensorflowModel(path, spark, getModelIfNotSet.tensorflow, "_nerdl", NerDLModel.tfFile, loadsContrib = true)
  }
}

trait ReadsNERGraph extends ParamsAndFeaturesReadable[NerDLModel] with ReadTensorflowModel {

  override val tfFile = "tensorflow"

  def readNerGraph(instance: NerDLModel, path: String, spark: SparkSession): Unit = {
    val tf = readTensorflowModel(path, spark, "_nerdl", loadContrib = true)
    instance.setModelIfNotSet(spark: SparkSession, tf)
  }

  addReader(readNerGraph)
}

trait PretrainedNerDL {
  def pretrained(name: String = "ner_dl", language: Option[String] = Some("en"), remoteLoc: String = ResourceDownloader.publicLoc): NerDLModel = {
    val finalName = if (name == "ner_dl") {
      if (SystemUtils.IS_OS_WINDOWS)
        "ner_dl"
      else
        // Download better model if not windows
        "ner_dl_contrib"
      }
    else name
    ResourceDownloader.downloadModel(NerDLModel, name, language, remoteLoc)
  }
}


object NerDLModel extends ParamsAndFeaturesReadable[NerDLModel] with ReadsNERGraph with PretrainedNerDL
