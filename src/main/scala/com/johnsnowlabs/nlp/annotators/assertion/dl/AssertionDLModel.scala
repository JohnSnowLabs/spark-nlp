package com.johnsnowlabs.nlp.annotators.assertion.dl

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.embeddings.EmbeddingsReadable
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import org.apache.spark.ml.param.{IntParam, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

/**
  * Created by jose on 14/03/18.
  */
class AssertionDLModel(override val uid: String) extends AnnotatorModel[AssertionDLModel]
  with ModelWithWordEmbeddings
  with WriteTensorflowModel
  with ParamsAndFeaturesWritable {

  override val requiredAnnotatorTypes: Array[String] = Array(DOCUMENT, CHUNK)
  override val annotatorType: AnnotatorType = ASSERTION

  def this() = this(Identifiable.randomUID("ASSERTION"))

  def setDatasetParams(params: DatasetEncoderParams): AssertionDLModel = set(this.datasetParams, params)

  var tensorflow: TensorflowWrapper = null

  def setTensorflow(tf: TensorflowWrapper): AssertionDLModel = {
    this.tensorflow = tf
    this
  }

  val batchSize = new IntParam(this, "batchSize", "Size of every batch.")
  def setBatchSize(size: Int): this.type = set(batchSize, size)

  val datasetParams = new StructFeature[DatasetEncoderParams](this, "datasetParams")

  @transient
  private var _model: TensorflowAssertion = null

  def model: TensorflowAssertion = {
    if (_model == null) {
      require(tensorflow != null, "Tensorflow must be set before usage. Use method setTensorflow() for it.")
      require(datasetParams.isSet, "datasetParams must be set before usage")

      val encoder = new AssertionDatasetEncoder(embeddings.getEmbeddings, datasetParams.get.get)
      _model = new TensorflowAssertion(
        tensorflow,
        encoder,
        ${batchSize},
        Verbose.Silent)
    }

    _model
  }

  private case class IndexedChunk(sentenceTokens: Array[String], chunkBegin: Int, chunkEnd: Int)

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    /** Take all raw sentences */
    val sentences = annotations
      .filter(_.annotatorType == AnnotatorType.DOCUMENT)
      .map(_.result)
      .toArray

    /** Take all chunk content */
    val chunks = annotations
      .filter(_.annotatorType == AnnotatorType.CHUNK)
      .map(_.result)
      .toArray

    /** Find chunk index in sentence, reference to entire sentence*/
    val indexed = sentences.flatMap(sentence => {
      chunks.flatMap(chunk => {
        if (sentence.contains(chunk)) {
          val tokenIndexBegin = sentence.indexOf(chunk)
          val tokenIndexEnd = tokenIndexBegin + chunk.length - 1
          Some(IndexedChunk(sentence.split(" "), tokenIndexBegin, tokenIndexEnd))
        } else {
          None
        }
      })
    })

    /** Predict with chunk indexes s */
    indexed.map(marked => {
      val prediction = model.predict(Array(marked.sentenceTokens), Array(marked.chunkBegin), Array(marked.chunkEnd)).head
      Annotation(ASSERTION, marked.chunkBegin, marked.chunkEnd, prediction, Map())
    })

  }

  /** requirement for annotators copies */
  override def copy(extra: ParamMap): AssertionDLModel = defaultCopy(extra)

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(path, spark, tensorflow, "_assertiondl")
  }

}

trait ReadsAssertionGraph extends ParamsAndFeaturesReadable[AssertionDLModel] with ReadTensorflowModel {

  override val tfFile = "tensorflow"

  def readAssertionGraph(instance: AssertionDLModel, path: String, spark: SparkSession): Unit = {
    val tf = readTensorflowModel(path, spark, "_assertiondl")
    instance.setTensorflow(tf)
  }

  addReader(readAssertionGraph)
}

trait PretrainedDLAssertionStatus {
  def pretrained(name: String = "as_fast_dl", language: Option[String] = Some("en"), folder: String = ResourceDownloader.publicLoc): AssertionDLModel =
    ResourceDownloader.downloadModel(AssertionDLModel, name, language, folder)
}

object AssertionDLModel extends EmbeddingsReadable[AssertionDLModel] with ReadsAssertionGraph with PretrainedDLAssertionStatus