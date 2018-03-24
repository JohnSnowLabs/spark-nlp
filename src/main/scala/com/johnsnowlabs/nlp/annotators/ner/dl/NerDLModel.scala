package com.johnsnowlabs.nlp.annotators.ner.dl

import java.nio.file.{Files, Paths}
import java.util.UUID

import com.johnsnowlabs.ml.tensorflow.{DatasetEncoder, DatasetEncoderParams, TensorflowNer, TensorflowWrapper}
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, NAMED_ENTITY, TOKEN}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.Annotated.NerTaggedSentence
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.ner.Verbose
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.util.FileHelper
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.param.{FloatParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

import scala.collection.JavaConverters._


class NerDLModel(override val uid: String)
  extends AnnotatorModel[NerDLModel]
    with HasWordEmbeddings
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

      val encoder = new DatasetEncoder(embeddings.get.getEmbeddings, datasetParams.get.get)
      _model = new TensorflowNer(
        tensorflow,
        encoder,
        1,//${batchSize}, For some reasons Tensorflow doesn't clear state in batch
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

    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)

    // 1. Create tmp folder
    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_nerdl")
      .toAbsolutePath.toString
    val tfFile = Paths.get(tmpFolder, NerDLModel.tfFile).toString

    // 2. Save Tensorflow state
    tensorflow.saveToFile(tfFile)

    // 3. Copy to dest folder
    fs.copyFromLocalFile(new Path(tfFile), new Path(path))

    // 4. Remove tmp folder
    FileHelper.delete(tmpFolder)
  }
}

object NerDLModel extends ParamsAndFeaturesReadable[NerDLModel] {

  val tfFile = "tensorflow"

  override def onRead(instance: NerDLModel, path: String, spark: SparkSession): Unit = {

    val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)

    // 1. Create tmp directory
    val tmpFolder = Files.createTempDirectory(UUID.randomUUID().toString.takeRight(12) + "_nerdl")
      .toAbsolutePath.toString

    // 2. Copy to local dir
    fs.copyToLocalFile(new Path(path, tfFile), new Path(tmpFolder))

    // 3. Read Tensorflow state
    val tf = TensorflowWrapper.read(new Path(tmpFolder, tfFile).toString)
    instance.setTensorflow(tf)

    // 4. Remove tmp folder
    FileHelper.delete(tmpFolder)
  }
}
