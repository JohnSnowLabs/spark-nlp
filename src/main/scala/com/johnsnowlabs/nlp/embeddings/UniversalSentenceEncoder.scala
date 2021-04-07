package com.johnsnowlabs.nlp.embeddings

import java.io.File
import com.johnsnowlabs.ml.tensorflow.{ReadTensorflowModel, TensorflowUSE, TensorflowWrapper, WriteTensorflowModel}
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp.annotators.common.SentenceSplit
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasPretrained, ParamsAndFeaturesReadable, HasSimpleAnnotate}
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * The Universal Sentence Encoder encodes text into high dimensional vectors that can be used for text classification, semantic similarity, clustering and other natural language tasks.
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/UniversalSentenceEncoderTestSpec.scala]] for further reference on how to use this API.
  *
  * '''Sources :'''
  *
  * [[https://arxiv.org/abs/1803.11175]]
  *
  * [[https://tfhub.dev/google/universal-sentence-encoder/2]]
  *
  * ''' Paper abstract: '''
  * We present models for encoding sentences into embedding vectors that specifically target transfer learning to other NLP tasks. The models are efficient and result in accurate performance on diverse transfer tasks. Two variants of the encoding models allow for trade-offs between accuracy and compute resources. For both variants, we investigate and report the relationship between model complexity, resource consumption, the availability of transfer task training data, and task performance. Comparisons are made with baselines that use word level transfer learning via pretrained word embeddings as well as baselines do not use any transfer learning. We find that transfer learning using sentence embeddings tends to outperform word level transfer. With transfer learning via sentence embeddings, we observe surprisingly good performance with minimal amounts of supervised training data for a transfer task. We obtain encouraging results on Word Embedding Association Tests (WEAT) targeted at detecting model bias. Our pre-trained sentence encoding models are made freely available for download and on TF Hub.
  *
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
  * @groupdesc Parameters A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  */
class UniversalSentenceEncoder(override val uid: String)
    extends AnnotatorModel[UniversalSentenceEncoder] with HasSimpleAnnotate[UniversalSentenceEncoder]
    with HasEmbeddingsProperties
    with HasStorageRef
    with WriteTensorflowModel {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  def this() = this(Identifiable.randomUID("UNIVERSAL_SENTENCE_ENCODER"))

  /** Output annotator type : SENTENCE_EMBEDDINGS
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = SENTENCE_EMBEDDINGS
  /** Input annotator type : DOCUMENT
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)
  /** Number of embedding dimensions
    *
    * @group param
    **/
  override val dimension = new IntParam(this, "dimension", "Number of embedding dimensions")
  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
    *
    * @group param
    **/
  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  val loadSP = new BooleanParam(this, "loadSP", "Whether to load SentencePiece ops file which is required only by multi-lingual models. " +
    "This is not changeable after it's set with a pretrained model nor it is compatible with Windows.")

  /** set loadSP
    *
    * @group setParam
    * */
  def setLoadSP(value: Boolean): this.type = {
    if (get(loadSP).isEmpty)
      set(this.loadSP, value)
    this
  }

  /** get loadSP
    *
    * @group getParam
    **/

  def getLoadSP: Boolean = $(loadSP)

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
    *
    * @group setParam
    **/
  def setConfigProtoBytes(
                           bytes: Array[Int]
                         ): UniversalSentenceEncoder.this.type = set(this.configProtoBytes, bytes)

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
    *
    * @group getParam
    **/
  def getConfigProtoBytes: Option[Array[Byte]] =
    get(this.configProtoBytes).map(_.map(_.toByte))

  private var _model: Option[Broadcast[TensorflowUSE]] = None

  /** @group getParam */
  def getModelIfNotSet: TensorflowUSE = _model.get.value

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, tensorflow: TensorflowWrapper, loadSP: Boolean = false): this.type = {
    if (_model.isEmpty) {

      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowUSE(tensorflow, configProtoBytes = getConfigProtoBytes, loadSP = loadSP)
        )
      )
    }
    this
  }

  setDefault(
    dimension -> 512,
    storageRef -> "tfhub_use",
    loadSP -> false
  )

  /**
    * Takes a document and annotations and produces new annotations of this annotator's annotation type
    *
    * @param annotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return any number of annotations processed for every input annotation. Not necessary one to one relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = SentenceSplit.unpack(annotations)
    val nonEmptySentences = sentences.filter(_.content.nonEmpty)

    if (nonEmptySentences.nonEmpty)
      getModelIfNotSet.calculateEmbeddings(nonEmptySentences)
    else Seq.empty[Annotation]
  }

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    dataset.withColumn(
      getOutputCol,
      wrapSentenceEmbeddingsMetadata(dataset.col(getOutputCol), $(dimension), Some($(storageRef)))
    )
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(
      path,
      spark,
      getModelIfNotSet.tensorflow,
      "_use",
      UniversalSentenceEncoder.tfFile,
      configProtoBytes = getConfigProtoBytes
    )
  }

}

trait ReadablePretrainedUSEModel
    extends ParamsAndFeaturesReadable[UniversalSentenceEncoder]
    with HasPretrained[UniversalSentenceEncoder] {
  override val defaultModelName: Some[String] = Some("tfhub_use")

  /** Java compliant-overrides */
  override def pretrained(): UniversalSentenceEncoder = super.pretrained()
  override def pretrained(name: String): UniversalSentenceEncoder = super.pretrained(name)
  override def pretrained(name: String, lang: String): UniversalSentenceEncoder = super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): UniversalSentenceEncoder =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadUSETensorflowModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[UniversalSentenceEncoder] =>

  /*Needs to point to an actual folder rather than a .pb file*/
  override val tfFile: String = "use_tensorflow"

  def readTensorflow(instance: UniversalSentenceEncoder, path: String, spark: SparkSession): Unit = {
    val loadSP = instance.getLoadSP
    val tf = readTensorflowWithSPModel(path, spark, "_use_tf", initAllTables = true, loadSP = loadSP)
    instance.setModelIfNotSet(spark, tf)
  }

  addReader(readTensorflow)

  def loadSavedModel(folder: String, spark: SparkSession, loadSP: Boolean = false): UniversalSentenceEncoder = {
    val f = new File(folder)
    val savedModel = new File(folder, "saved_model.pb")

    require(f.exists, s"Folder $folder not found")
    require(f.isDirectory, s"File $folder is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $folder"
    )

    val wrapper =
      TensorflowWrapper.readWithSP(folder, zipped = false, useBundle = true, tags = Array("serve"), initAllTables = true, loadSP = loadSP)

    new UniversalSentenceEncoder()
      .setModelIfNotSet(spark, wrapper, loadSP)
      .setLoadSP(loadSP)
  }
}

object UniversalSentenceEncoder extends ReadablePretrainedUSEModel with ReadUSETensorflowModel
