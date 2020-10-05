package com.johnsnowlabs.nlp.embeddings

import java.io.File

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.ml.tensorflow.sentencepiece._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}

/** XlnetEmbeddings (XLNet): Generalized Autoregressive Pretraining for Language Understanding
  *
  * Note that this is a very computationally expensive module compared to word embedding modules that only perform embedding lookups.
  * The use of an accelerator is recommended.
  *
  * XLNet is a new unsupervised language representation learning method based on a novel generalized permutation language modeling objective. Additionally, XLNet employs Transformer-XL as the backbone model, exhibiting excellent performance for language tasks involving long context. Overall, XLNet achieves state-of-the-art (SOTA) results on various downstream language tasks including question answering, natural language inference, sentiment analysis, and document ranking.
  *
  * XLNet-Large     = [[https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip]]    | 24-layer, 1024-hidden, 16-heads
  * XLNet-Base    = [[https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip]]   |  12-layer, 768-hidden, 12-heads. This model is trained on full data (different from the one in the paper).
  *
  * @param uid required internal uid for saving annotator
  *
  *            '''Sources :'''
  *
  *            [[ https://arxiv.org/abs/1906.08237]]
  *
  *            [[ https://github.com/zihangdai/xlnet]]
  *
  *            '''Paper abstract: '''
  *
  *            With the capability of modeling bidirectional contexts, denoising autoencoding based pretraining like BERT achieves better performance than pretraining approaches based on autoregressive language modeling. However, relying on corrupting the input with masks, BERT neglects dependency between the masked positions and suffers from a pretrain-finetune discrepancy. In light of these pros and cons, we propose XLNet, a generalized autoregressive pretraining method that (1) enables learning bidirectional contexts by maximizing the expected likelihood over all permutations of the factorization order and (2) overcomes the limitations of BERT thanks to its autoregressive formulation. Furthermore, XLNet integrates ideas from Transformer-XL, the state-of-the-art autoregressive model, into pretraining. Empirically, under comparable experiment settings, XLNet outperforms BERT on 20 tasks, often by a large margin, including question answering, natural language inference, sentiment analysis, and document ranking.
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
class XlnetEmbeddings(override val uid: String)
  extends AnnotatorModel[XlnetEmbeddings]
    with WithAnnotate[XlnetEmbeddings]
    with WriteTensorflowModel
    with WriteSentencePieceModel
    with HasEmbeddingsProperties
    with HasStorageRef
    with HasCaseSensitiveProperties {

  /** Input Annotator Type : TOKEN DOCUMENT
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT, AnnotatorType.TOKEN)
  /** Output Annotator Type : WORD_EMBEDDINGS
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.WORD_EMBEDDINGS

  /** Batch size. Large values allows faster processing but requires more memory.
    *
    * @group param
    **/
  val batchSize = new IntParam(this, "batchSize", "Batch size. Large values allows faster processing but requires more memory.")
  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
    *
    * @group param
    **/
  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")
  /** Max sentence length to process
    *
    * @group param
    **/
  val maxSentenceLength = new IntParam(this, "maxSentenceLength", "Max sentence length to process")

  /** The Tensorflow XLNet Model
    *
    * @group param
    **/
  private var _model: Option[Broadcast[TensorflowXlnet]] = None

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  def this() = this(Identifiable.randomUID("XLNET_EMBEDDINGS"))

  /** Batch size. Large values allows faster processing but requires more memory.
    *
    * @group setParam
    **/
  def setBatchSize(size: Int): this.type = {
    if (get(batchSize).isEmpty)
      set(batchSize, size)
    this
  }

  /**
    * Set dimension of Embeddings
    * Since output shape depends on the model selected, see [[ https://github.com/zihangdai/xlnet]]for further reference
    *
    * @group setParam
    **/
  override def setDimension(value: Int): this.type = {
    if (get(dimension).isEmpty)
      set(this.dimension, value)
    this

  }

  /** Max sentence length to process
    *
    * @group setParam
    **/
  def setMaxSentenceLength(value: Int): this.type = {
    require(value <= 512, "XLNet model does not support sequences longer than 512 because of trainable positional embeddings")

    if (get(maxSentenceLength).isEmpty)
      set(maxSentenceLength, value)
    this
  }

  /** Max sentence length to process
    *
    * @group getParam
    **/
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
    *
    * @group getSaram
    **/
  def setConfigProtoBytes(bytes: Array[Int]): XlnetEmbeddings.this.type = set(this.configProtoBytes, bytes)

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
    *
    * @group setGaram
    **/
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  setDefault(
    batchSize -> 32,
    dimension -> 768,
    maxSentenceLength -> 128,
    caseSensitive -> true
  )

  /** Sets XLNet tensorflow Model
    *
    * @group setParam
    **/
  def setModelIfNotSet(spark: SparkSession, tensorflow: TensorflowWrapper, spp: SentencePieceWrapper): this.type = {
    if (_model.isEmpty) {

      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowXlnet(
            tensorflow,
            spp,
            configProtoBytes = getConfigProtoBytes
          )
        )
      )
    }

    this
  }

  /** Gets XLNet tensorflow Model
    *
    * @group setParam
    **/
  def getModelIfNotSet: TensorflowXlnet = _model.get.value

  /**
    * takes a document and annotations and produces new annotations of this annotator's annotation type
    *
    * @param annotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return any number of annotations processed for every input annotation. Not necessary one to one relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val tokenizedSentences = TokenizedWithSentence.unpack(annotations)

    /*Return empty if the real tokens are empty*/
    if(tokenizedSentences.nonEmpty) {
      val embeddings = getModelIfNotSet.calculateEmbeddings(
        tokenizedSentences,
        $(batchSize),
        $(maxSentenceLength),
        $(caseSensitive)
      )
      WordpieceEmbeddingsSentence.pack(embeddings)
    } else {
      Seq.empty[Annotation]
    }
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(path, spark, getModelIfNotSet.tensorflow, "_xlnet", XlnetEmbeddings.tfFile, configProtoBytes = getConfigProtoBytes)
    writeSentencePieceModel(path, spark, getModelIfNotSet.spp, "_xlnet",  XlnetEmbeddings.sppFile)

  }

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    dataset.withColumn(getOutputCol, wrapEmbeddingsMetadata(dataset.col(getOutputCol), $(dimension), Some($(storageRef))))
  }

}

trait ReadablePretrainedXlnetModel extends ParamsAndFeaturesReadable[XlnetEmbeddings] with HasPretrained[XlnetEmbeddings] {
  override val defaultModelName: Some[String] = Some("xlnet_base_cased")
  /** Java compliant-overrides */
  override def pretrained(): XlnetEmbeddings = super.pretrained()
  override def pretrained(name: String): XlnetEmbeddings = super.pretrained(name)
  override def pretrained(name: String, lang: String): XlnetEmbeddings = super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): XlnetEmbeddings = super.pretrained(name, lang, remoteLoc)
}

trait ReadXlnetTensorflowModel extends ReadTensorflowModel with ReadSentencePieceModel {
  this: ParamsAndFeaturesReadable[XlnetEmbeddings] =>

  override val tfFile: String = "xlnet_tensorflow"
  override val sppFile: String = "xlnet_spp"

  def readTensorflow(instance: XlnetEmbeddings, path: String, spark: SparkSession): Unit = {
    val tf = readTensorflowModel(path, spark, "_xlnet_tf", initAllTables = true)
    val spp = readSentencePieceModel(path, spark, "_xlnet_spp", sppFile)
    instance.setModelIfNotSet(spark, tf, spp)
  }

  addReader(readTensorflow)

  def loadSavedModel(folder: String, spark: SparkSession): XlnetEmbeddings = {

    val f = new File(folder)
    val sppModelPath = folder+"/assets"
    val savedModel = new File(folder, "saved_model.pb")
    val sppModel = new File(sppModelPath, "spiece.model")

    require(f.exists, s"Folder $folder not found")
    require(f.isDirectory, s"File $folder is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $folder"
    )
    require(sppModel.exists(), s"SentencePiece model spiece.model not found in folder $sppModelPath")

    val wrapper = TensorflowWrapper.read(folder, zipped = false, useBundle = true, tags = Array("serve"), initAllTables = true)
    val spp = SentencePieceWrapper.read(sppModel.toString)

    val xlnet = new XlnetEmbeddings()
      .setModelIfNotSet(spark, wrapper, spp)
    xlnet
  }
}


object XlnetEmbeddings extends ReadablePretrainedXlnetModel with ReadXlnetTensorflowModel with ReadSentencePieceModel
