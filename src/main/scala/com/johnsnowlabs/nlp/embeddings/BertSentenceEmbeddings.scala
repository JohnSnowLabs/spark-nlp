package com.johnsnowlabs.nlp.embeddings

import java.io.File

import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.storage.HasStorageRef

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam, BooleanParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * BERT (Bidirectional Encoder Representations from Transformers) provides dense vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture
 *
 *
 * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/BertSentenceEmbeddingsTestSpec.scala]] for further reference on how to use this API.
 * Sources:
 *
 *
 * '''Sources''' :
 *
 * [[https://arxiv.org/abs/1810.04805]]
 *
 * [[https://github.com/google-research/bert]]
 *
 * ''' Paper abstract '''
 *
 * We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.
 * BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1 question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD v2.0 Test F1 to 83.1 (5.1 point absolute improvement).
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
 * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
 * */
class BertSentenceEmbeddings(override val uid: String)
  extends AnnotatorModel[BertSentenceEmbeddings]
    with HasBatchedAnnotate[BertSentenceEmbeddings]
    with WriteTensorflowModel
    with HasEmbeddingsProperties
    with HasStorageRef
    with HasCaseSensitiveProperties {

  def this() = this(Identifiable.randomUID("BERT_SENTENCE_EMBEDDINGS"))

  /** vocabulary
   *
   * @group param
   * */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary")

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
   *
   * @group param
   * */
  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** Max sentence length to process
   *
   * @group param
   * */
  val maxSentenceLength = new IntParam(this, "maxSentenceLength", "Max sentence length to process")

  /** Use Long type instead of Int type for inputs
   *
   * @group param
   * */
  val isLong = new BooleanParam(parent = this, name = "isLong", "Use Long type instead of Int type for inputs buffer - Some Bert models require Long instead of Int.")

  /** set isLong
   *
   * @group setParam
   * */
  def setIsLong(value: Boolean): this.type = {
    if (get(isLong).isEmpty)
      set(this.isLong, value)
    this
  }

  /** get isLong
   *
   * @group getParam
   * */

  def getIsLong: Boolean = $(isLong)

  /** @group setParam */
  def sentenceStartTokenId: Int = {
    $$(vocabulary)("[CLS]")
  }

  /** @group setParam */
  def sentenceEndTokenId: Int = {
    $$(vocabulary)("[SEP]")
  }

  /** Set Embeddings dimensions for the BERT model
   * Only possible to set this when the first time is saved
   * dimension is not changeable, it comes from BERT config file
   *
   * @group setParam
   * */
  override def setDimension(value: Int): this.type = {
    if (get(dimension).isEmpty)
      set(this.dimension, value)
    this

  }

  /** Whether to lowercase tokens or not
   *
   * @group setParam
   * */
  override def setCaseSensitive(value: Boolean): this.type = {
    if (get(caseSensitive).isEmpty)
      set(this.caseSensitive, value)
    this
  }

  /** Vocabulary used to encode the words to ids with WordPieceEncoder
   *
   * @group setParam
   * */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
   *
   * @group setParam
   * */
  def setConfigProtoBytes(bytes: Array[Int]): BertSentenceEmbeddings.this.type = set(this.configProtoBytes, bytes)

  /**
   * Max sentence length to process
   *
   * @group setParam
   * */
  def setMaxSentenceLength(value: Int): this.type = {
    require(value <= 512, "BERT models do not support sequences longer than 512 because of trainable positional embeddings")

    if (get(maxSentenceLength).isEmpty)
      set(maxSentenceLength, value)
    this
  }

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
   *
   * @group getParam
   * */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** Max sentence length to process
   *
   * @group getParam
   * */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  setDefault(
    dimension -> 768,
    batchSize -> 8,
    maxSentenceLength -> 128,
    caseSensitive -> false,
    isLong -> false
  )

  private var _model: Option[Broadcast[TensorflowBert]] = None

  /** @group getParam */
  def getModelIfNotSet: TensorflowBert = _model.get.value

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, tensorflow: TensorflowWrapper): this.type = {
    if (_model.isEmpty) {

      _model = Some(spark.sparkContext.broadcast(
        new TensorflowBert(
          tensorflow,
          sentenceStartTokenId,
          sentenceEndTokenId,
          configProtoBytes = getConfigProtoBytes
        )))
    }

    this
  }

  def tokenize(sentences: Seq[Sentence]): Seq[WordpieceTokenizedSentence] = {
    val basicTokenizer = new BasicTokenizer($(caseSensitive))
    val encoder = new WordpieceEncoder($$(vocabulary))
    sentences.map { s =>
      val tokens = basicTokenizer.tokenize(s)
      val wordpieceTokens = tokens.flatMap(token => encoder.encode(token))
      WordpieceTokenizedSentence(wordpieceTokens)
    }
  }

  /**
   * takes a document and annotations and produces new annotations of this annotator's annotation type
   *
   * @param batchedAnnotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
   * @return any number of annotations processed for every input annotation. Not necessary one to one relationship
   */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    /*Return empty if the real sentences are empty*/
    batchedAnnotations.map(annotations => {
      val sentences = SentenceSplit.unpack(annotations).toArray

      if (sentences.nonEmpty) {
        val tokenized = tokenize(sentences)
        getModelIfNotSet.calculateSentenceEmbeddings(
          tokenized,
          sentences,
          $(batchSize),
          $(maxSentenceLength),
          getIsLong
        )
      } else {
        Seq.empty[Annotation]
      }
    })
  }

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    dataset.withColumn(
      getOutputCol,
      wrapSentenceEmbeddingsMetadata(dataset.col(getOutputCol), $(dimension), Some($(storageRef)))
    )
  }

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModelV2(path, spark, getModelIfNotSet.tensorflowWrapper, "_bert_sentence", BertSentenceEmbeddings.tfFile, configProtoBytes = getConfigProtoBytes)
  }

}

trait ReadablePretrainedBertSentenceModel extends ParamsAndFeaturesReadable[BertSentenceEmbeddings] with HasPretrained[BertSentenceEmbeddings] {
  override val defaultModelName: Some[String] = Some("sent_small_bert_L2_768")

  /** Java compliant-overrides */
  override def pretrained(): BertSentenceEmbeddings = super.pretrained()

  override def pretrained(name: String): BertSentenceEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): BertSentenceEmbeddings = super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): BertSentenceEmbeddings = super.pretrained(name, lang, remoteLoc)
}

trait ReadBertSentenceTensorflowModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[BertSentenceEmbeddings] =>

  override val tfFile: String = "bert_sentence_tensorflow"

  def readTensorflow(instance: BertSentenceEmbeddings, path: String, spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_bert_sentence_tf")
    instance.setModelIfNotSet(spark, tf)
  }

  addReader(readTensorflow)

  def loadSavedModel(folder: String, spark: SparkSession): BertSentenceEmbeddings = {

    val f = new File(folder)
    val savedModel = new File(folder, "saved_model.pb")
    require(f.exists, s"Folder $folder not found")
    require(f.isDirectory, s"File $folder is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $folder"
    )

    val vocab = new File(folder + "/assets", "vocab.txt")
    require(f.exists, s"Folder $folder not found")
    require(f.isDirectory, s"File $folder is not folder")
    require(vocab.exists(), s"Vocabulary file vocab.txt not found in folder $folder")

    val vocabResource = new ExternalResource(vocab.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    val words = ResourceHelper.parseLines(vocabResource).zipWithIndex.toMap

    val (wrapper, _) = TensorflowWrapper.read(folder, zipped = false, useBundle = true, tags = Array("serve"))

    new BertSentenceEmbeddings()
      .setVocabulary(words)
      .setModelIfNotSet(spark, wrapper)
  }
}


object BertSentenceEmbeddings extends ReadablePretrainedBertSentenceModel with ReadBertSentenceTensorflowModel
