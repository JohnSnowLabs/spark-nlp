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
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * BERT (Bidirectional Encoder Representations from Transformers) provides dense vector representations for natural language by using a deep, pre-trained neural network with the Transformer architecture
  *
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/BertEmbeddingsTestSpec.scala]] for further reference on how to use this API.
  * Sources:
  *
  *
  * 0  : corresponds to first layer (embeddings)
  *
  * -1 :  corresponds to last layer
  *
  * 2  :  second-to-last layer
  *
  * Paper:  [[ https://arxiv.org/abs/1810.04805]]
  *
  * Source:  [[https://github.com/google-research/bert]]
  **/
class BertEmbeddings(override val uid: String) extends
  AnnotatorModel[BertEmbeddings]
  with WriteTensorflowModel
  with HasEmbeddingsProperties
  with HasStorageRef
  with HasCaseSensitiveProperties {

  def this() = this(Identifiable.randomUID("BERT_EMBEDDINGS"))

  /** Batch size. Large values allows faster processing but requires more memory. */
  val batchSize = new IntParam(this, "batchSize", "Batch size. Large values allows faster processing but requires more memory.")
  /** vocabulary */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary")
  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString() */
  val configProtoBytes = new IntArrayParam(this, "configProtoBytes", "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")
  /** Max sentence length to process */
  val maxSentenceLength = new IntParam(this, "maxSentenceLength", "Max sentence length to process")
  /** Set BERT pooling layer to: -1 for last hidden layer, -2 for second-to-last hidden layer, and 0 for first layer which is called embeddings */
  val poolingLayer = new IntParam(this, "poolingLayer", "Set BERT pooling layer to: -1 for last hidden layer, -2 for second-to-last hidden layer, and 0 for first layer which is called embeddings")

  def sentenceStartTokenId: Int = {
    $$(vocabulary)("[CLS]")
  }

  def sentenceEndTokenId: Int = {
    $$(vocabulary)("[SEP]")
  }

  /**
    * Defines the output layer of BERT when calculating Embeddings. See extractPoolingLayer() in TensorflowBert for further reference.
    **/
  override def setDimension(value: Int): this.type = {
    if (get(dimension).isEmpty)
      set(this.dimension, value)
    this

  }


  /** Whether to lowercase tokens or not
    * */
  override def setCaseSensitive(value: Boolean): this.type = {
    if (get(caseSensitive).isEmpty)
      set(this.caseSensitive, value)
    this
  }


  /** Batch size. Large values allows faster processing but requires more memory.
    * */
  def setBatchSize(size: Int): this.type = {
    if (get(batchSize).isEmpty)
      set(batchSize, size)
    this
  }


  /** Vocabulary used to encode the words to ids with WordPieceEncoder
    * */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  /** ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()
    * */
  def setConfigProtoBytes(bytes: Array[Int]): BertEmbeddings.this.type = set(this.configProtoBytes, bytes)

  /**
    * Max sentence length to process
    **/
  def setMaxSentenceLength(value: Int): this.type = {
    if (get(maxSentenceLength).isEmpty)
      set(maxSentenceLength, value)
    this
  }

  /**
    * PoolingLayer must be either
    *
    * 0  : corresponds to first layer (embeddings)
    *
    * -1 :  corresponds to last layer
    *
    * 2  :  second-to-last layer
    *
    * Since output shape depends on the model selected, see [[https://github.com/google-research/bert]] for further reference
    **/
  def setPoolingLayer(layer: Int): this.type = {
    layer match {
      case 0 => set(poolingLayer, 0)
      case -1 => set(poolingLayer, -1)
      case -2 => set(poolingLayer, -2)
      case _ => throw new MatchError("poolingLayer must be either 0, -1, or -2: first layer (embeddings), last layer, second-to-last layer")
    }
  }

  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** Max sentence length to process
    * */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  /** Get currently configured BERT output layer
    * */
  def getPoolingLayer: Int = $(poolingLayer)

  setDefault(
    dimension -> 768,
    batchSize -> 32,
    maxSentenceLength -> 128,
    caseSensitive -> true,
    poolingLayer -> 0
  )

  private var _model: Option[Broadcast[TensorflowBert]] = None

  def getModelIfNotSet: TensorflowBert = _model.get.value

  def setModelIfNotSet(spark: SparkSession, tensorflow: TensorflowWrapper): this.type = {
    if (_model.isEmpty) {

      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowBert(
            tensorflow,
            sentenceStartTokenId,
            sentenceEndTokenId,
            configProtoBytes = getConfigProtoBytes
          )
        )
      )
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
    * @param annotations Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return any number of annotations processed for every input annotation. Not necessary one to one relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val tokenizedSentences = TokenizedWithSentence.unpack(annotations)
    /*Return empty if the real tokens are empty*/
    if(tokenizedSentences.nonEmpty) {
      val sentences = SentenceSplit.unpack(annotations)
      val tokenized = tokenize(sentences)
      val withEmbeddings = getModelIfNotSet.calculateEmbeddings(
        tokenized,
        tokenizedSentences,
        $(poolingLayer),
        $(batchSize),
        $(maxSentenceLength),
        $(dimension),
        $(caseSensitive)
      )
      WordpieceEmbeddingsSentence.pack(withEmbeddings)
    } else {
      Seq.empty[Annotation]
    }
  }

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    dataset.withColumn(getOutputCol, wrapEmbeddingsMetadata(dataset.col(getOutputCol), $(dimension), Some($(storageRef))))
  }

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT, AnnotatorType.TOKEN)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.WORD_EMBEDDINGS

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModel(path, spark, getModelIfNotSet.tensorflow, "_bert", BertEmbeddings.tfFile, configProtoBytes = getConfigProtoBytes)
  }

}

trait ReadablePretrainedBertModel extends ParamsAndFeaturesReadable[BertEmbeddings] with HasPretrained[BertEmbeddings] {
  override val defaultModelName: Some[String] = Some("bert_base_cased")

  /** Java compliant-overrides */
  override def pretrained(): BertEmbeddings = super.pretrained()
  override def pretrained(name: String): BertEmbeddings = super.pretrained(name)
  override def pretrained(name: String, lang: String): BertEmbeddings = super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): BertEmbeddings = super.pretrained(name, lang, remoteLoc)
}

trait ReadBertTensorflowModel extends ReadTensorflowModel {
  this:ParamsAndFeaturesReadable[BertEmbeddings] =>

  override val tfFile: String = "bert_tensorflow"

  def readTensorflow(instance: BertEmbeddings, path: String, spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_bert_tf", initAllTables = true)
    instance.setModelIfNotSet(spark, tf)
  }

  addReader(readTensorflow)

  def loadSavedModel(folder: String, spark: SparkSession): BertEmbeddings = {

    val f = new File(folder)
    val savedModel = new File(folder, "saved_model.pb")
    require(f.exists, s"Folder $folder not found")
    require(f.isDirectory, s"File $folder is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $folder"
    )

    val vocab = new File(folder+"/assets", "vocab.txt")
    require(f.exists, s"Folder $folder not found")
    require(f.isDirectory, s"File $folder is not folder")
    require(vocab.exists(), s"Vocabulary file vocab.txt not found in folder $folder")

    val vocabResource = new ExternalResource(vocab.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    val words = ResourceHelper.parseLines(vocabResource).zipWithIndex.toMap

    val wrapper = TensorflowWrapper.read(folder, zipped = false, useBundle = true, tags = Array("serve"), initAllTables = true)

    new BertEmbeddings()
      .setVocabulary(words)
      .setModelIfNotSet(spark, wrapper)
  }
}


object BertEmbeddings extends ReadablePretrainedBertModel with ReadBertTensorflowModel
