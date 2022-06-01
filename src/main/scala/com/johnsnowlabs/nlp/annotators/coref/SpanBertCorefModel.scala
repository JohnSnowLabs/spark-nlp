package com.johnsnowlabs.nlp.annotators.coref

import com.johnsnowlabs.ml.tensorflow.{
  ReadTensorflowModel,
  TensorflowSpanBertCoref,
  TensorflowWrapper,
  WriteTensorflowModel
}
import com.johnsnowlabs.nlp.annotators.common.{
  IndexedToken,
  Sentence,
  TokenPiece,
  TokenizedSentence,
  TokenizedWithSentence,
  WordpieceTokenizedSentence
}
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}
import com.johnsnowlabs.nlp.{
  Annotation,
  AnnotatorModel,
  AnnotatorType,
  HasCaseSensitiveProperties,
  HasPretrained,
  HasSimpleAnnotate,
  ParamsAndFeaturesReadable
}
import com.johnsnowlabs.nlp.embeddings.HasEmbeddingsProperties
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.slf4j.{Logger, LoggerFactory}

import java.io.File

class SpanBertCorefModel(override val uid: String)
    extends AnnotatorModel[SpanBertCorefModel]
    with HasSimpleAnnotate[SpanBertCorefModel]
    with WriteTensorflowModel
    with HasEmbeddingsProperties
    with HasStorageRef
    with HasCaseSensitiveProperties {

  def this() = this(Identifiable.randomUID("SPANBERTCOREFMODEL"))

  override val inputAnnotatorTypes: Array[String] =
    Array(AnnotatorType.DOCUMENT, AnnotatorType.TOKEN)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.CATEGORY

  def sentenceStartTokenId: Int = {
    $$(vocabulary)("[CLS]")
  }

  /** @group setParam */
  def sentenceEndTokenId: Int = {
    $$(vocabulary)("[SEP]")
  }

  /** Vocabulary used to encode the words to ids with WordPieceEncoder
    *
    * @group param
    */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary")

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * `config_proto.SerializeToString()`
    *
    * @group param
    */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): SpanBertCorefModel.this.type =
    set(this.configProtoBytes, bytes)

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** Max sentence length to process (Default: `128`)
    *
    * @group param
    */
  val maxSentenceLength =
    new IntParam(this, "maxSentenceLength", "Max sentence length to process")

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(
      value <= 512,
      "BERT models do not support sequences longer than 512 because of trainable positional embeddings.")
    require(value >= 1, "The maxSentenceLength must be at least 1")
    set(maxSentenceLength, value)
    this
  }

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  /** It contains TF model signatures for the laded saved model
    *
    * @group param
    */
  val signatures = new MapFeature[String, String](model = this, name = "signatures")

  /** @group setParam */
  def setSignatures(value: Map[String, String]): this.type = {
    if (get(signatures).isEmpty)
      set(signatures, value)
    this
  }

  /** @group getParam */
  def getSignatures: Option[Map[String, String]] = get(this.signatures)

  val _textGenres: Array[String] = Array(
    "bc", // Broadcast conversation, default
    "bn", // Broadcast news
    "mz", //
    "nw", // News wire
    "pt", // Pivot text: Old Testament and New Testament text
    "tc", // Telephone conversation
    "wb" // Web data
  )

  /** Text genre, one of the following values: `bc`: Broadcast conversation, default `bn:
    * Broadcast news `nw`: News wire `pt`: Pivot text: Old Testament and New Testament text `tc`:
    * Telephone conversation `wb`: Web data
    *
    * @group param
    */
  val textGenre =
    new Param[String](
      this,
      "textGenre",
      s"Text genre, one of %s. Default is 'bc'.".format(
        _textGenres.map("\"" + _ + "\"").mkString(", ")))

  /** @group setParam */
  def setTextGenre(value: String): this.type = {
    require(
      Array().contains(value.toLowerCase),
      s"Text text genre must be one of %s".format(
        _textGenres.map("\"" + _ + "\"").mkString(", ")))
    set(textGenre, value.toLowerCase)
    this
  }

  /** @group getParam */
  def getTextGenre: String = $(textGenre)

  /** Max segment length to process (Read-only, depends on model)
    *
    * @group param
    */
  val maxSegmentLength = new IntParam(this, "maxSegmentLength", "Maximum segment length")

  /** @group setParam */
  def setMaxSegmentLength(value: Int): this.type = {
    if (get(maxSegmentLength).isEmpty)
      set(maxSegmentLength, value)
    this
  }

  /** @group getParam */
  def getMaxSegmentLength: Int = $(maxSegmentLength)

  private var _model: Option[Broadcast[TensorflowSpanBertCoref]] = None

  setDefault(
    maxSentenceLength -> 512,
    caseSensitive -> true,
    textGenre -> _textGenres(0)
//    maxSegmentLength -> 384,
  )

  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: TensorflowWrapper): SpanBertCorefModel = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new TensorflowSpanBertCoref(
            tensorflowWrapper,
            sentenceStartTokenId,
            sentenceEndTokenId,
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures)))
    }

    this
  }

  def getModelIfNotSet: TensorflowSpanBertCoref = _model.get.value

  def tokenizeSentence(tokens: Seq[TokenizedSentence]): Seq[WordpieceTokenizedSentence] = {
    val basicTokenizer = new BasicTokenizer($(caseSensitive))
    val encoder = new WordpieceEncoder($$(vocabulary))

    tokens.map { tokenIndex =>
      // filter empty and only whitespace tokens
      val bertTokens =
        tokenIndex.indexedTokens.filter(x => x.token.nonEmpty && !x.token.equals(" ")).map {
          token =>
            val content = if ($(caseSensitive)) token.token else token.token.toLowerCase()
            val sentenceBegin = token.begin
            val sentenceEnd = token.end
            val sentenceIndex = tokenIndex.sentenceIndex
            val result = basicTokenizer.tokenize(
              Sentence(content, sentenceBegin, sentenceEnd, sentenceIndex))
            if (result.nonEmpty) result.head else IndexedToken("")
        }
      val wordPieceTokens =
        bertTokens.flatMap(token => encoder.encode(token)).take($(maxSentenceLength) - 2)
      WordpieceTokenizedSentence(wordPieceTokens)
    }
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val sentencesWithRow = TokenizedWithSentence.unpack(annotations)
    val tokenizedSentences = tokenizeSentence(sentencesWithRow).toArray
    val inputIds = tokenizedSentences.map(x => x.tokens.map(_.pieceId))

    if (inputIds.map(x => x.length).sum < 2) {
      return Seq()
    }

    val predictedClusters = getModelIfNotSet.predict(
      inputIds = inputIds,
      genre = _textGenres.indexOf($(textGenre)),
      maxSegmentLength = $(maxSegmentLength))

    def getTokensFromSpan(span: ((Int, Int), (Int, Int))): Array[TokenPiece] = {
      val sentence1 = span._1._1
      val sentence2 = span._2._1
      val tokenStart = span._1._2
      val tokenEnd = span._2._2
      if (sentence1 == sentence2) {
        tokenizedSentences(sentence1).tokens.slice(tokenStart, tokenEnd + 1)
      } else {
        (tokenizedSentences(sentence1).tokens
          .slice(tokenStart, tokenizedSentences(sentence1).tokens.length - 1)
          ++
            tokenizedSentences(sentence2).tokens.slice(0, tokenEnd + 1))
      }
    }

//    predictedClusters.zipWithIndex.foreach{
//      case (cluster, i) =>
//        print(s"Cluster #$i\n")
//        print(s"\t%s\n".format(
//          cluster.map(
//            xy =>
//              getTokensFromSpan(xy).map(x => (if (x.isWordStart) " " else "") + x.wordpiece.replaceFirst("##", "") ).mkString("").trim,
//            ).mkString(", ")))
//    }
    predictedClusters.flatMap(cluster => {
      val spans = cluster.map(xy => getTokensFromSpan(xy))

      spans.zipWithIndex.flatMap { case (span1, span1Id) =>
        spans.zipWithIndex
          .filter(_._2 > span1Id)
          .map(_._1)
          .map { span2 =>
            val minSpanBegin = scala.math.min(span1.head.begin, span2.head.begin)
            val maxSpanEnd = scala.math.min(span1.last.end, span2.last.end)

            new Annotation(
              annotatorType = AnnotatorType.CATEGORY,
              begin = minSpanBegin,
              end = maxSpanEnd,
              metadata = Map(
                "entity1" -> "COREFSPAN",
                "entity2" -> "COREFSPAN",
                "entity1_begin" -> span1.head.begin.toString,
                "entity1_end" -> span2.head.begin.toString,
                "entity2_begin" -> span1.head.begin.toString,
                "entity2_end" -> span2.head.begin.toString,
                "chunk1" -> span1
                  .map(x => (if (x.isWordStart) " " else "") + x.wordpiece.replaceFirst("##", ""))
                  .mkString("")
                  .trim,
                "chunk2" -> span2
                  .map(x => (if (x.isWordStart) " " else "") + x.wordpiece.replaceFirst("##", ""))
                  .mkString("")
                  .trim),
              result = "COREF")
          }
      }
    })
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    writeTensorflowModelV2(
      path,
      spark,
      getModelIfNotSet.tensorflowWrapper,
      "_bert",
      SpanBertCorefModel.tfFile,
      configProtoBytes = getConfigProtoBytes)
  }
}

trait ReadablePretrainedSpanBertCorefModel
    extends ParamsAndFeaturesReadable[SpanBertCorefModel]
    with HasPretrained[SpanBertCorefModel] {
  override val defaultModelName: Some[String] = Some("")

  /** Java compliant-overrides */
  override def pretrained(): SpanBertCorefModel = super.pretrained()

  override def pretrained(name: String): SpanBertCorefModel = super.pretrained(name)

  override def pretrained(name: String, lang: String): SpanBertCorefModel =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): SpanBertCorefModel =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadSpanBertCorefTensorflowModel extends ReadTensorflowModel {
  this: ParamsAndFeaturesReadable[SpanBertCorefModel] =>

  override val tfFile: String = "spanbert_tensorflow"

  def readTensorflow(instance: SpanBertCorefModel, path: String, spark: SparkSession): Unit = {

    val tf = readTensorflowModel(path, spark, "_bert_tf", initAllTables = false)
    instance.setModelIfNotSet(spark, tf)
  }

  addReader(readTensorflow)

  def loadSavedModel(tfModelPath: String, spark: SparkSession): SpanBertCorefModel = {

    val f = new File(tfModelPath)
    val savedModel = new File(tfModelPath, "saved_model.pb")

    require(f.exists, s"Folder $tfModelPath not found")
    require(f.isDirectory, s"File $tfModelPath is not folder")
    require(
      savedModel.exists(),
      s"savedModel file saved_model.pb not found in folder $tfModelPath")

    val vocab = new File(tfModelPath + "/assets", "vocab.txt")

    require(f.exists, s"Folder $tfModelPath not found")
    require(f.isDirectory, s"File $tfModelPath is not folder")
    require(vocab.exists(), s"Vocabulary file vocab.txt not found in folder $tfModelPath")

    val vocabResource =
      new ExternalResource(vocab.getAbsolutePath, ReadAs.TEXT, Map("format" -> "text"))
    val words = ResourceHelper.parseLines(vocabResource).zipWithIndex.toMap

    val (wrapper, signatures) =
      TensorflowWrapper.read(tfModelPath, zipped = false, useBundle = true)

    val _signatures = signatures match {
      case Some(s) => s
      case None => throw new Exception("Cannot load signature definitions from model!")
    }

    /** the order of setSignatures is important if we use getSignatures inside setModelIfNotSet */
    new SpanBertCorefModel()
      .setVocabulary(words)
      .setSignatures(_signatures)
      .setModelIfNotSet(spark, wrapper)
  }
}

/** This is the companion object of [[BertEmbeddings]]. Please refer to that class for the
  * documentation.
  */
object SpanBertCorefModel
    extends ReadablePretrainedSpanBertCorefModel
    with ReadSpanBertCorefTensorflowModel {
  private[SpanBertCorefModel] val logger: Logger = LoggerFactory.getLogger("SpanBertCorefModel")
}
