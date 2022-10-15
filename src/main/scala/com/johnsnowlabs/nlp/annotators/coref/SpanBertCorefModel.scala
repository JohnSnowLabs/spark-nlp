/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.johnsnowlabs.nlp.annotators.coref

import com.johnsnowlabs.ml.tensorflow.{
  ReadTensorflowModel,
  TensorflowSpanBertCoref,
  TensorflowWrapper,
  WriteTensorflowModel
}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.ModelEngine
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}
import com.johnsnowlabs.nlp.embeddings.HasEmbeddingsProperties
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.slf4j.{Logger, LoggerFactory}

/** A coreference resolution model based on SpanBert
  *
  * A coreference resolution model identifies expressions which refer to the same entity in a
  * text. For example, given a sentence "John told Mary he would like to borrow a book from her."
  * the model will link "he" to "John" and "her" to "Mary".
  *
  * This model is based on SpanBert, which is fine-tuned on the OntoNotes 5.0 data set.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val dependencyParserApproach = SpanBertCorefModel.pretrained()
  *   .setInputCols("sentence", "token")
  *   .setOutputCol("corefs")
  * }}}
  * The default model is `"spanbert_base_coref"`, if no name is provided. For available pretrained
  * models please see the [[https://nlp.johnsnowlabs.com/models Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/annotation/english/coreference-resolution/Coreference_Resolution_SpanBertCorefModel.ipynb Spark NLP Workshop]]
  *
  * '''References:'''
  *   - [[https://github.com/mandarjoshi90/coref]]
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotator._
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val sentence = new SentenceDetector()
  *   .setInputCols("document")
  *   .setOutputCol("sentence")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols("sentence")
  *   .setOutputCol("token")
  *
  * val corefResolution = SpanBertCorefModel.pretrained()
  *   .setInputCols("sentence", "token")
  *   .setOutputCol("corefs")
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   sentence,
  *   tokenizer,
  *   corefResolution
  * ))
  *
  * val data = Seq(
  *   "John told Mary he would like to borrow a book from her."
  * ).toDF("text")
  *
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr(""explode(corefs) AS coref"")
  *   .selectExpr("coref.result as token", "coref.metadata").show(truncate = false)
  * +-----+------------------------------------------------------------------------------------+
  * |token|metadata                                                                            |
  * +-----+------------------------------------------------------------------------------------+
  * |John |{head.sentence -> -1, head -> ROOT, head.begin -> -1, head.end -> -1, sentence -> 0}|
  * |he   |{head.sentence -> 0, head -> John, head.begin -> 0, head.end -> 3, sentence -> 0}   |
  * |Mary |{head.sentence -> -1, head -> ROOT, head.begin -> -1, head.end -> -1, sentence -> 0}|
  * |her  |{head.sentence -> 0, head -> Mary, head.begin -> 10, head.end -> 13, sentence -> 0} |
  * +-----+------------------------------------------------------------------------------------+
  * }}}
  *
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
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
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
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
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.DEPENDENCY

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

  /** Text genre, one of the following values: `bc`: Broadcast conversation, default `bn`:
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

    def getTokensFromSpan(span: ((Int, Int), (Int, Int))): Array[(TokenPiece, Int)] = {
      val sentence1 = span._1._1
      val sentence2 = span._2._1
      val tokenStart = span._1._2
      val tokenEnd = span._2._2
      if (sentence1 == sentence2) {
        tokenizedSentences(sentence1).tokens.slice(tokenStart, tokenEnd + 1).map((_, sentence1))
      } else {
        (tokenizedSentences(sentence1).tokens
          .slice(tokenStart, tokenizedSentences(sentence1).tokens.length - 1)
          .map((_, sentence1))
          ++
            tokenizedSentences(sentence2).tokens.slice(0, tokenEnd + 1).map((_, sentence2)))
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

      val clusterSpans = cluster.map(xy => getTokensFromSpan(xy))
      val clusterHeadSpan = clusterSpans.head
      val clusterHeadSpanText = clusterHeadSpan
        .map(x => (if (x._1.isWordStart) " " else "") + x._1.wordpiece.replaceFirst("##", ""))
        .mkString("")
        .trim
      Array(
        Annotation(
          annotatorType = AnnotatorType.DEPENDENCY,
          begin = clusterHeadSpan.head._1.begin,
          end = clusterHeadSpan.last._1.end,
          result = clusterHeadSpanText,
          metadata = Map(
            "head" -> "ROOT",
            "head.begin" -> "-1",
            "head.end" -> "-1",
            "head.sentence" -> "-1",
            "sentence" -> clusterHeadSpan.head._2.toString))) ++ clusterSpans.tail.map(span => {
        Annotation(
          annotatorType = AnnotatorType.DEPENDENCY,
          begin = span.head._1.begin,
          end = span.last._1.end,
          result = span
            .map(x => (if (x._1.isWordStart) " " else "") + x._1.wordpiece.replaceFirst("##", ""))
            .mkString("")
            .trim,
          metadata = Map(
            "head" -> clusterHeadSpanText,
            "head.begin" -> clusterHeadSpan.head._1.begin.toString,
            "head.end" -> clusterHeadSpan.last._1.end.toString,
            "head.sentence" -> clusterHeadSpan.head._2.toString,
            "sentence" -> span.head._2.toString))
      })
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
  override val defaultModelName: Some[String] = Some("spanbert_base_coref")

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

  def loadSavedModel(modelPath: String, spark: SparkSession): SpanBertCorefModel = {

    val detectedEngine = modelSanityCheck(modelPath)

    val vocabs = loadTextAsset(modelPath, "vocab.txt").zipWithIndex.toMap

    /*Universal parameters for all engines*/
    val annotatorModel = new SpanBertCorefModel()
      .setVocabulary(vocabs)

    detectedEngine match {
      case ModelEngine.tensorflow =>
        val (wrapper, signatures) =
          TensorflowWrapper.read(modelPath, zipped = false, useBundle = true)

        val _signatures = signatures match {
          case Some(s) => s
          case None => throw new Exception("Cannot load signature definitions from model!")
        }

        /** the order of setSignatures is important if we use getSignatures inside
          * setModelIfNotSet
          */
        annotatorModel
          .setSignatures(_signatures)
          .setModelIfNotSet(spark, wrapper)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[SpanBertCorefModel]]. Please refer to that class for the
  * documentation.
  */
object SpanBertCorefModel
    extends ReadablePretrainedSpanBertCorefModel
    with ReadSpanBertCorefTensorflowModel {
  private[SpanBertCorefModel] val logger: Logger = LoggerFactory.getLogger("SpanBertCorefModel")
}
