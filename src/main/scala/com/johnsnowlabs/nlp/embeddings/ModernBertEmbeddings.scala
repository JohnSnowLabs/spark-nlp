/*
 * Copyright 2017-2025 John Snow Labs
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

package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.ml.ai.ModernBert
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.openvino.{OpenvinoWrapper, ReadOpenvinoModel, WriteOpenvinoModel}
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadJsonStringAsset,
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ModelArch, ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.storage.HasStorageRef
import com.johnsnowlabs.util.FileHelper
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.{Logger, LoggerFactory}
import org.json4s._
import org.json4s.jackson.JsonMethods._

import java.nio.file.Files
import java.util.UUID

/** Token-level embeddings using ModernBERT. ModernBERT is a state-of-the-art encoder model
  * designed for improved efficiency and performance compared to traditional BERT models.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val embeddings = ModernBertEmbeddings.pretrained()
  *   .setInputCols("token", "document")
  *   .setOutputCol("modernbert_embeddings")
  * }}}
  * The default model is `"modernbert-base"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Embeddings Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/dl-ner/ner_bert.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/ModernBertEmbeddingsTestSpec.scala ModernBertEmbeddingsTestSpec]].
  * To see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]].
  *
  * '''Sources''' :
  *
  * [[https://arxiv.org/abs/2412.13663 Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Applications]]
  *
  * [[https://huggingface.co/answerdotai/ModernBERT-base]]
  *
  * ''' Paper abstract '''
  *
  * ''We introduce ModernBERT, a modernized bidirectional encoder model that is 8x faster, uses 5x
  * less memory, and achieves better downstream performance than traditional BERT models.
  * ModernBERT incorporates modern improvements including Flash Attention, unpadding, and GeGLU
  * activation functions. The model supports sequence lengths up to 8192 tokens while maintaining
  * competitive performance on tasks requiring long context understanding.''
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.embeddings.ModernBertEmbeddings
  * import com.johnsnowlabs.nlp.EmbeddingsFinisher
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols("document")
  *   .setOutputCol("token")
  *
  * val embeddings = ModernBertEmbeddings.pretrained("modernbert-base", "en")
  *   .setInputCols("token", "document")
  *   .setOutputCol("modernbert_embeddings")
  *
  * val embeddingsFinisher = new EmbeddingsFinisher()
  *   .setInputCols("modernbert_embeddings")
  *   .setOutputCols("finished_embeddings")
  *   .setOutputAsVector(true)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   tokenizer,
  *   embeddings,
  *   embeddingsFinisher
  * ))
  *
  * val data = Seq("This is a sentence.").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  * +--------------------------------------------------------------------------------+
  * |                                                                          result|
  * +--------------------------------------------------------------------------------+
  * |[-2.3497989177703857,0.480538547039032,-0.3238905668258667,-1.612930893898010...|
  * |[-2.1357314586639404,0.32984697818756104,-0.6032363176345825,-1.6791689395904...|
  * |[-1.8244884014129639,-0.27088963985443115,-1.059438943862915,-0.9817547798156...|
  * |[-1.1648050546646118,-0.4725411534309387,-0.5938255786895752,-1.5780693292617...|
  * |[-0.9125322699546814,0.4563939869403839,-0.3975459933280945,-1.81611204147338...|
  * +--------------------------------------------------------------------------------+
  * }}}
  *
  * @see
  *   [[ModernBertSentenceEmbeddings]] for sentence-level embeddings
  * @see
  *   [[com.johnsnowlabs.nlp.annotators.classifier.dl.ModernBertForTokenClassification ModernBertForTokenClassification]]
  *   For ModernBertEmbeddings with a token classification layer on top
  * @see
  *   [[https://sparknlp.org/docs/en/annotators Annotators Main Page]] for a list of transformer
  *   based embeddings
  * @param uid
  *   required uid for storing annotator to disk
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
class ModernBertEmbeddings(override val uid: String)
    extends AnnotatorModel[ModernBertEmbeddings]
    with HasBatchedAnnotate[ModernBertEmbeddings]
    with WriteTensorflowModel
    with WriteOnnxModel
    with WriteOpenvinoModel
    with HasEmbeddingsProperties
    with HasStorageRef
    with HasCaseSensitiveProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("MODERNBERT_EMBEDDINGS"))

  /** Input Annotator Types: DOCUMENT, TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] =
    Array(AnnotatorType.DOCUMENT, AnnotatorType.TOKEN)

  /** Output Annotator Types: WORD_EMBEDDINGS
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.WORD_EMBEDDINGS

  /** @group setParam */
  def sentenceStartTokenId: Int = {
    $$(vocabulary)("[CLS]")
  }

  /** @group setParam */
  def sentenceEndTokenId: Int = {
    $$(vocabulary)("[SEP]")
  }

  /** Vocabulary used to encode the words to ids with BPE tokenizer
    *
    * @group param
    */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary").setProtected()

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  /** Merges used by the BPE tokenizer
    *
    * @group param
    */
  val merges: MapFeature[(String, String), Int] = new MapFeature(this, "merges").setProtected()

  /** @group setParam */
  def setMerges(value: Map[(String, String), Int]): this.type = set(merges, value)

  /** Added tokens for the BPE tokenizer
    *
    * @group param
    */
  val addedTokens: MapFeature[String, Int] = new MapFeature(this, "addedTokens").setProtected()

  /** @group setParam */
  def setAddedTokens(value: Map[String, Int]): this.type = set(addedTokens, value)

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
  def setConfigProtoBytes(bytes: Array[Int]): ModernBertEmbeddings.this.type =
    set(this.configProtoBytes, bytes)

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** Max sentence length to process (Default: `8192`)
    *
    * @group param
    */
  val maxSentenceLength =
    new IntParam(this, "maxSentenceLength", "Max sentence length to process")

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(
      value <= 8192,
      "ModernBERT models do not support sequences longer than 8192 because of trainable positional embeddings.")
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
  val signatures =
    new MapFeature[String, String](model = this, name = "signatures").setProtected()

  /** @group setParam */
  def setSignatures(value: Map[String, String]): this.type = {
    set(signatures, value)
    this
  }

  /** @group getParam */
  def getSignatures: Option[Map[String, String]] = get(this.signatures)

  private var _model: Option[Broadcast[ModernBert]] = None

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrapper: Option[OnnxWrapper],
      openvinoWrapper: Option[OpenvinoWrapper]): ModernBertEmbeddings = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new ModernBert(
            tensorflowWrapper,
            onnxWrapper,
            openvinoWrapper,
            sentenceStartTokenId,
            sentenceEndTokenId,
            $$(merges),
            $$(vocabulary),
            $$(addedTokens),
            configProtoBytes = getConfigProtoBytes,
            signatures = getSignatures,
            modelArch = ModelArch.wordEmbeddings)))
    }

    this
  }

  def getModelIfNotSet: ModernBert = _model.get.value

  /** Set Embeddings dimensions for the ModernBERT model Only possible to set this when the first
    * time is saved dimension is not changeable, it comes from ModernBERT config file
    *
    * @group setParam
    */
  override def setDimension(value: Int): this.type = {
    set(this.dimension, value)
  }

  /** Whether to lowercase tokens or not
    *
    * @group setParam
    */
  override def setCaseSensitive(value: Boolean): this.type = {
    set(this.caseSensitive, value)
  }

  setDefault(dimension -> 768, batchSize -> 8, maxSentenceLength -> 8192, caseSensitive -> false)

  /** takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param batchedAnnotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {

    // Unpack annotations and zip each sentence to the index or the row it belongs to
    val sentencesWithRow = batchedAnnotations.zipWithIndex
      .flatMap { case (annotations, i) =>
        TokenizedWithSentence.unpack(annotations).toArray.map(x => (x, i))
      }

    // Tokenize sentences
    val tokenizedSentences = getModelIfNotSet.tokenizeWithAlignment(
      sentencesWithRow.map(_._1),
      $(maxSentenceLength),
      $(caseSensitive))

    // Process all sentences
    val sentenceWordEmbeddings = getModelIfNotSet.predict(
      tokenizedSentences,
      sentencesWithRow.map(_._1),
      $(batchSize),
      $(maxSentenceLength),
      $(caseSensitive))

    // Group resulting annotations by rows. If there are not sentences in a given row, return empty sequence
    batchedAnnotations.indices.map(rowIndex => {
      val rowEmbeddings = sentenceWordEmbeddings
        // zip each annotation with its corresponding row index
        .zip(sentencesWithRow)
        // select the sentences belonging to the current row
        .filter(_._2._2 == rowIndex)
        // leave the annotation only
        .map(_._1)

      if (rowEmbeddings.nonEmpty)
        WordpieceEmbeddingsSentence.pack(rowEmbeddings)
      else
        Seq.empty[Annotation]
    })

  }

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    dataset.withColumn(
      getOutputCol,
      wrapEmbeddingsMetadata(dataset.col(getOutputCol), $(dimension), Some($(storageRef))))
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    val suffix = "_modernbert"

    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          suffix,
          ModernBertEmbeddings.tfFile,
          configProtoBytes = getConfigProtoBytes)
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          suffix,
          ModernBertEmbeddings.onnxFile)
      case Openvino.name =>
        writeOpenvinoModel(
          path,
          spark,
          getModelIfNotSet.openvinoWrapper.get,
          suffix,
          ModernBertEmbeddings.openvinoFile)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

}

trait ReadablePretrainedModernBertModel
    extends ParamsAndFeaturesReadable[ModernBertEmbeddings]
    with HasPretrained[ModernBertEmbeddings] {
  override val defaultModelName: Some[String] = Some("modernbert-base")

  /** Java compliant-overrides */
  override def pretrained(): ModernBertEmbeddings = super.pretrained()

  override def pretrained(name: String): ModernBertEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): ModernBertEmbeddings =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): ModernBertEmbeddings =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadModernBertDLModel
    extends ReadTensorflowModel
    with ReadOnnxModel
    with ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[ModernBertEmbeddings] =>

  override val tfFile: String = "modernbert_tensorflow"
  override val onnxFile: String = "modernbert_onnx"
  override val openvinoFile: String = "modernbert_openvino"

  def readModel(instance: ModernBertEmbeddings, path: String, spark: SparkSession): Unit = {

    instance.getEngine match {
      case TensorFlow.name =>
        val tfWrapper = readTensorflowModel(path, spark, "_modernbert_tf", initAllTables = false)
        instance.setModelIfNotSet(spark, Some(tfWrapper), None, None)

      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(path, spark, "_modernbert_onnx", zipped = true, useBundle = false, None)
        instance.setModelIfNotSet(spark, None, Some(onnxWrapper), None)

      case Openvino.name =>
        val openvinoWrapper = readOpenvinoModel(path, spark, "_modernbert_ov")
        instance.setModelIfNotSet(spark, None, None, Some(openvinoWrapper))
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(
      modelPath: String,
      spark: SparkSession,
      useOpenvino: Boolean = false): ModernBertEmbeddings = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    implicit val formats: DefaultFormats.type = DefaultFormats // for json4s

    // Check if tokenizer.json exists
    val tokenizerPath = s"$localModelPath/tokenizer.json"
    val tokenizerExists = new java.io.File(tokenizerPath).exists()
    val (vocabs, addedTokens, bytePairs) = if (tokenizerExists) {
      val tokenizerConfig: JValue = parse(loadJsonStringAsset(localModelPath, "tokenizer.json"))
      // extract vocab from tokenizer.json ( model -> vocab)
      var vocabs: Map[String, Int] =
        (tokenizerConfig \ "model" \ "vocab").extract[Map[String, Int]]

      // extract merges from tokenizer.json ( model -> merges)
      val bytePairs = (tokenizerConfig \ "model" \ "merges")
        .extract[List[String]]
        .map(_.split(" "))
        .filter(w => w.length == 2)
        .map { case Array(c1, c2) => (c1, c2) }
        .zipWithIndex
        .toMap

      // extract added_tokens from tokenizer.json (added_tokens)
      val addedTokens = (tokenizerConfig \ "added_tokens")
        .extract[List[Map[String, Any]]]
        .map { token =>
          val id = token("id").asInstanceOf[BigInt].intValue()
          val content = token("content").asInstanceOf[String]
          (content, id)
        }
        .toMap
      // Add the added tokens to the vocabulary
      addedTokens.foreach { case (content, id) =>
        vocabs += (content -> id)
      }
      (vocabs, addedTokens, bytePairs)
    } else {
      // Fallback to old format
      val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap
      val addedTokens = Map.empty[String, Int]
      val bytePairs = Map.empty[(String, String), Int]
      (vocabs, addedTokens, bytePairs)
    }

    /*Universal parameters for all engines*/
    val annotatorModel = new ModernBertEmbeddings()
      .setVocabulary(vocabs)
      .setMerges(bytePairs)
      .setAddedTokens(addedTokens)

    val modelEngine =
      if (useOpenvino)
        Openvino.name
      else
        detectedEngine
    annotatorModel.set(annotatorModel.engine, modelEngine)

    modelEngine match {
      case Openvino.name =>
        val ovWrapper: OpenvinoWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine)
        annotatorModel
          .setModelIfNotSet(spark, None, None, Some(ovWrapper))

      case TensorFlow.name =>
        val (tfWrapper, signatures) =
          TensorflowWrapper.read(localModelPath, zipped = false, useBundle = true)

        val _signatures = signatures match {
          case Some(s) => s
          case None => throw new Exception("Cannot load signature definitions from model!")
        }

        /** the order of setSignatures is important if we use getSignatures inside
          * setModelIfNotSet
          */
        annotatorModel
          .setSignatures(_signatures)
          .setModelIfNotSet(spark, Some(tfWrapper), None, None)

      case ONNX.name =>
        val onnxWrapper =
          OnnxWrapper.read(spark, localModelPath, zipped = false, useBundle = true)
        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrapper), None)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }

}

/** This is the companion object of [[ModernBertEmbeddings]]. Please refer to that class for the
  * documentation.
  */
object ModernBertEmbeddings extends ReadablePretrainedModernBertModel with ReadModernBertDLModel {
  private[ModernBertEmbeddings] val logger: Logger =
    LoggerFactory.getLogger("ModernBertEmbeddings")
}
