/*
 * Copyright 2017-2024 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.ml.ai.CrossEncoderClassification
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.openvino.{OpenvinoWrapper, ReadOpenvinoModel, WriteOpenvinoModel}
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadJsonStringAsset,
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ONNX, Openvino, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{IntArrayParam, IntParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.json4s._
import org.json4s.jackson.JsonMethods._

import java.io.File
import scala.io.Source

/** CrossEncoderForSequenceClassification brings cross-encoder scoring (as in
  * `sentence-transformers` `CrossEncoder`) into Spark NLP as a first-class annotator.
  *
  * It takes two row-aligned document columns, jointly encodes each row's pair as a single
  * sequence `[CLS] text_a [SEP] text_b [SEP]`, runs one forward pass through a BERT-family
  * transformer with a classification/regression head, and writes one score per row to a single
  * output column. Row `i` of the first column and row `i` of the second column produce row `i` of
  * the output — there is no cross-row interaction. Any 1-query-vs-N-candidates reranking use case
  * is a `crossJoin`/`explode` the user performs upstream, not something the annotator does
  * internally.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val crossEncoder = CrossEncoderForSequenceClassification.pretrained()
  *   .setInputCols("document1", "document2")
  *   .setOutputCol("score")
  * }}}
  * The default model is `"cross_encoder_ms_marco_minilm_l6_v2"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?task=Text+Classification Models Hub]].
  *
  * Models from the HuggingFace 🤗 Transformers library are also compatible with Spark NLP 🚀. To
  * see which models are compatible and how to import them see
  * [[https://github.com/JohnSnowLabs/spark-nlp/discussions/5669]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotator._
  * import org.apache.spark.ml.Pipeline
  *
  * val document = new MultiDocumentAssembler()
  *   .setInputCols("query", "passage")
  *   .setOutputCols("document1", "document2")
  *
  * val crossEncoder = CrossEncoderForSequenceClassification.pretrained()
  *   .setInputCols("document1", "document2")
  *   .setOutputCol("score")
  *
  * val pipeline = new Pipeline().setStages(Array(document, crossEncoder))
  *
  * val data = Seq(
  *   ("How many people live in Berlin?", "Berlin is well known for its museums."))
  *   .toDF("query", "passage")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.select("score.result").show(false)
  * }}}
  *
  * @see
  *   [[BertForSequenceClassification]] for single-sequence classification
  * @see
  *   [[https://sparknlp.org/docs/en/annotators Annotators Main Page]] for a list of transformer
  *   based classifiers
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
class CrossEncoderForSequenceClassification(override val uid: String)
    extends AnnotatorModel[CrossEncoderForSequenceClassification]
    with HasBatchedAnnotate[CrossEncoderForSequenceClassification]
    with WriteTensorflowModel
    with WriteOnnxModel
    with WriteOpenvinoModel
    with HasCaseSensitiveProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("CrossEncoderForSequenceClassification"))

  /** Input Annotator Types: DOCUMENT, DOCUMENT
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] =
    Array(AnnotatorType.DOCUMENT, AnnotatorType.DOCUMENT)

  /** Output Annotator Types: CATEGORY
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.CATEGORY

  def sentenceStartTokenId: Int = $$(vocabulary)("[CLS]")

  def sentenceEndTokenId: Int = $$(vocabulary)("[SEP]")

  /** Vocabulary used to encode the words to ids with WordPieceEncoder
    *
    * @group param
    */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary").setProtected()

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  /** Labels used to decode predicted IDs back to string tags. Empty for regression heads (the
    * common reranking case), in which case the raw score is returned instead of a label.
    *
    * @group param
    */
  val labels: MapFeature[String, Int] = new MapFeature(this, "labels").setProtected()

  /** @group setParam */
  def setLabels(value: Map[String, Int]): this.type = set(labels, value)

  /** Returns labels used to train this model */
  def getClasses: Array[String] = $$(labels).keys.toArray

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
  def setConfigProtoBytes(bytes: Array[Int]): this.type = set(this.configProtoBytes, bytes)

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** The model's hard ceiling for the combined sequence length, read from the model config
    * (`max_position_embeddings`). `maxSentenceLength` defaults to and is capped at this value.
    *
    * @group param
    */
  val modelMaxLength =
    new IntParam(
      this,
      "modelMaxLength",
      "The model's max sequence length ceiling from its config")

  /** @group setParam */
  def setModelMaxLength(value: Int): this.type = set(modelMaxLength, value)

  /** @group getParam */
  def getModelMaxLength: Int = $(modelMaxLength)

  /** Max sequence length to process. Shared across both texts combined (not per text). Defaults
    * to and is hard-capped at the model's `max_position_embeddings`.
    *
    * @group param
    */
  val maxSentenceLength =
    new IntParam(this, "maxSentenceLength", "Max sentence length to process")

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(value >= 1, "The maxSentenceLength must be at least 1")
    val ceiling = get(modelMaxLength).getOrElse(CrossEncoderForSequenceClassification.DefaultCap)
    require(
      value <= ceiling,
      s"CrossEncoder models do not support sequences longer than $ceiling because of trainable " +
        s"positional embeddings. The combined length of both texts (plus special tokens) is capped there.")
    set(maxSentenceLength, value)
  }

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  /** The activation function applied to the model logits to obtain the final score. One of
    * `"sigmoid"`, `"softmax"` or `"identity"`. Matches `sentence-transformers`'
    * `num_labels`/`activation_fn`: regression + `sigmoid` is the default reranking case, raw
    * logits (`identity`) otherwise. (Default: `"sigmoid"`)
    *
    * @group param
    */
  val activation: Param[String] = new Param[String](
    this,
    "activation",
    "Activation applied to the logits: sigmoid, softmax or identity")

  /** @group setParam */
  def setActivation(value: String): this.type = {
    val normalized = value.toLowerCase
    require(
      CrossEncoderForSequenceClassification.SupportedActivations.contains(normalized),
      s"Unsupported activation '$value'. " +
        s"Supported: ${CrossEncoderForSequenceClassification.SupportedActivations.mkString(", ")}")
    set(activation, normalized)
  }

  /** @group getParam */
  def getActivation: String = $(activation)

  /** How to truncate a pair when the combined length exceeds `maxSentenceLength`. One of
    * `"longest_first"` (HuggingFace default: drop tokens from whichever text is longer) or
    * `"query_first"` (keep the first text intact, truncate the second). (Default:
    * `"longest_first"`)
    *
    * @group param
    */
  val truncationStrategy: Param[String] = new Param[String](
    this,
    "truncationStrategy",
    "Pair truncation strategy: longest_first or query_first")

  /** @group setParam */
  def setTruncationStrategy(value: String): this.type = {
    val normalized = value.toLowerCase
    require(
      CrossEncoderForSequenceClassification.SupportedTruncations.contains(normalized),
      s"Unsupported truncation strategy '$value'. " +
        s"Supported: ${CrossEncoderForSequenceClassification.SupportedTruncations.mkString(", ")}")
    set(truncationStrategy, normalized)
  }

  /** @group getParam */
  def getTruncationStrategy: String = $(truncationStrategy)

  /** It contains TF model signatures for the loaded saved model
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

  private var _model: Option[Broadcast[CrossEncoderClassification]] = None

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrapper: Option[OnnxWrapper],
      openvinoWrapper: Option[OpenvinoWrapper]): CrossEncoderForSequenceClassification = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new CrossEncoderClassification(
            tensorflowWrapper,
            onnxWrapper,
            openvinoWrapper,
            sentenceStartTokenId,
            sentenceEndTokenId,
            configProtoBytes = getConfigProtoBytes,
            tags = $$(labels),
            signatures = getSignatures,
            vocabulary = $$(vocabulary))))
    }

    this
  }

  /** @group getParam */
  def getModelIfNotSet: CrossEncoderClassification = _model.get.value

  /** Whether to lowercase tokens or not (Default: `true`).
    *
    * @group setParam
    */
  override def setCaseSensitive(value: Boolean): this.type = set(this.caseSensitive, value)

  setDefault(
    batchSize -> 8,
    maxSentenceLength -> 512,
    modelMaxLength -> CrossEncoderForSequenceClassification.DefaultCap,
    caseSensitive -> false,
    activation -> ActivationFunction.sigmoid,
    truncationStrategy -> CrossEncoderClassification.LongestFirst)

  /** Takes a batch of row-aligned document pairs and produces one CATEGORY score per row.
    *
    * @param batchedAnnotations
    *   Annotations that correspond to inputAnnotationCols (document1, document2) generated by
    *   previous annotators
    * @return
    *   exactly one score Annotation per input row, in input order
    */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {

    // Input columns are concatenated in inputCols order, so the first DOCUMENT belongs to
    // document1 and the second to document2 (row-aligned, one document per column per row).
    val indexedPairs = batchedAnnotations.zipWithIndex.map { case (annotations, i) =>
      val documents = annotations.filter(_.annotatorType == AnnotatorType.DOCUMENT)
      val pair = if (documents.length >= 2) Some((documents.head, documents(1))) else None
      (i, pair)
    }

    val presentPairs = indexedPairs.collect { case (i, Some(pair)) => (i, pair) }

    if (presentPairs.isEmpty) return batchedAnnotations.map(_ => Seq.empty[Annotation])

    val scores = getModelIfNotSet.predictScore(
      presentPairs.map(_._2),
      $(batchSize),
      $(maxSentenceLength),
      $(caseSensitive),
      $(activation),
      $(truncationStrategy))

    val scoreByIndex = presentPairs.map(_._1).zip(scores).toMap

    batchedAnnotations.indices.map { i =>
      scoreByIndex.get(i).map(Seq(_)).getOrElse(Seq.empty[Annotation])
    }
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    val suffix = "_cross_encoder_classification"

    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          suffix,
          CrossEncoderForSequenceClassification.tfFile,
          configProtoBytes = getConfigProtoBytes)
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          suffix,
          CrossEncoderForSequenceClassification.onnxFile)
      case Openvino.name =>
        writeOpenvinoModel(
          path,
          spark,
          getModelIfNotSet.openvinoWrapper.get,
          "openvino_model.xml",
          CrossEncoderForSequenceClassification.openvinoFile)
    }
  }

}

trait ReadablePretrainedCrossEncoderForSequenceModel
    extends ParamsAndFeaturesReadable[CrossEncoderForSequenceClassification]
    with HasPretrained[CrossEncoderForSequenceClassification] {
  override val defaultModelName: Some[String] = Some("cross_encoder_ms_marco_minilm_l6_v2")

  /** Java compliant-overrides */
  override def pretrained(): CrossEncoderForSequenceClassification = super.pretrained()

  override def pretrained(name: String): CrossEncoderForSequenceClassification =
    super.pretrained(name)

  override def pretrained(name: String, lang: String): CrossEncoderForSequenceClassification =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): CrossEncoderForSequenceClassification =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadCrossEncoderForSequenceDLModel
    extends ReadTensorflowModel
    with ReadOnnxModel
    with ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[CrossEncoderForSequenceClassification] =>

  override val tfFile: String = "cross_encoder_classification_tensorflow"
  override val onnxFile: String = "cross_encoder_classification_onnx"
  override val openvinoFile: String = "cross_encoder_classification_openvino"

  def readModel(
      instance: CrossEncoderForSequenceClassification,
      path: String,
      spark: SparkSession): Unit = {

    instance.getEngine match {
      case TensorFlow.name =>
        val tensorFlow =
          readTensorflowModel(
            path,
            spark,
            "_cross_encoder_classification_tf",
            initAllTables = false)
        instance.setModelIfNotSet(spark, Some(tensorFlow), None, None)
      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(path, spark, "cross_encoder_classification_onnx")
        instance.setModelIfNotSet(spark, None, Some(onnxWrapper), None)
      case Openvino.name =>
        val openvinoWrapper = readOpenvinoModel(path, spark, "cross_encoder_classification_ov")
        instance.setModelIfNotSet(spark, None, None, Some(openvinoWrapper))
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(
      modelPath: String,
      spark: SparkSession): CrossEncoderForSequenceClassification = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap

    // Read the model config to derive the max sequence length ceiling, the (optional) label set,
    // and the sbert default activation. The config lives at the model root in the standard
    // sequence-classifier export, but we also accept it under assets/ for robustness.
    implicit val formats: DefaultFormats.type = DefaultFormats
    val modelConfig: Option[JValue] =
      CrossEncoderForSequenceClassification.readConfigJson(localModelPath)

    val modelMaxLength: Int =
      modelConfig
        .flatMap(c => (c \ "max_position_embeddings").extractOpt[Int])
        .getOrElse(CrossEncoderForSequenceClassification.DefaultCap)

    val id2label: Map[String, String] =
      modelConfig
        .flatMap(c => (c \ "id2label").extractOpt[Map[String, String]])
        .getOrElse(Map.empty)
    val numLabels: Int =
      modelConfig.flatMap(c => (c \ "num_labels").extractOpt[Int]).getOrElse(id2label.size)

    // Regression heads (num_labels == 1) carry no meaningful labels: the raw score is returned.
    val labels: Map[String, Int] =
      if (numLabels <= 1) Map.empty
      else id2label.map { case (idx, label) => label -> idx.toInt }

    // Honor the activation declared by sentence-transformers, defaulting to sigmoid when absent.
    val defaultActivation: String =
      modelConfig
        .flatMap(c => (c \ "sbert_ce_default_activation_function").extractOpt[String])
        .map(CrossEncoderForSequenceClassification.mapSbertActivation)
        .getOrElse(ActivationFunction.sigmoid)

    val annotatorModel = new CrossEncoderForSequenceClassification()
      .setVocabulary(vocabs)
      .setLabels(labels)
      .setModelMaxLength(modelMaxLength)
      .setMaxSentenceLength(modelMaxLength)
      .setActivation(defaultActivation)

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case TensorFlow.name =>
        val (wrapper, signatures) =
          TensorflowWrapper.read(localModelPath, zipped = false, useBundle = true)

        val _signatures = signatures match {
          case Some(s) => s
          case None => throw new Exception("Cannot load signature definitions from model!")
        }

        annotatorModel
          .setSignatures(_signatures)
          .setModelIfNotSet(spark, Some(wrapper), None, None)

      case ONNX.name =>
        val onnxWrapper =
          OnnxWrapper.read(spark, localModelPath, zipped = false, useBundle = true)
        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrapper), None)

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

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[CrossEncoderForSequenceClassification]]. Please refer to
  * that class for the documentation.
  */
object CrossEncoderForSequenceClassification
    extends ReadablePretrainedCrossEncoderForSequenceModel
    with ReadCrossEncoderForSequenceDLModel {

  /** Fallback ceiling when the model config does not declare `max_position_embeddings`. */
  private[dl] val DefaultCap = 512

  private[dl] val SupportedActivations =
    Set(ActivationFunction.sigmoid, ActivationFunction.softmax, ActivationFunction.identity)

  private[dl] val SupportedTruncations =
    Set(CrossEncoderClassification.LongestFirst, CrossEncoderClassification.QueryFirst)

  /** Reads `config.json` from the model root (standard sequence-classifier export) or, failing
    * that, from `assets/`. Returns `None` when no config is present so loading can fall back to
    * sensible defaults.
    */
  private[dl] def readConfigJson(localModelPath: String): Option[JValue] = {
    val rootFile = new File(localModelPath, "config.json")
    val assetsFile = new File(localModelPath + "/assets", "config.json")

    val content: Option[String] =
      if (rootFile.exists()) {
        val src = Source.fromFile(rootFile)(scala.io.Codec.UTF8)
        try Some(src.mkString)
        finally src.close()
      } else if (assetsFile.exists()) {
        Some(loadJsonStringAsset(localModelPath, "config.json"))
      } else None

    content.map(parse(_))
  }

  /** Maps a `sentence-transformers` activation class name (e.g.
    * `torch.nn.modules.linear.Identity`) to one of the supported activation keys.
    */
  private[dl] def mapSbertActivation(fn: String): String = {
    val lower = fn.toLowerCase
    if (lower.contains("sigmoid")) ActivationFunction.sigmoid
    else if (lower.contains("softmax")) ActivationFunction.softmax
    else if (lower.contains("identity")) ActivationFunction.identity
    else ActivationFunction.sigmoid
  }
}
