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

package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.ml.ai.MiniLM
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.openvino.{OpenvinoWrapper, ReadOpenvinoModel, WriteOpenvinoModel}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ONNX, Openvino}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.storage.HasStorageRef
import com.johnsnowlabs.util.FileHelper
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.{Logger, LoggerFactory}

import java.nio.file.Files
import java.util.UUID

/** Sentence embeddings using MiniLM.
  *
  * MiniLM, a lightweight and efficient sentence embedding model that can generate text embeddings
  * for various NLP tasks (e.g., classification, retrieval, clustering, text evaluation, etc.)
  *
  * Note that this annotator is only supported for Spark Versions 3.4 and up.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val embeddings = MiniLMEmbeddings.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("minilm_embeddings")
  * }}}
  * The default model is `"minilm_l6_v2"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?q=MiniLM Models Hub]].
  *
  * For extended examples of usage, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/MiniLMEmbeddingsTestSpec.scala MiniLMEmbeddingsTestSpec]].
  *
  * '''Sources''' :
  *
  * [[https://arxiv.org/abs/2002.10957 MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers]]
  *
  * [[https://github.com/microsoft/MiniLM MiniLM Github Repository]]
  *
  * ''' Paper abstract '''
  *
  * ''We present a simple and effective approach to compress large pre-trained Transformer models
  * by distilling the self-attention module of the last Transformer layer. The compressed model
  * (called MiniLM) can be trained with task-agnostic distillation and then fine-tuned on various
  * downstream tasks. We evaluate MiniLM on the GLUE benchmark and show that it achieves
  * comparable results with BERT-base while being 4.3x smaller and 5.5x faster. We also show that
  * MiniLM can be further compressed to 22x smaller and 12x faster than BERT-base while
  * maintaining comparable performance.''
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.embeddings.MiniLMEmbeddings
  * import com.johnsnowlabs.nlp.EmbeddingsFinisher
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val embeddings = MiniLMEmbeddings.pretrained("minilm_l6_v2", "en")
  *   .setInputCols("document")
  *   .setOutputCol("minilm_embeddings")
  *
  * val embeddingsFinisher = new EmbeddingsFinisher()
  *   .setInputCols("minilm_embeddings")
  *   .setOutputCols("finished_embeddings")
  *   .setOutputAsVector(true)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   embeddings,
  *   embeddingsFinisher
  * ))
  *
  * val data = Seq("This is a sample sentence for embedding generation.",
  * "Another example sentence to demonstrate MiniLM embeddings.")
  *
  * ).toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(finished_embeddings) as result").show(1, 80)
  * +--------------------------------------------------------------------------------+
  * |                                                                          result|
  * +--------------------------------------------------------------------------------+
  * |[[0.1234567, -0.2345678, 0.3456789, -0.4567890, 0.5678901, -0.6789012...|
  * |[[0.2345678, -0.3456789, 0.4567890, -0.5678901, 0.6789012, -0.7890123...|
  * +--------------------------------------------------------------------------------+
  * }}}
  *
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
class MiniLMEmbeddings(override val uid: String)
    extends AnnotatorModel[MiniLMEmbeddings]
    with HasBatchedAnnotate[MiniLMEmbeddings]
    with WriteOnnxModel
    with WriteOpenvinoModel
    with HasEmbeddingsProperties
    with HasStorageRef
    with HasCaseSensitiveProperties
    with HasEngine {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  override val inputAnnotatorTypes: Array[String] =
    Array(AnnotatorType.DOCUMENT)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS

  /** ConfigProto from tensorflow, serialized into byte array. Get with
    * `config_proto.SerializeToString()`
    *
    * @group param
    */
  val configProtoBytes = new IntArrayParam(
    this,
    "configProtoBytes",
    "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()")

  /** Max sentence length to process (Default: `128`)
    *
    * @group param
    */
  val maxSentenceLength =
    new IntParam(this, "maxSentenceLength", "Max sentence length to process")

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
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary").setProtected()

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  /** It contains TF model signatures for the laded saved model
    *
    * @group param
    */
  val signatures =
    new MapFeature[String, String](model = this, name = "signatures").setProtected()
  private var _model: Option[Broadcast[MiniLM]] = None

  def this() = this(Identifiable.randomUID("MINILM_EMBEDDINGS"))

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): MiniLMEmbeddings.this.type =
    set(this.configProtoBytes, bytes)

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(
      value <= 512,
      "MiniLM models do not support sequences longer than 512 because of trainable positional embeddings.")
    require(value >= 1, "The maxSentenceLength must be at least 1")
    set(maxSentenceLength, value)
    this
  }

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  /** @group setParam */
  def setSignatures(value: Map[String, String]): this.type = {
    if (get(signatures).isEmpty)
      set(signatures, value)
    this
  }

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      onnxWrapper: Option[OnnxWrapper],
      openvinoWrapper: Option[OpenvinoWrapper]): MiniLMEmbeddings = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new MiniLM(
            onnxWrapper,
            openvinoWrapper,
            configProtoBytes = getConfigProtoBytes,
            sentenceStartTokenId = sentenceStartTokenId,
            sentenceEndTokenId = sentenceEndTokenId,
            signatures = getSignatures)))
    }

    this
  }

  /** Set Embeddings dimensions for the BERT model Only possible to set this when the first time
    * is saved dimension is not changeable, it comes from BERT config file
    *
    * @group setParam
    */
  override def setDimension(value: Int): this.type = {
    if (get(dimension).isEmpty)
      set(this.dimension, value)
    this
  }

  /** Whether to lowercase tokens or not
    *
    * @group setParam
    */
  override def setCaseSensitive(value: Boolean): this.type = {
    if (get(caseSensitive).isEmpty)
      set(this.caseSensitive, value)
    this
  }

  setDefault(dimension -> 384, batchSize -> 8, maxSentenceLength -> 128, caseSensitive -> false)

  def tokenize(sentences: Seq[Annotation]): Seq[WordpieceTokenizedSentence] = {
    val basicTokenizer = new BasicTokenizer($(caseSensitive))
    val encoder = new WordpieceEncoder($$(vocabulary))
    sentences.map { s =>
      val sent = Sentence(
        content = s.result,
        start = s.begin,
        end = s.end,
        metadata = Some(s.metadata),
        index = s.begin)
      val tokens = basicTokenizer.tokenize(sent)
      val wordpieceTokens = tokens.flatMap(token => encoder.encode(token))
      WordpieceTokenizedSentence(wordpieceTokens)
    }
  }

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

    val allAnnotations = batchedAnnotations
      .filter(_.nonEmpty)
      .zipWithIndex
      .flatMap { case (annotations, i) =>
        annotations.filter(_.result.nonEmpty).map(x => (x, i))
      }

    // Tokenize sentences
    val tokenizedSentences = tokenize(allAnnotations.map(_._1))
    val processedAnnotations = if (allAnnotations.nonEmpty) {
      this.getModelIfNotSet.predict(
        sentences = allAnnotations.map(_._1),
        tokenizedSentences = tokenizedSentences,
        batchSize = $(batchSize),
        maxSentenceLength = $(maxSentenceLength))
    } else {
      Seq()
    }

    // Group resulting annotations by rows. If there are not sentences in a given row, return empty sequence
    batchedAnnotations.indices.map(rowIndex => {
      val rowAnnotations = processedAnnotations
        // zip each annotation with its corresponding row index
        .zip(allAnnotations)
        // select the sentences belonging to the current row
        .filter(_._2._2 == rowIndex)
        // leave the annotation only
        .map(_._1)

      if (rowAnnotations.nonEmpty)
        rowAnnotations
      else
        Seq.empty[Annotation]
    })

  }

  /** @group getParam */
  def getModelIfNotSet: MiniLM = _model.get.value

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    val suffix = "_minilm"

    getEngine match {
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          suffix,
          MiniLMEmbeddings.onnxFile)
      case Openvino.name =>
        writeOpenvinoModel(
          path,
          spark,
          getModelIfNotSet.openvinoWrapper.get,
          suffix,
          MiniLMEmbeddings.openvinoFile)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  /** @group getParam */
  def getConfigProtoBytes: Option[Array[Byte]] = get(this.configProtoBytes).map(_.map(_.toByte))

  /** @group getParam */
  def getSignatures: Option[Map[String, String]] = get(this.signatures)

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    dataset.withColumn(
      getOutputCol,
      wrapSentenceEmbeddingsMetadata(
        dataset.col(getOutputCol),
        $(dimension),
        Some($(storageRef))))
  }

}

trait ReadablePretrainedMiniLMModel
    extends ParamsAndFeaturesReadable[MiniLMEmbeddings]
    with HasPretrained[MiniLMEmbeddings] {
  override val defaultModelName: Some[String] = Some("minilm_l6_v2")

  /** Java compliant-overrides */
  override def pretrained(): MiniLMEmbeddings = super.pretrained()

  override def pretrained(name: String): MiniLMEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): MiniLMEmbeddings =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): MiniLMEmbeddings =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadMiniLMDLModel extends ReadOnnxModel with ReadOpenvinoModel {
  this: ParamsAndFeaturesReadable[MiniLMEmbeddings] =>

  override val onnxFile: String = "minilm_onnx"
  override val openvinoFile: String = "minilm_openvino"

  def readModel(instance: MiniLMEmbeddings, path: String, spark: SparkSession): Unit = {

    instance.getEngine match {
      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(path, spark, "_minilm_onnx", zipped = true, useBundle = false, None)
        instance.setModelIfNotSet(spark, Some(onnxWrapper), None)

      case Openvino.name =>
        val openvinoWrapper = readOpenvinoModel(path, spark, "_minilm_ov", zipped = true)
        instance.setModelIfNotSet(spark, None, Some(openvinoWrapper))

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

  }

  addReader(readModel)

  def loadSavedModel(
      modelPath: String,
      spark: SparkSession,
      useOpenvino: Boolean = false): MiniLMEmbeddings = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap

    /*Universal parameters for all engines*/
    val annotatorModel = new MiniLMEmbeddings()
      .setVocabulary(vocabs)
    val modelEngine =
      if (useOpenvino)
        Openvino.name
      else
        detectedEngine
    annotatorModel.set(annotatorModel.engine, modelEngine)

    modelEngine match {
      case ONNX.name =>
        val onnxWrapper =
          OnnxWrapper.read(spark, localModelPath, zipped = false, useBundle = true)
        annotatorModel
          .setModelIfNotSet(spark, Some(onnxWrapper), None)

      case Openvino.name =>
        val ovWrapper: OpenvinoWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine)
        annotatorModel
          .setModelIfNotSet(spark, None, Some(ovWrapper))

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[MiniLMEmbeddings]]. Please refer to that class for the
  * documentation.
  */
object MiniLMEmbeddings extends ReadablePretrainedMiniLMModel with ReadMiniLMDLModel {
  private[MiniLMEmbeddings] val logger: Logger =
    LoggerFactory.getLogger("MiniLMEmbeddings")
}
