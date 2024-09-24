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

import com.johnsnowlabs.ml.ai.Mxbai
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.tensorflow._
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ONNX, TensorFlow}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.tokenizer.wordpiece.{BasicTokenizer, WordpieceEncoder}
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.{Logger, LoggerFactory}

import scala.util.Try

/** Sentence embeddings using Mxbai Embeddings.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val embeddings = MxbaiEmbeddings.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("Mxbai_embeddings")
  * }}}
  * The default model is `"mxbai_large_v1"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?q=Mxbai Models Hub]].
  *
  * For extended examples of usage, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/MxbaiEmbeddingsTestSpec.scala MxbaiEmbeddingsTestSpec]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.embeddings.MxbaiEmbeddings
  * import com.johnsnowlabs.nlp.EmbeddingsFinisher
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val embeddings = MxbaiEmbeddings.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("Mxbai_embeddings")
  *
  * val embeddingsFinisher = new EmbeddingsFinisher()
  *   .setInputCols("Mxbai_embeddings")
  *   .setOutputCols("finished_embeddings")
  *   .setOutputAsVector(true)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   embeddings,
  *   embeddingsFinisher
  * ))
  *
  * val data = Seq("hello world", "hello moon").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  * +--------------------------------------------------------------------------------+
  * |                                                                          result|
  * +--------------------------------------------------------------------------------+
  * |[0.50387806, 0.5861606, 0.35129607, -0.76046336, -0.32446072, -0.117674336, 0...|
  * |[0.6660665, 0.961762, 0.24854276, -0.1018044, -0.6569202, 0.027635604, 0.1915...|
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
class MxbaiEmbeddings(override val uid: String)
    extends AnnotatorModel[MxbaiEmbeddings]
    with HasBatchedAnnotate[MxbaiEmbeddings]
    with WriteTensorflowModel
    with WriteOnnxModel
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
  private var _model: Option[Broadcast[Mxbai]] = None

  def this() = this(Identifiable.randomUID("Mxbai_EMBEDDINGS"))

  /** @group setParam */
  def setConfigProtoBytes(bytes: Array[Int]): MxbaiEmbeddings.this.type =
    set(this.configProtoBytes, bytes)

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(
      value <= 512,
      "Mxbai models do not support sequences longer than 512 because of trainable positional embeddings.")
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

  /** Pooling strategy to use for sentence embeddings.
    *
    * Available pooling strategies for sentence embeddings are:
    *
    *   - `"cls"`: leading `[CLS]` token
    *   - `"cls_avg"`: leading `[CLS]` token + mean of all other tokens
    *   - `"last"`: embeddings of the last token in the sequence
    *   - `"avg"`: mean of all tokens
    *   - `"max"`: max of all embedding values for the token sequence
    *   - `"all"`: return all token embeddings
    *   - `"int"`: An integer number, which represents the index of the token to use as the
    *     embedding
    *
    * @group param
    */
  val poolingStrategy =
    new Param[String](this, "poolingStrategy", "Pooling strategy to use for sentence embeddings")

  def getPoolingStrategy: String = $(poolingStrategy)

  /** Pooling strategy to use for sentence embeddings.
    *
    * Available pooling strategies for sentence embeddings are:
    *
    *   - `"cls"`: leading `[CLS]` token
    *   - `"cls_avg"`: leading `[CLS]` token + mean of all other tokens
    *   - `"last"`: embeddings of the last token in the sequence
    *   - `"avg"`: mean of all tokens
    *   - `"max"`: max of all embedding features of the entire token sequence
    *   - `"int"`: An integer number, which represents the index of the token to use as the
    *     embedding
    *
    * @group setParam
    */
  def setPoolingStrategy(value: String): this.type = {
    val validStrategies = Set("cls", "cls_avg", "last", "avg", "max")

    if (validStrategies.contains(value) || Try(value.toInt).isSuccess) {
      set(poolingStrategy, value)
    } else {
      throw new IllegalArgumentException(
        s"Invalid pooling strategy: $value. " +
          s"Valid strategies are: ${validStrategies.mkString(", ")} or an integer.")
    }
  }

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      tensorflowWrapper: Option[TensorflowWrapper],
      onnxWrapper: Option[OnnxWrapper]): MxbaiEmbeddings = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new Mxbai(
            tensorflowWrapper,
            onnxWrapper,
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

  setDefault(
    dimension -> 1024,
    batchSize -> 8,
    maxSentenceLength -> 512,
    caseSensitive -> false,
    poolingStrategy -> "cls")

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
        maxSentenceLength = $(maxSentenceLength),
        poolingStrategy = getPoolingStrategy)
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
  def getModelIfNotSet: Mxbai = _model.get.value

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    val suffix = "_Mxbai"

    getEngine match {
      case TensorFlow.name =>
        writeTensorflowModelV2(
          path,
          spark,
          getModelIfNotSet.tensorflowWrapper.get,
          suffix,
          MxbaiEmbeddings.tfFile,
          configProtoBytes = getConfigProtoBytes,
          savedSignatures = getSignatures)
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          suffix,
          MxbaiEmbeddings.onnxFile)

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

trait ReadablePretrainedMxbaiModel
    extends ParamsAndFeaturesReadable[MxbaiEmbeddings]
    with HasPretrained[MxbaiEmbeddings] {
  override val defaultModelName: Some[String] = Some("mxbai_large_v1")

  /** Java compliant-overrides */
  override def pretrained(): MxbaiEmbeddings = super.pretrained()

  override def pretrained(name: String): MxbaiEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): MxbaiEmbeddings =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): MxbaiEmbeddings =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadMxbaiDLModel extends ReadTensorflowModel with ReadOnnxModel {
  this: ParamsAndFeaturesReadable[MxbaiEmbeddings] =>

  override val tfFile: String = "Mxbai_tensorflow"
  override val onnxFile: String = "Mxbai_onnx"

  def readModel(instance: MxbaiEmbeddings, path: String, spark: SparkSession): Unit = {

    instance.getEngine match {
      case TensorFlow.name =>
        val tfWrapper = readTensorflowModel(path, spark, "_Mxbai_tf")
        instance.setModelIfNotSet(spark, Some(tfWrapper), None)

      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(path, spark, "_Mxbai_onnx", zipped = true, useBundle = false, None)
        instance.setModelIfNotSet(spark, None, Some(onnxWrapper))

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): MxbaiEmbeddings = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val vocabs = loadTextAsset(localModelPath, "vocab.txt").zipWithIndex.toMap

    /*Universal parameters for all engines*/
    val annotatorModel = new MxbaiEmbeddings()
      .setVocabulary(vocabs)

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case TensorFlow.name =>
        val (wrapper, signatures) =
          TensorflowWrapper.read(
            localModelPath,
            zipped = false,
            useBundle = true,
            tags = Array("serve"))

        val _signatures = signatures match {
          case Some(s) => s
          case None => throw new Exception("Cannot load signature definitions from model!")
        }

        /** the order of setSignatures is important if we use getSignatures inside
          * setModelIfNotSet
          */
        annotatorModel
          .setSignatures(_signatures)
          .setModelIfNotSet(spark, Some(wrapper), None)

      case ONNX.name =>
        val onnxWrapper =
          OnnxWrapper.read(spark, localModelPath, zipped = false, useBundle = true)
        annotatorModel
          .setModelIfNotSet(spark, None, Some(onnxWrapper))

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[MxbaiEmbeddings]]. Please refer to that class for the
  * documentation.
  */
object MxbaiEmbeddings extends ReadablePretrainedMxbaiModel with ReadMxbaiDLModel {
  private[MxbaiEmbeddings] val logger: Logger =
    LoggerFactory.getLogger("MxbaiEmbeddings")
}
