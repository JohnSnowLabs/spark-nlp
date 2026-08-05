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

package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.ml.ai.BGEM3
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.openvino.{OpenvinoWrapper, ReadOpenvinoModel, WriteOpenvinoModel}
import com.johnsnowlabs.ml.tensorflow.sentencepiece.{
  ReadSentencePieceModel,
  SentencePieceWrapper,
  WriteSentencePieceModel
}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadSentencePieceAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.{ONNX, Openvino}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.slf4j.{Logger, LoggerFactory}

/** Sentence embeddings using BGE-M3.
  *
  * BGE-M3 is a versatile multilingual embedding model from BAAI built on the xlm-roberta-large
  * backbone. Unlike the English dense-only BGE models exposed through [[BGEEmbeddings]], BGE-M3
  * supports up to 8192 tokens, over 100 languages, and produces both:
  *   - a '''dense''' embedding (packed into `Annotation.embeddings`), and
  *   - a '''sparse''' / lexical `{token: weight}` map (packed into `Annotation.metadata` when
  *     [[setReturnSparseEmbeddings]] is enabled).
  *
  * Both outputs are emitted from a single `SENTENCE_EMBEDDINGS` output column. The sparse weights
  * follow the convention used elsewhere in Spark NLP for packing extra information into metadata
  * (e.g. `NerDLModel.includeAllConfidenceScores`).
  *
  * This annotator loads the model through ONNX or OpenVINO. The exported graph is expected to
  * fold in the dense pooling and the `sparse_linear` head so it exposes both a `dense_embedding`
  * (CLS-pooled and L2-normalized) and a `token_weights` output (see the accompanying export
  * notebook). The multi-vector / ColBERT head is out of scope.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val embeddings = BGEM3Embeddings.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("embeddings")
  * }}}
  * The default model is `"bge_m3"`, if no name is provided.
  *
  * For available pretrained models please see the
  * [[https://sparknlp.org/models?q=BGE Models Hub]].
  *
  * '''Sources''' :
  *
  * [[https://arxiv.org/abs/2402.03216 BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation]]
  *
  * [[https://github.com/FlagOpen/FlagEmbedding BGE Github Repository]]
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.embeddings.BGEM3Embeddings
  * import com.johnsnowlabs.nlp.EmbeddingsFinisher
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val embeddings = BGEM3Embeddings.pretrained("bge_m3", "xx")
  *   .setInputCols("document")
  *   .setOutputCol("bge_m3_embeddings")
  *   .setReturnSparseEmbeddings(true)
  *
  * val embeddingsFinisher = new EmbeddingsFinisher()
  *   .setInputCols("bge_m3_embeddings")
  *   .setOutputCols("finished_embeddings")
  *   .setOutputAsVector(true)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   embeddings,
  *   embeddingsFinisher
  * ))
  *
  * val data = Seq("El BGE-M3 admite recuperación densa y dispersa.").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  * }}}
  *
  * @see
  *   [[BGEEmbeddings]] for the English dense-only BGE models
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
class BGEM3Embeddings(override val uid: String)
    extends AnnotatorModel[BGEM3Embeddings]
    with HasBatchedAnnotate[BGEM3Embeddings]
    with WriteOnnxModel
    with WriteOpenvinoModel
    with WriteSentencePieceModel
    with HasEmbeddingsProperties
    with HasStorageRef
    with HasCaseSensitiveProperties
    with HasEngine {

  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT)
  override val outputAnnotatorType: AnnotatorType = AnnotatorType.SENTENCE_EMBEDDINGS

  def this() = this(Identifiable.randomUID("BGE_M3_EMBEDDINGS"))

  /** Max sentence length to process (Default: `512`). BGE-M3 supports up to 8192 tokens.
    *
    * @group param
    */
  val maxSentenceLength =
    new IntParam(this, "maxSentenceLength", "Max sentence length to process")

  /** Whether to compute the sparse / lexical embeddings and pack the `{token: weight}` pairs into
    * the annotation metadata (Default: `false`).
    *
    * @group param
    */
  val returnSparseEmbeddings = new BooleanParam(
    this,
    "returnSparseEmbeddings",
    "Whether to compute the sparse lexical embeddings and pack them into the annotation metadata")

  private var _model: Option[Broadcast[BGEM3]] = None

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(value <= 8192, "BGE-M3 models do not support sequences longer than 8192.")
    require(value >= 1, "The maxSentenceLength must be at least 1")
    set(maxSentenceLength, value)
    this
  }

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  /** @group setParam */
  def setReturnSparseEmbeddings(value: Boolean): this.type = set(returnSparseEmbeddings, value)

  /** @group getParam */
  def getReturnSparseEmbeddings: Boolean = $(returnSparseEmbeddings)

  /** Set Embeddings dimensions for the BGE-M3 model. Only possible to set this the first time it
    * is saved; the dimension is not changeable, it comes from the model config.
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

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      onnxWrapper: Option[OnnxWrapper],
      openvinoWrapper: Option[OpenvinoWrapper],
      spp: SentencePieceWrapper): BGEM3Embeddings = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new BGEM3(onnxWrapper, openvinoWrapper, spp, caseSensitive = $(caseSensitive))))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: BGEM3 = _model.get.value

  setDefault(
    dimension -> 1024,
    batchSize -> 8,
    maxSentenceLength -> 512,
    caseSensitive -> true,
    returnSparseEmbeddings -> false)

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

    val processedAnnotations = if (allAnnotations.nonEmpty) {
      val tokenizedSentences = getModelIfNotSet.tokenize(allAnnotations.map(_._1))
      getModelIfNotSet.predict(
        sentences = allAnnotations.map(_._1),
        tokenizedSentences = tokenizedSentences,
        batchSize = $(batchSize),
        maxSentenceLength = $(maxSentenceLength),
        returnSparse = $(returnSparseEmbeddings))
    } else {
      Seq()
    }

    // Group resulting annotations by rows. If there are no sentences in a given row, return empty
    batchedAnnotations.indices.map(rowIndex => {
      val rowAnnotations = processedAnnotations
        .zip(allAnnotations)
        .filter(_._2._2 == rowIndex)
        .map(_._1)

      if (rowAnnotations.nonEmpty)
        rowAnnotations
      else
        Seq.empty[Annotation]
    })
  }

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    dataset.withColumn(
      getOutputCol,
      wrapSentenceEmbeddingsMetadata(
        dataset.col(getOutputCol),
        $(dimension),
        Some($(storageRef))))
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    val suffix = "_bge_m3"

    writeSentencePieceModel(path, spark, getModelIfNotSet.spp, suffix, BGEM3Embeddings.sppFile)

    getEngine match {
      case ONNX.name =>
        writeOnnxModel(
          path,
          spark,
          getModelIfNotSet.onnxWrapper.get,
          suffix,
          BGEM3Embeddings.onnxFile)
      case Openvino.name =>
        writeOpenvinoModel(
          path,
          spark,
          getModelIfNotSet.openvinoWrapper.get,
          "openvino_model.xml",
          BGEM3Embeddings.openvinoFile)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

}

trait ReadablePretrainedBGEM3Model
    extends ParamsAndFeaturesReadable[BGEM3Embeddings]
    with HasPretrained[BGEM3Embeddings] {
  override val defaultModelName: Some[String] = Some("bge_m3")
  override val defaultLang: String = "xx"

  /** Java compliant-overrides */
  override def pretrained(): BGEM3Embeddings = super.pretrained()

  override def pretrained(name: String): BGEM3Embeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): BGEM3Embeddings =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): BGEM3Embeddings =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadBGEM3DLModel extends ReadOnnxModel with ReadOpenvinoModel with ReadSentencePieceModel {
  this: ParamsAndFeaturesReadable[BGEM3Embeddings] =>

  override val onnxFile: String = "model.onnx"
  override val openvinoFile: String = "bge_m3_openvino"
  override val sppFile: String = "bge_m3_spp"

  private val suffix: String = "_bge_m3"
  private val onnxDataFileSuffix: String = ".onnx_data"

  def readModel(instance: BGEM3Embeddings, path: String, spark: SparkSession): Unit = {
    val spp = readSentencePieceModel(path, spark, "_bge_m3_spp", sppFile)

    instance.getEngine match {
      case ONNX.name =>
        val onnxWrapper =
          readOnnxModel(
            path,
            spark,
            suffix,
            zipped = true,
            useBundle = false,
            modelName = Some(onnxFile),
            dataFilePostfix = Some(onnxDataFileSuffix))
        instance.setModelIfNotSet(spark, Some(onnxWrapper), None, spp)
      case Openvino.name =>
        val openvinoWrapper = readOpenvinoModel(path, spark, "_bge_m3_openvino")
        instance.setModelIfNotSet(spark, None, Some(openvinoWrapper), spp)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): BGEM3Embeddings = {

    val (localModelPath, detectedEngine) = modelSanityCheck(modelPath)

    val spModel = loadSentencePieceAsset(localModelPath, "sentencepiece.bpe.model")

    /*Universal parameters for all engines*/
    val annotatorModel = new BGEM3Embeddings()

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case ONNX.name =>
        val onnxWrapper =
          OnnxWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            dataFileSuffix = Some(onnxDataFileSuffix),
            onnxFileSuffix = Some(suffix))
        annotatorModel
          .setModelIfNotSet(spark, Some(onnxWrapper), None, spModel)

      case Openvino.name =>
        val ovWrapper: OpenvinoWrapper =
          OpenvinoWrapper.read(
            spark,
            localModelPath,
            zipped = false,
            useBundle = true,
            detectedEngine = detectedEngine)
        annotatorModel
          .setModelIfNotSet(spark, None, Some(ovWrapper), spModel)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }
}

/** This is the companion object of [[BGEM3Embeddings]]. Please refer to that class for the
  * documentation.
  */
object BGEM3Embeddings extends ReadablePretrainedBGEM3Model with ReadBGEM3DLModel {
  private[BGEM3Embeddings] val logger: Logger =
    LoggerFactory.getLogger("BGEM3Embeddings")
}
