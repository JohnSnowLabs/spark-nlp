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
package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.ml.gguf.GGUFWrapper
import com.johnsnowlabs.ml.util.LlamaCPP
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.llama.LlamaExtensions
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import de.kherud.llama.{LlamaException, LlamaModel, Pair}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession

import scala.jdk.CollectionConverters._

/** Annotator that uses the llama.cpp library to rerank text documents based on their relevance to
  * a given query using GGUF-format reranking models.
  *
  * This annotator is specifically designed for text reranking tasks, where multiple documents or
  * text passages are ranked according to their relevance to a query. It uses specialized
  * reranking models in GGUF format that output relevance scores for each input document.
  *
  * The reranker takes a query (set via `setQuery`) and a list of documents, then returns the same
  * documents with added metadata containing relevance scores. The documents are processed in
  * batches and each receives a `relevance_score` in its metadata indicating how relevant it is to
  * the provided query.
  *
  * For settable parameters, and their explanations, see [[HasLlamaCppInferenceProperties]],
  * [[HasLlamaCppModelProperties]] and refer to the llama.cpp documentation of
  * [[https://github.com/ggerganov/llama.cpp/tree/7d5e8777ae1d21af99d4f95be10db4870720da91/examples/server server.cpp]]
  * for more information.
  *
  * If the parameters are not set, the annotator will default to use the parameters provided by
  * the model.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val reranker = AutoGGUFReranker.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("reranked_documents")
  *   .setQuery("A man is eating pasta.")
  * }}}
  * The default model is `"bge_reranker_v2_m3_Q4_K_M"`, if no name is provided.
  *
  * For available pretrained models please see the [[https://sparknlp.org/models Models Hub]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/AutoGGUFRerankerTest.scala AutoGGUFRerankerTest]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/examples/python/llama.cpp/llama.cpp_in_Spark_NLP_AutoGGUFReranker.ipynb example notebook]].
  *
  * ==Note==
  * This annotator is designed for reranking tasks and requires setting a query using `setQuery`.
  * The query represents the search intent against which documents will be ranked. Each input
  * document receives a relevance score in the output metadata.
  *
  * To use GPU inference with this annotator, make sure to use the Spark NLP GPU package and set
  * the number of GPU layers with the `setNGpuLayers` method.
  *
  * When using larger models, we recommend adjusting GPU usage with `setNCtx` and `setNGpuLayers`
  * according to your hardware to avoid out-of-memory errors.
  *
  * ==Example==
  *
  * {{{
  * import com.johnsnowlabs.nlp.base._
  * import com.johnsnowlabs.nlp.annotator._
  * import org.apache.spark.ml.Pipeline
  * import spark.implicits._
  *
  * val document = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val reranker = AutoGGUFReranker
  *   .pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("reranked_documents")
  *   .setBatchSize(4)
  *   .setQuery("A man is eating pasta.")
  *
  * val pipeline = new Pipeline().setStages(Array(document, reranker))
  *
  * val data = Seq(
  *   "A man is eating food.",
  *   "A man is eating a piece of bread.",
  *   "The girl is carrying a baby.",
  *   "A man is riding a horse."
  * ).toDF("text")
  * val result = pipeline.fit(data).transform(data)
  * result.select("reranked_documents").show(truncate = false)
  * // Each document will have a relevance_score in metadata showing how relevant it is to the query
  * }}}
  *
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
class AutoGGUFReranker(override val uid: String)
    extends AnnotatorModel[AutoGGUFReranker]
    with HasBatchedAnnotate[AutoGGUFReranker]
    with HasEngine
    with HasLlamaCppModelProperties
    with HasLlamaCppInferenceProperties
    with HasProtectedParams {

  override val outputAnnotatorType: AnnotatorType = AnnotatorType.DOCUMENT
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(AnnotatorType.DOCUMENT)

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("AutoGGUFReranker"))

  private var _model: Option[Broadcast[GGUFWrapper]] = None

  /** @group getParam */
  def getModelIfNotSet: GGUFWrapper = _model.get.value

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, wrapper: GGUFWrapper): this.type = {
    if (_model.isEmpty) {
      _model = Some(spark.sparkContext.broadcast(wrapper))
    }
    this
  }

  /** Closes the llama.cpp model backend freeing resources. The model is reloaded when used again.
   */
  def close(): Unit = GGUFWrapper.closeBroadcastModel(_model)

  val query = new Param[String](
    this,
    "query",
    "The query to be used for reranking. If not set, the input text will be used as the query.")

  /** @group getParam */
  def getQuery: String = $(query)

  /** @group setParam */
  def setQuery(value: String): this.type = set(query, value)

  private[johnsnowlabs] def setEngine(engineName: String): this.type = set(engine, engineName)

  setDefault(
    engine -> LlamaCPP.name,
    useChatTemplate -> true,
    nCtx -> 4096,
    nBatch -> 512,
    nGpuLayers -> 99,
    systemPrompt -> "You are a helpful assistant.",
    batchSize -> 2,
    query -> "")

  /** Sets the number of parallel processes for decoding. This is an alias for `setBatchSize`.
    *
    * @group setParam
    * @param nParallel
    *   The number of parallel processes for decoding
    */
  def setNParallel(nParallel: Int): this.type = {
    setBatchSize(nParallel)
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    getModelIfNotSet.saveToFile(path)
  }

  /** Completes the batch of annotations.
    *
    * @param batchedAnnotations
    *   Annotations (single element arrays) in batches
    * @return
    *   Completed text sequences
    */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
    if (getQuery.isEmpty) {
      throw new IllegalArgumentException(
        "Query must be set for AutoGGUFReranker. Use setQuery to provide a query string.")
    }
    val annotations: Seq[Annotation] = batchedAnnotations.flatten
    // TODO: group by doc and sentence
    if (annotations.nonEmpty) {
      val annotationsText = annotations.map(_.result).toArray

      val modelParams =
        getModelParameters
          .setParallel(getBatchSize) // set parallel decoding to batch size
          .enableReranking() // enable reranking mode

      val model: LlamaModel = getModelIfNotSet.getSession(modelParams)
      val (completedTexts: Array[String], metadata: Array[Map[String, String]]) =
        try {
          val results: Array[Pair[String, java.lang.Float]] =
            model.rerank(true, getQuery, annotationsText: _*).asScala.toArray

          val (rerankedTexts: Array[String], metadata: Array[Map[String, String]]) =
            results.zipWithIndex.map { case (text, index) =>
              val documentText: String = text.getKey
              val rerankScore: Float = text.getValue
              val metadata = Map("query" -> getQuery, "relevance_score" -> rerankScore.toString)
              (documentText, metadata)
            }.unzip
          (rerankedTexts, metadata)
        } catch {
          case e: LlamaException =>
            logger.error("Error in llama.cpp batch completion", e)
            (Array.fill(annotationsText.length)(""), Map("llamacpp_exception" -> e.getMessage))
        }
      annotations.zip(completedTexts).zip(metadata).map { case ((annotation, text), meta) =>
        Seq(
          new Annotation(
            outputAnnotatorType,
            0,
            text.length - 1,
            text,
            annotation.metadata ++ meta))
      }
    } else Seq(Seq.empty[Annotation])
  }
}

trait ReadablePretrainedAutoGGUFReranker
    extends ParamsAndFeaturesFallbackReadable[AutoGGUFReranker]
    with HasPretrained[AutoGGUFReranker] {
  override val defaultModelName: Some[String] = Some("bge_reranker_v2_m3_Q4_K_M")
  override val defaultLang: String = "en"

  /** Java compliant-overrides */
  override def pretrained(): AutoGGUFReranker = super.pretrained()

  override def pretrained(name: String): AutoGGUFReranker = super.pretrained(name)

  override def pretrained(name: String, lang: String): AutoGGUFReranker =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): AutoGGUFReranker =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadAutoGGUFReranker {
  this: ParamsAndFeaturesFallbackReadable[AutoGGUFReranker] =>

  override def fallbackLoad(folder: String, spark: SparkSession): AutoGGUFReranker = {
    val actualFolderPath: String = ResourceHelper.resolvePath(folder)
    val localFolder = ResourceHelper.copyToLocal(actualFolderPath)
    val ggufFile = GGUFWrapper.findGGUFModelInFolder(localFolder)
    loadSavedModel(ggufFile, spark)
  }

  def readModel(instance: AutoGGUFReranker, path: String, spark: SparkSession): Unit = {
    val model: GGUFWrapper = GGUFWrapper.readModel(path, spark)
    instance.setModelIfNotSet(spark, model)
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): AutoGGUFReranker = {
    // TODO potentially enable download from HF-URLS
    val localPath: String = ResourceHelper.copyToLocal(modelPath)
    val annotatorModel = new AutoGGUFReranker()
    annotatorModel
      .setModelIfNotSet(spark, GGUFWrapper.read(spark, localPath))
      .setEngine(LlamaCPP.name)

    val metadata = LlamaExtensions.getMetadataFromFile(localPath)
    if (metadata.nonEmpty) annotatorModel.setMetadata(metadata)
    annotatorModel
  }
}

/** This is the companion object of [[AutoGGUFReranker]]. Please refer to that class for the
  * documentation.
  */
object AutoGGUFReranker extends ReadablePretrainedAutoGGUFReranker with ReadAutoGGUFReranker
