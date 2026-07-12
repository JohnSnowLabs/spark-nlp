/*
 * Copyright 2017-2026 John Snow Labs
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

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.embeddings.MPNetEmbeddings
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{FloatParam, IntParam, Param, Params}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset
import org.slf4j.LoggerFactory

/** Task-level parameters shared between the [[Summarization]] estimator and the fitted
  * [[SummarizationModel]].
  */
trait SummarizationParams extends Params {

  /** Summarization method: `llm`, `encoder_decoder` or `extractive` (Default: `llm`).
    *
    *   - `llm`: an instruction-tuned GGUF LLM (via AutoGGUFModel / llama.cpp) prompted for
    *     summarization.
    *   - `encoder_decoder`: a specialized abstractive summarization model (DistilBART).
    *   - `extractive`: selects the most central sentences of the original document
    *     (embedding-based centrality with position prior and MMR redundancy control).
    *
    * @group param
    */
  val method =
    new Param[String](this, "method", "Summarization method: llm, encoder_decoder or extractive")

  /** @group setParam */
  def setMethod(value: String): this.type = {
    require(
      Summarization.supportedMethods.contains(value),
      s"Method '$value' is not supported. Choose one of: ${Summarization.supportedMethods.mkString(", ")}")
    set(method, value)
  }

  /** @group getParam */
  def getMethod: String = $(method)

  /** Optional pretrained model name overriding the method's default model (Default: `""` = use
    * the method's default).
    *
    * @group param
    */
  val model =
    new Param[String](this, "model", "Pretrained model name overriding the method default")

  /** @group setParam */
  def setModel(value: String): this.type = set(model, value)

  /** @group getParam */
  def getModel: String = $(model)

  /** Target maximum summary length in words/tokens, approximate (Default: `250`).
    * @group param
    */
  val maxSummaryLength =
    new IntParam(this, "maxSummaryLength", "Target maximum summary length (approximate words)")

  /** @group setParam */
  def setMaxSummaryLength(value: Int): this.type = {
    require(value > 0, "maxSummaryLength must be > 0")
    set(maxSummaryLength, value)
  }

  /** @group getParam */
  def getMaxSummaryLength: Int = $(maxSummaryLength)

  /** Minimum summary length in words/tokens, generative methods only (Default: `20`).
    * @group param
    */
  val minSummaryLength =
    new IntParam(this, "minSummaryLength", "Minimum summary length (approximate words)")

  /** @group setParam */
  def setMinSummaryLength(value: Int): this.type = {
    require(value >= 0, "minSummaryLength must be >= 0")
    set(minSummaryLength, value)
  }

  /** @group getParam */
  def getMinSummaryLength: Int = $(minSummaryLength)

  /** Summary style used to build the LLM prompt: `concise`, `detailed` or `bullets` (Default:
    * `concise`). Only applies to the `llm` method.
    * @group param
    */
  val summaryStyle =
    new Param[String](this, "summaryStyle", "Summary style: concise, detailed or bullets")

  /** @group setParam */
  def setSummaryStyle(value: String): this.type = {
    require(
      Summarization.supportedStyles.contains(value),
      s"Style '$value' is not supported. Choose one of: ${Summarization.supportedStyles.mkString(", ")}")
    set(summaryStyle, value)
  }

  /** @group getParam */
  def getSummaryStyle: String = $(summaryStyle)

  /** Optional free-text focus hint included in the LLM prompt, e.g. "main findings" (Default:
    * `""`). Only applies to the `llm` method.
    * @group param
    */
  val focus = new Param[String](this, "focus", "Focus hint for the summary (llm method only)")

  /** @group setParam */
  def setFocus(value: String): this.type = set(focus, value)

  /** @group getParam */
  def getFocus: String = $(focus)

  /** Strategy for documents longer than the model context: `auto`, `truncate` or `hierarchical`
    * (Default: `auto`). `auto` summarizes directly when the document fits and falls back to
    * hierarchical chunk-then-combine summarization when it does not.
    * @group param
    */
  val longDocumentStrategy = new Param[String](
    this,
    "longDocumentStrategy",
    "Long document strategy: auto, truncate or hierarchical")

  /** @group setParam */
  def setLongDocumentStrategy(value: String): this.type = {
    require(
      Summarization.supportedStrategies.contains(value),
      s"Strategy '$value' is not supported. Choose one of: ${Summarization.supportedStrategies
          .mkString(", ")}")
    set(longDocumentStrategy, value)
  }

  /** @group getParam */
  def getLongDocumentStrategy: String = $(longDocumentStrategy)

  /** Number of beams for beam search (Default: `4`). Only applies to the `encoder_decoder`
    * method.
    * @group param
    */
  val numBeams = new IntParam(this, "numBeams", "Number of beams (encoder_decoder method only)")

  /** @group setParam */
  def setNumBeams(value: Int): this.type = {
    require(value > 0, "numBeams must be > 0")
    set(numBeams, value)
  }

  /** @group getParam */
  def getNumBeams: Int = $(numBeams)

  /** Size of n-grams that may not repeat in the generated summary (Default: `3`). Only applies to
    * the `encoder_decoder` method; `0` disables the constraint.
    * @group param
    */
  val noRepeatNgramSize = new IntParam(
    this,
    "noRepeatNgramSize",
    "Forbid repeating this n-gram size (encoder_decoder method only, 0 = disabled)")

  /** @group setParam */
  def setNoRepeatNgramSize(value: Int): this.type = {
    require(value >= 0, "noRepeatNgramSize must be >= 0")
    set(noRepeatNgramSize, value)
  }

  /** @group getParam */
  def getNoRepeatNgramSize: Int = $(noRepeatNgramSize)

  /** Generation temperature (Default: `0.2`). Applies to the `llm` method.
    * @group param
    */
  val temperature = new FloatParam(this, "temperature", "Generation temperature")

  /** @group setParam */
  def setTemperature(value: Float): this.type = {
    require(value >= 0, "temperature must be >= 0")
    set(temperature, value)
  }

  /** @group getParam */
  def getTemperature: Float = $(temperature)

  /** Top-p (nucleus) sampling (Default: `0.9`). Applies to the `llm` method.
    * @group param
    */
  val topP = new FloatParam(this, "topP", "Top-p (nucleus) sampling")

  /** @group setParam */
  def setTopP(value: Float): this.type = {
    require(value > 0 && value <= 1, "topP must be in (0, 1]")
    set(topP, value)
  }

  /** @group getParam */
  def getTopP: Float = $(topP)

  /** Chunk size in approximate tokens used for hierarchical summarization (Default: `0` = derive
    * automatically from the model's context limit).
    * @group param
    */
  val chunkSize =
    new IntParam(this, "chunkSize", "Chunk size in approximate tokens (0 = auto)")

  /** @group setParam */
  def setChunkSize(value: Int): this.type = {
    require(value >= 0, "chunkSize must be >= 0")
    set(chunkSize, value)
  }

  /** @group getParam */
  def getChunkSize: Int = $(chunkSize)

  /** Number of sentences repeated between consecutive chunks (Default: `1`).
    * @group param
    */
  val chunkOverlap =
    new IntParam(this, "chunkOverlap", "Sentence overlap between consecutive chunks")

  /** @group setParam */
  def setChunkOverlap(value: Int): this.type = {
    require(value >= 0, "chunkOverlap must be >= 0")
    set(chunkOverlap, value)
  }

  /** @group getParam */
  def getChunkOverlap: Int = $(chunkOverlap)

  /** MMR trade-off between relevance and redundancy in extractive selection, higher = more
    * relevance-driven (Default: `0.7`). Only applies to the `extractive` method.
    * @group param
    */
  val mmrLambda =
    new FloatParam(this, "mmrLambda", "MMR relevance/redundancy trade-off (extractive only)")

  /** @group setParam */
  def setMmrLambda(value: Float): this.type = {
    require(value >= 0 && value <= 1, "mmrLambda must be in [0, 1]")
    set(mmrLambda, value)
  }

  /** @group getParam */
  def getMmrLambda: Float = $(mmrLambda)

  /** Weight of the lead-position prior in extractive sentence ranking (Default: `0.3`). Only
    * applies to the `extractive` method.
    * @group param
    */
  val positionBias =
    new FloatParam(this, "positionBias", "Lead-position prior weight (extractive only)")

  /** @group setParam */
  def setPositionBias(value: Float): this.type = {
    require(value >= 0, "positionBias must be >= 0")
    set(positionBias, value)
  }

  /** @group getParam */
  def getPositionBias: Float = $(positionBias)

  /** Number of model layers offloaded to the GPU for the `llm` method (Default: `99` = offload
    * all). Set to `0` on CPU-only clusters. Ignored by the other methods.
    * @group param
    */
  val gpuLayers =
    new IntParam(this, "gpuLayers", "GPU layers for the llm method (0 = CPU only)")

  /** @group setParam */
  def setGpuLayers(value: Int): this.type = {
    require(value >= 0, "gpuLayers must be >= 0")
    set(gpuLayers, value)
  }

  /** @group getParam */
  def getGpuLayers: Int = $(gpuLayers)

  setDefault(
    method -> "llm",
    model -> "",
    maxSummaryLength -> 250,
    minSummaryLength -> 20,
    summaryStyle -> "concise",
    focus -> "",
    longDocumentStrategy -> "auto",
    numBeams -> 4,
    noRepeatNgramSize -> 3,
    temperature -> 0.2f,
    topP -> 0.9f,
    chunkSize -> 0,
    chunkOverlap -> 1,
    mmrLambda -> 0.7f,
    positionBias -> 0.3f,
    gpuLayers -> 99)
}

/** High-level, task-oriented document summarization.
  *
  * `Summarization` is a zero-configuration estimator: users state what they want (summary length,
  * style, focus) and the annotator decides how to produce it (model selection, prompting,
  * generation settings, long-document handling). No prompt writing or model choice is required:
  *
  * {{{
  * val summarizer = new Summarization()
  *   .setInputCols("document")
  *   .setOutputCol("summary")
  *
  * val model = pipeline.fit(data) // default model resolves and downloads here
  * }}}
  *
  * Three methods are supported, each with an automatically selected default model:
  *
  *   - `llm` (default): an instruction-tuned GGUF LLM run with llama.cpp ([[AutoGGUFModel]]); the
  *     annotator owns the summarization prompt, system prompt, safe generation defaults and
  *     reasoning-mode suppression.
  *   - `encoder_decoder`: a specialized abstractive summarization model ([[BartTransformer]],
  *     DistilBART fine-tuned on XSum).
  *   - `extractive`: selects the most central sentences from the original document using sentence
  *     embeddings, position-augmented centrality and MMR redundancy control; the output contains
  *     only text from the source document.
  *
  * Documents longer than the model context are handled automatically (see
  * [[SummarizationParams.longDocumentStrategy]]): the document is split at sentence boundaries
  * into overlapping chunks, each chunk is summarized, and the intermediate summaries are combined
  * and summarized again.
  *
  * `fit()` downloads the default (or user-overridden) pretrained model and returns a
  * [[SummarizationModel]]; saved fitted pipelines embed the resolved model and load offline.
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.seq2seq.Summarization
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
  *
  * val summarizer = new Summarization()
  *   .setInputCols("document")
  *   .setOutputCol("summary")
  *   .setMethod("extractive")
  *   .setMaxSummaryLength(100)
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, summarizer))
  * val result = pipeline.fit(data).transform(data)
  * result.select("summary.result").show(false)
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
  * @groupprio param 1
  * @groupprio anno 2
  * @groupprio Ungrouped 3
  * @groupprio setParam 4
  * @groupprio getParam 5
  */
class Summarization(override val uid: String)
    extends AnnotatorApproach[SummarizationModel]
    with SummarizationParams {

  def this() = this(Identifiable.randomUID("SUMMARIZATION"))

  override val description: String =
    "High-level task-oriented document summarization (llm, encoder_decoder or extractive)"

  /** Input Annotator Type: DOCUMENT
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)

  /** Output Annotator Type: DOCUMENT
    * @group anno
    */
  override val outputAnnotatorType: String = DOCUMENT

  private val logger = LoggerFactory.getLogger(classOf[Summarization])

  private def warnIgnored(): Unit = {
    val m = $(method)
    if (m != "llm") {
      if (isSet(focus) && $(focus).nonEmpty)
        logger.warn(s"Parameter 'focus' only applies to method 'llm' and is ignored for '$m'")
      if (isSet(summaryStyle))
        logger.warn(
          s"Parameter 'summaryStyle' only applies to method 'llm' and is ignored for '$m'")
      if (isSet(temperature))
        logger.warn(
          s"Parameter 'temperature' only applies to method 'llm' and is ignored for '$m'")
      if (isSet(topP))
        logger.warn(s"Parameter 'topP' only applies to method 'llm' and is ignored for '$m'")
    }
    if (m != "encoder_decoder" && isSet(numBeams))
      logger.warn(
        s"Parameter 'numBeams' only applies to method 'encoder_decoder' and is ignored for '$m'")
    if (m != "encoder_decoder" && isSet(noRepeatNgramSize))
      logger.warn(
        "Parameter 'noRepeatNgramSize' only applies to method 'encoder_decoder' " +
          s"and is ignored for '$m'")
    if (m != "extractive") {
      if (isSet(mmrLambda))
        logger.warn(
          s"Parameter 'mmrLambda' only applies to method 'extractive' and is ignored for '$m'")
      if (isSet(positionBias))
        logger.warn(
          s"Parameter 'positionBias' only applies to method 'extractive' and is ignored for '$m'")
    } else if (isSet(minSummaryLength)) {
      logger.warn(
        "Parameter 'minSummaryLength' does not apply to method 'extractive' " +
          "(maxSummaryLength is the sentence-selection budget) and is ignored")
    }
  }

  /** Resolves the summarization method to a configured [[SummarizationModel]], downloading the
    * default pretrained delegate when no explicit model name was set.
    */
  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): SummarizationModel = {
    require(
      $(minSummaryLength) <= $(maxSummaryLength),
      s"minSummaryLength (${$(minSummaryLength)}) must be <= maxSummaryLength " +
        s"(${$(maxSummaryLength)})")
    warnIgnored()

    val summarizationModel = new SummarizationModel()
    val requested = $(model)

    $(method) match {
      case "llm" =>
        val name = if (requested.nonEmpty) requested else Summarization.defaultLlmModel
        logger.info(s"Summarization: resolving llm delegate '$name'")
        val delegate = AutoGGUFModel.pretrained(name)
        summarizationModel
          .setLlmDelegate(delegate)
          .setResolvedModel(name)

      case "encoder_decoder" =>
        val name =
          if (requested.nonEmpty) requested else Summarization.defaultEncoderDecoderModel
        logger.info(s"Summarization: resolving encoder_decoder delegate '$name'")
        val delegate = BartTransformer.pretrained(name)
        summarizationModel
          .setSeq2SeqDelegate(delegate)
          .setResolvedModel(name)

      case "extractive" =>
        val name = if (requested.nonEmpty) requested else Summarization.defaultEmbeddingsModel
        logger.info(s"Summarization: resolving extractive embeddings delegate '$name'")
        val delegate = MPNetEmbeddings.pretrained(name)
        summarizationModel
          .setEmbeddingsDelegate(delegate)
          .setResolvedModel(name)
    }

    summarizationModel
  }
}

/** This is the companion object of [[Summarization]]. Please refer to that class for the
  * documentation.
  */
object Summarization extends DefaultParamsReadable[Summarization] {

  val supportedMethods: Array[String] = Array("llm", "encoder_decoder", "extractive")
  val supportedStyles: Array[String] = Array("concise", "detailed", "bullets")
  val supportedStrategies: Array[String] = Array("auto", "truncate", "hierarchical")

  /** Default pretrained model per method. */
  val defaultLlmModel: String = "qwen3_4b_q8_0_gguf"
  val defaultEncoderDecoderModel: String = "distilbart_xsum_12_6"
  val defaultEmbeddingsModel: String = "all_mpnet_base_v2"
}
