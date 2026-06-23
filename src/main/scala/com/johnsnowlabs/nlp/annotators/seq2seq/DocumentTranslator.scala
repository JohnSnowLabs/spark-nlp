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

package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.ml.gguf.GGUFWrapper
import com.johnsnowlabs.ml.util.LlamaCPP
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.sbd.sat.SentenceDetectorSaTModel
import com.johnsnowlabs.nlp.llama.LlamaExtensions
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.reader.Reader2Doc
import de.kherud.llama.{InferenceParameters, LlamaException, LlamaModel}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{BooleanParam, FloatParam, IntParam, Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}

/** DocumentTranslator reads documents from any supported file type and translates them with a
  * llama.cpp GGUF large-language-model – all in a single Pipeline stage.
  *
  *   1. A [[Reader2Doc]] stage reads files from `contentPath` (or from a DataFrame column set via
  *      `inputCol`) and converts them to Spark NLP `DOCUMENT` annotations. Every file format that
  *      [[Reader2Doc]] supports is accepted: PDF, Word (.doc/.docx), ODT, Excel (.xls/.xlsx),
  *      PowerPoint (.ppt/.pptx), HTML, plain-text, RTF, Markdown, XML, CSV, and email
  *      (.eml/.msg).
  *   1. A [[SentenceDetectorSaTModel]] splits each document into sentences. Setting
  *      [[minSentenceLength]] / [[maxSentenceLength]] (characters) makes the SaT model emit
  *      length-bounded segments directly, so DocumentTranslator never has to pack/merge chunks.
  *   1. The GGUF model translates the sentences. Each sentence is embedded into
  *      [[promptTemplate]] (with [[srcLang]] / [[tgtLang]]) and completed with the same
  *      `multiComplete` batch logic used by [[AutoGGUFModel]]. Every sentence from every
  *      collected document is flattened into one batch and translated with up to [[batchSize]]
  *      sentences decoding concurrently (llama.cpp parallel slots), keeping the model fully
  *      utilised regardless of how the sentences are distributed across documents.
  *   1. The translated sentences are regrouped per document and merged back into one `DOCUMENT`
  *      annotation.
  *
  * All llama.cpp model / inference parameters are inherited from [[HasLlamaCppModelProperties]]
  * and [[HasLlamaCppInferenceProperties]] – the same parameters exposed by [[AutoGGUFModel]] – so
  * e.g. `setNCtx`, `setNPredict`, `setTemperature`, `setTopK`, `setReasoningBudget`,
  * `setSystemPrompt` are available directly.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val translator = DocumentTranslator.pretrained()
  *   .setContentPath("src/test/resources/reader/html/")
  *   .setContentType("text/html")
  *   .setSrcLang("English")
  *   .setTgtLang("French")
  *   .setMaxSentenceLength(600)
  *   .setOutputCol("translation")
  * }}}
  * The default model is `"Phi_4_mini_instruct_Q4_K_M_gguf"`, default language is `"en"`, if no
  * values are provided.
  *
  * For available pretrained models please see the [[https://sparknlp.org/models Models Hub]].
  *
  * '''Note:''' Translation is computationally expensive; a GPU is recommended. With parallel
  * decoding the total context `nCtx` (in tokens) is split across the [[batchSize]] slots, so the
  * per-sentence budget is `nCtx / batchSize` and must cover the prompt plus the generated
  * translation. Size it as `nCtx >= batchSize * (promptTokens + nPredict)`; raise [[setNCtx]]
  * when raising [[setBatchSize]], [[setMaxSentenceLength]] or [[setNPredict]].
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
  * @groupprio param  1
  * @groupprio anno   2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class DocumentTranslator(override val uid: String)
    extends Transformer
    with HasOutputAnnotatorType
    with HasOutputAnnotationCol
    with HasEngine
    with HasLlamaCppModelProperties
    with HasLlamaCppInferenceProperties
    with HasProtectedParams
    with CompletionPostProcessing {

  def this() = this(Identifiable.randomUID("DOCUMENT_TRANSLATOR"))

  /** Output Annotator Type: DOCUMENT
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  // ── Reader2Doc parameters ─────────────────────────────────────────────────

  val contentPath: Param[String] =
    new Param[String](this, "contentPath", "Path to the file or directory to read documents from")

  /** @group setParam */
  def setContentPath(value: String): this.type = set(contentPath, value)

  /** @group getParam */
  def getContentPath: String = $(contentPath)

  val contentType: Param[String] =
    new Param[String](
      this,
      "contentType",
      "MIME content-type hint forwarded to Reader2Doc (empty = auto-detect from file extension)")

  /** @group setParam */
  def setContentType(value: String): this.type = set(contentType, value)

  /** @group getParam */
  def getContentType: String = $(contentType)

  val inputCol: Param[String] =
    new Param[String](
      this,
      "inputCol",
      "DataFrame column holding raw text for Reader2Doc to parse instead of reading from contentPath")

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group getParam */
  def getInputCol: String = $(inputCol)

  /** When `true` (default) [[Reader2Doc]] merges all extracted elements from a single file into
    * one `DOCUMENT` annotation before translation. When `false` every extracted element becomes
    * its own `DOCUMENT` annotation and is translated independently.
    *
    * @group param
    */
  val outputAsDocument: BooleanParam =
    new BooleanParam(
      this,
      "outputAsDocument",
      "Whether Reader2Doc merges all extracted elements into a single DOCUMENT annotation per file")

  /** @group setParam */
  def setOutputAsDocument(value: Boolean): this.type = set(outputAsDocument, value)

  /** @group getParam */
  def getOutputAsDocument: Boolean = $(outputAsDocument)

  /** String used by [[Reader2Doc]] to join multiple extracted text elements when
    * `outputAsDocument` is `true` (Default: `"\n"`).
    *
    * @group param
    */
  val joinString: Param[String] =
    new Param[String](
      this,
      "joinString",
      "String used to join extracted elements into one document when outputAsDocument is true")

  /** @group setParam */
  def setJoinString(value: String): this.type = set(joinString, value)

  /** @group getParam */
  def getJoinString: String = $(joinString)

  // ── SentenceDetectorSaTModel parameters ───────────────────────────────────

  /** Minimum sentence length in characters used by the [[SentenceDetectorSaTModel]] (Default: `0`
    * \= no minimum). When [[minSentenceLength]] or [[maxSentenceLength]] is `> 0`, the SaT model
    * switches to length-constrained segmentation.
    *
    * @group param
    */
  val minSentenceLength: IntParam =
    new IntParam(
      this,
      "minSentenceLength",
      "Minimum sentence length in characters for SentenceDetectorSaTModel (0 = unset)")

  /** @group setParam */
  def setMinSentenceLength(value: Int): this.type = {
    require(value >= 0, "minSentenceLength must be >= 0.")
    set(minSentenceLength, value)
  }

  /** @group getParam */
  def getMinSentenceLength: Int = $(minSentenceLength)

  /** Maximum sentence length in characters used by the [[SentenceDetectorSaTModel]] (Default: `0`
    * \= no maximum). See [[minSentenceLength]].
    *
    * @group param
    */
  val maxSentenceLength: IntParam =
    new IntParam(
      this,
      "maxSentenceLength",
      "Maximum sentence length in characters for SentenceDetectorSaTModel (0 = unset)")

  /** @group setParam */
  def setMaxSentenceLength(value: Int): this.type = {
    require(value >= 0, "maxSentenceLength must be >= 0.")
    set(maxSentenceLength, value)
  }

  /** @group getParam */
  def getMaxSentenceLength: Int = $(maxSentenceLength)

  /** Boundary probability threshold for the [[SentenceDetectorSaTModel]] (Default: `0.25`).
    * Ignored when length-constrained segmentation is active.
    *
    * @group param
    */
  val sentenceThreshold: FloatParam =
    new FloatParam(
      this,
      "sentenceThreshold",
      "Boundary probability threshold for SentenceDetectorSaTModel (default 0.25)")

  /** @group setParam */
  def setSentenceThreshold(value: Float): this.type = set(sentenceThreshold, value)

  /** @group getParam */
  def getSentenceThreshold: Float = $(sentenceThreshold)

  // ── Translation prompt parameters ─────────────────────────────────────────

  /** Source language interpolated into [[promptTemplate]] (e.g. `"English"`).
    * @group param
    */
  val srcLang: Param[String] =
    new Param[String](this, "srcLang", "Source language used to build the translation prompt")

  /** @group setParam */
  def setSrcLang(value: String): this.type = set(srcLang, value)

  /** @group getParam */
  def getSrcLang: String = $(srcLang)

  /** Target language interpolated into [[promptTemplate]] (e.g. `"French"`).
    * @group param
    */
  val tgtLang: Param[String] =
    new Param[String](this, "tgtLang", "Target language used to build the translation prompt")

  /** @group setParam */
  def setTgtLang(value: String): this.type = set(tgtLang, value)

  /** @group getParam */
  def getTgtLang: String = $(tgtLang)

  /** Per-sentence translation prompt template. Three placeholders are interpolated per sentence:
    *   - `{srcLang}` -> [[srcLang]]
    *   - `{tgtLang}` -> [[tgtLang]]
    *   - `{text}` -> the sentence to translate
    *
    * The filled-in prompt becomes the model's completion input, so the source text is part of the
    * prompt itself. The default mirrors the few-shot completion format expected by
    * translation-tuned GGUF models, e.g.:
    * {{{
    * Translate the following text from Portuguese into English.
    * Portuguese: Um grupo de investigadores lançou um novo modelo.
    * English:
    * }}}
    *
    * @group param
    */
  val promptTemplate: Param[String] =
    new Param[String](
      this,
      "promptTemplate",
      "Per-sentence translation prompt template; {srcLang}, {tgtLang} and {text} are interpolated")

  /** @group setParam */
  def setPromptTemplate(value: String): this.type = set(promptTemplate, value)

  /** @group getParam */
  def getPromptTemplate: String = $(promptTemplate)

  /** Number of sentences translated concurrently (Default: `4`). This is the llama.cpp
    * parallel-decoding slot count (`n_parallel`): sentences from all documents are flattened into
    * one batch and up to `batchSize` of them decode at once. The total context [[nCtx]] is split
    * across these slots, so `nCtx / batchSize` must cover one sentence's prompt plus
    * [[nPredict]].
    *
    * @group param
    */
  val batchSize: IntParam =
    new IntParam(this, "batchSize", "Number of sentences translated concurrently (llama slots)")

  /** @group setParam */
  def setBatchSize(value: Int): this.type = {
    require(value > 0, "batchSize must be > 0.")
    set(batchSize, value)
  }

  /** @group getParam */
  def getBatchSize: Int = $(batchSize)

  /** Alias for [[setBatchSize]] (number of llama.cpp parallel decoding slots).
    * @group setParam
    */
  def setNParallel(value: Int): this.type = setBatchSize(value)

  setDefault(
    contentPath -> "",
    contentType -> "",
    inputCol -> "",
    outputAsDocument -> true,
    joinString -> "\n",
    minSentenceLength -> 0,
    maxSentenceLength -> 0,
    sentenceThreshold -> 0.25f,
    srcLang -> "English",
    tgtLang -> "French",
    promptTemplate ->
      ("Translate the following text from {srcLang} into {tgtLang}.\n" +
        "{srcLang}: {text}\n{tgtLang}:"),
    batchSize -> 4,
    // Inherited llama.cpp params – translation-tuned defaults.
    // nCtx is the TOTAL token budget split across the batchSize parallel slots, so
    // nCtx / batchSize must cover one sentence's prompt + nPredict. 4096 / 4 = 1024 tokens/slot.
    engine -> LlamaCPP.name,
    useChatTemplate -> true,
    nCtx -> 8192,
    nBatch -> 512,
    nPredict -> 512,
    nGpuLayers -> 99,
    reasoningBudget -> 0,
    systemPrompt -> "You are a helpful assistant.")

  // ── GGUF model (held + broadcast, mirrors AutoGGUFModel) ──────────────────

  private var _model: Option[Broadcast[GGUFWrapper]] = None

  /** @group getParam */
  def getModelIfNotSet: GGUFWrapper = _model.get.value

  /** @group setParam */
  def setModelIfNotSet(spark: SparkSession, wrapper: GGUFWrapper): this.type = {
    if (_model.isEmpty) _model = Some(spark.sparkContext.broadcast(wrapper))
    this
  }

  /** Closes the llama.cpp model backend, freeing resources. Reloaded on next use. */
  def close(): Unit = GGUFWrapper.closeBroadcastModel(_model)

  private[johnsnowlabs] def setEngine(engineName: String): this.type = set(engine, engineName)

  private var _satModel: Option[SentenceDetectorSaTModel] = None

  /** Provide a custom [[SentenceDetectorSaTModel]] for sentence splitting.
    * @group setParam
    */
  def setSatModel(value: SentenceDetectorSaTModel): this.type = {
    _satModel = Some(value)
    this
  }

  /** Returns the held [[SentenceDetectorSaTModel]], lazily downloading the default SaT model the
    * first time it is needed.
    */
  def getSatModel: SentenceDetectorSaTModel = _satModel.getOrElse {
    val model = SentenceDetectorSaTModel.pretrained()
    _satModel = Some(model)
    model
  }

  private val internalDocCol: String = s"__${uid}_doc__"
  private val internalSentenceCol: String = s"__${uid}_sentence__"
  private val internalDocIdCol: String = s"__${uid}_docId__"

  private lazy val columnMetadata: Metadata = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", outputAnnotatorType)
    metadataBuilder.build
  }

  override def transformSchema(schema: StructType): StructType = {
    val outputFields = schema.fields :+
      StructField(getOutputCol, ArrayType(Annotation.dataType), nullable = false, columnMetadata)
    StructType(outputFields)
  }

  /** Fills [[promptTemplate]] for a single sentence. */
  private def fillPrompt(template: String, src: String, tgt: String, text: String): String =
    template
      .replace("{srcLang}", src)
      .replace("{tgtLang}", tgt)
      .replace("{text}", text)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val spark = dataset.sparkSession
    val reader = buildReader2Doc()

    // Step 1: read documents into one DOCUMENT annotation per file and tag each with a stable id.
    val docDf = reader
      .transform(dataset)
      .withColumn(internalDocIdCol, monotonically_increasing_id())
      .persist()
    docDf.count() // force materialization so the document ids stay stable

    try {
      // Step 2: split every document into length-bounded sentences (kept as an array per row).
      val sentenceDf = getSatModel
        .setInputCols(Array(internalDocCol))
        .setOutputCol(internalSentenceCol)
        .setMinSentenceLength($(minSentenceLength))
        .setMaxSentenceLength($(maxSentenceLength))
        .setThreshold($(sentenceThreshold))
        .setExplodeSentences(false)
        .transform(docDf)
        .select(internalDocIdCol, internalSentenceCol)

      // Collect the per-document sentences to the driver for the LLM step.
      val perDocument: Array[(Long, Seq[Annotation])] = sentenceDf.collect().map { row =>
        val docId = row.getLong(0)
        val sentences = Option(row.getAs[Seq[Row]](1))
          .getOrElse(Seq.empty[Row])
          .map(Annotation(_))
        docId -> sentences
      }

      // Step 3: build the per-sentence translation prompts for each document.
      val template = $(promptTemplate)
      val src = $(srcLang)
      val tgt = $(tgtLang)
      val documentPrompts: Seq[Seq[String]] = perDocument.toSeq.map { case (_, sentences) =>
        sentences.map(_.result).filter(_.trim.nonEmpty).map(fillPrompt(template, src, tgt, _))
      }

      // Step 4: translate – all sentences are flattened into one batch and decoded with up to
      // batchSize sentences concurrently, then regrouped per document.
      val documentTranslations: Seq[Seq[String]] = translateDocuments(documentPrompts)

      // Step 5: merge each document's translated sentences into one paragraph.
      val mergedByDoc: Map[Long, Annotation] =
        perDocument.toSeq
          .zip(documentTranslations)
          .map { case ((docId, sentences), translations) =>
            docId -> mergeTranslations(translations, sentences.headOption.map(_.metadata))
          }
          .toMap

      // Step 6: attach the merged translation to every original document row.
      val finalSchema = docDf.schema
        .add(getOutputCol, ArrayType(Annotation.dataType), nullable = false, columnMetadata)
      val docIdIdx = docDf.schema.fieldIndex(internalDocIdCol)

      val finalRows = docDf.collect().map { row =>
        val merged =
          mergedByDoc.getOrElse(row.getLong(docIdIdx), Annotation(DOCUMENT, 0, 0, "", Map.empty))
        val mergedAsRow = Row(
          merged.annotatorType,
          merged.begin,
          merged.end,
          merged.result,
          merged.metadata,
          merged.embeddings)
        Row.fromSeq(row.toSeq :+ Seq(mergedAsRow))
      }

      val finalDf = spark.createDataFrame(spark.sparkContext.parallelize(finalRows), finalSchema)

      finalDf.drop(internalDocCol, internalDocIdCol)
    } finally {
      docDf.unpersist()
    }
  }

  /** Translates the sentences of a batch of documents with llama.cpp, reusing [[AutoGGUFModel]]'s
    * `multiComplete` logic. Every sentence of every document is flattened into a single batch and
    * submitted in one call with `setParallel(batchSize)`, so llama.cpp keeps [[batchSize]]
    * sentences decoding concurrently (refilling a slot as soon as one finishes) regardless of how
    * the sentences are distributed across documents. The flat results are then scattered back
    * into per-document buffers, preserving order.
    *
    * @param documents
    *   one inner sequence of sentence prompts per document
    * @return
    *   one inner sequence of translated sentences per document, aligned with `documents`
    */
  private def translateDocuments(documents: Seq[Seq[String]]): Seq[Seq[String]] = {
    // Flatten all sentences across all documents, remembering where each one came from.
    val flattened: Seq[(Int, Int)] = documents.zipWithIndex.flatMap { case (sentences, docIdx) =>
      sentences.indices.map(sentenceIdx => (docIdx, sentenceIdx))
    }
    if (flattened.isEmpty) return documents.map(_ => Seq.empty[String])

    val modelParams = getModelParameters.setParallel(math.max(1, getBatchSize))
    val inferenceParams = getInferenceParameters
    val model: LlamaModel = getModelIfNotSet.getSession(modelParams)
    val systemPrompt = getSystemPrompt

    val prompts: Array[String] = flattened.map { case (d, s) => documents(d)(s) }.toArray
    val completed: Array[String] =
      try
        processCompletions(
          LlamaExtensions.multiComplete(model, inferenceParams, systemPrompt, prompts))
      catch {
        case e: LlamaException =>
          logger.error("Error in llama.cpp batch completion", e)
          Array.fill(prompts.length)("")
      }

    // Scatter the flat completions back into per-document, per-sentence order.
    val results: Array[Array[String]] = documents.map(d => Array.fill(d.length)("")).toArray
    flattened.zip(completed).foreach { case ((docIdx, sentenceIdx), text) =>
      results(docIdx)(sentenceIdx) = text
    }
    results.map(_.toSeq).toSeq
  }

  /** Merges translated sentences into a single `DOCUMENT` annotation (one paragraph). */
  private def mergeTranslations(
      translations: Seq[String],
      metadata: Option[scala.collection.Map[String, String]]): Annotation = {
    val mergedText = translations.map(_.trim).filter(_.nonEmpty).mkString(" ")
    Annotation(
      DOCUMENT,
      0,
      math.max(0, mergedText.length - 1),
      mergedText,
      metadata.map(_.toMap).getOrElse(Map.empty))
  }

  private def buildReader2Doc(): Reader2Doc = {
    val r = new Reader2Doc()
      .setOutputCol(internalDocCol)
      .setOutputAsDocument($(outputAsDocument))
      .setJoinString($(joinString))

    if ($(contentPath).nonEmpty) r.setContentPath($(contentPath))
    if ($(contentType).nonEmpty) r.setContentType($(contentType))
    if ($(inputCol).nonEmpty) r.setInputCol($(inputCol))
    r
  }

  override def copy(extra: ParamMap): Transformer = {
    val copied = defaultCopy[DocumentTranslator](extra)
    copied._model = _model
    _satModel.foreach(copied.setSatModel)
    copied
  }
}

trait ReadablePretrainedDocumentTranslator extends ParamsAndFeaturesReadable[DocumentTranslator] {

  val defaultModelName: String = "Phi_4_mini_instruct_Q4_K_M_gguf"
  val defaultLang: String = "en"

  def pretrained(): DocumentTranslator =
    pretrained(defaultModelName, defaultLang)

  def pretrained(name: String): DocumentTranslator =
    pretrained(name, defaultLang)

  def pretrained(name: String, lang: String): DocumentTranslator =
    fromAutoGGUF(AutoGGUFModel.pretrained(name, lang))

  def pretrained(name: String, lang: String, remoteLoc: String): DocumentTranslator =
    fromAutoGGUF(AutoGGUFModel.pretrained(name, lang, remoteLoc))

  /** Wraps a downloaded [[AutoGGUFModel]]'s GGUF model in a [[DocumentTranslator]]. */
  private def fromAutoGGUF(autoGGUF: AutoGGUFModel): DocumentTranslator = {
    val translator = new DocumentTranslator()
      .setModelIfNotSet(ResourceHelper.spark, autoGGUF.getModelIfNotSet)
      .setEngine(LlamaCPP.name)
    scala.util
      .Try(autoGGUF.getMetadata)
      .toOption
      .filter(_.nonEmpty)
      .foreach(translator.setMetadata)
    translator
  }
}

trait ReadDocumentTranslatorModel {

  def loadSavedModel(modelPath: String, spark: SparkSession): DocumentTranslator = {
    val localPath: String = ResourceHelper.copyToLocal(modelPath)
    val translator = new DocumentTranslator()
      .setModelIfNotSet(spark, GGUFWrapper.read(spark, localPath))
      .setEngine(LlamaCPP.name)
    val metadata = LlamaExtensions.getMetadataFromFile(localPath)
    if (metadata.nonEmpty) translator.setMetadata(metadata)
    translator
  }
}

/** This is the companion object of [[DocumentTranslator]]. Please refer to that class for the
  * documentation.
  */
object DocumentTranslator
    extends ReadablePretrainedDocumentTranslator
    with ReadDocumentTranslatorModel
