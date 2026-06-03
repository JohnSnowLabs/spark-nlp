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

import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.{Annotation, HasOutputAnnotationCol, HasOutputAnnotatorType}
import com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel
import com.johnsnowlabs.reader.Reader2Doc
import org.apache.spark.ml.{Pipeline, Transformer}
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param, ParamMap}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.slf4j.LoggerFactory

/** DocumentTranslator reads documents from any supported file type and translates them using an
  * [[M2M100Transformer]] model – all in a single Pipeline stage.
  *   1. A [[Reader2Doc]] stage that reads files from `contentPath` (or from a DataFrame column
  *      set via `inputCol`) and converts them to Spark NLP `DOCUMENT` annotations. Every file
  *      format that [[Reader2Doc]] supports is accepted: PDF, Word (.doc/.docx), ODT, Excel
  *      (.xls/.xlsx), PowerPoint (.ppt/.pptx), HTML, plain-text, RTF, Markdown, XML, CSV, and
  *      email (.eml/.msg). 2. An [[M2M100Transformer]] loaded via
  *      [[DocumentTranslator.pretrained]] that translates the resulting document annotations.
  *
  * The output column contains standard Spark NLP `DOCUMENT` annotations holding the translated
  * text, making it fully composable with any downstream Spark NLP annotator.
  *
  * Load a model using the companion object's `pretrained` method, passing the
  * [[M2M100Transformer]] model name and language. Hundreds of language-pair models are available
  * from the [[https://sparknlp.org/models?task=Translation Models Hub]].
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val translator = DocumentTranslator.pretrained("m2m100_418M", "xx")
  *   .setContentPath("src/test/resources/reader/html/")
  *   .setContentType("text/html")
  *   .setSrcLang("en")
  *   .setTgtLang("fr")
  *   .setOutputCol("translation")
  * }}}
  * The default model is `"m2m100_418M"`, default language is `"xx"` (multi-lingual), if no
  * values are provided.
  *
  * For available pretrained M2M100 models please see the
  * [[https://sparknlp.org/models?task=Translation Models Hub]].
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.annotators.seq2seq.DocumentTranslator
  * import org.apache.spark.ml.Pipeline
  * import com.johnsnowlabs.util.PipelineModels
  *
  * val translator = DocumentTranslator.pretrained("m2m100_418M", "xx")
  *   .setContentPath("src/test/resources/reader/html/")
  *   .setContentType("text/html")
  *   .setSrcLang("en")
  *   .setTgtLang("fr")
  *   .setOutputCol("translation")
  *
  * val pipeline = new Pipeline().setStages(Array(translator))
  * val result = pipeline
  *   .fit(PipelineModels.dummyDataset)
  *   .transform(PipelineModels.dummyDataset)
  *
  * result.select("fileName", "translation.result").show(truncate = false)
  * }}}
  *
  * '''Note:''' Translation is computationally expensive. The use of an accelerator (GPU) is
  * recommended for large documents or large batches.
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
    with DefaultParamsWritable
    with HasOutputAnnotatorType
    with HasOutputAnnotationCol {

  def this() = this(Identifiable.randomUID("DOCUMENT_TRANSLATOR"))

  private val logger = LoggerFactory.getLogger(getClass)

  /** Output Annotator Type: DOCUMENT
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

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
    * one `DOCUMENT` annotation before translation. When `false` every extracted element
    * (paragraph, title, table cell, …) becomes its own `DOCUMENT` annotation and is translated
    * independently.
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

  /** Maximum number of tokens per chunk passed to the [[M2M100Transformer]]. The document is
    * first split into sentences by [[com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel]], then whole sentences are greedily packed
    * into chunks that stay under this token limit (Default: `400`).
    *
    * @group param
    */
  val chunkSize: IntParam =
    new IntParam(
      this,
      "chunkSize",
      "Maximum number of tokens per chunk sent to the M2M100Transformer (Default: 400).")

  /** @group setParam */
  def setChunkSize(value: Int): this.type = {
    require(value > 0, "chunkSize must be > 0.")
    set(chunkSize, value)
  }

  /** @group getParam */
  def getChunkSize: Int = $(chunkSize)

  setDefault(
    contentPath -> "",
    contentType -> "",
    inputCol -> "",
    outputAsDocument -> true,
    joinString -> "\n",
    chunkSize -> 400)

  /** The [[M2M100Transformer]] model used for translation. Set by
    * [[DocumentTranslator.pretrained]] via [[setModel]] and not exposed as a public Spark ML
    * param because the model object itself is not serialisable as a param value.
    */
  private var _model: Option[M2M100Transformer] = None

  /** Stores the downloaded [[M2M100Transformer]] model inside this annotator. Called exclusively
    * by the companion object's `pretrained` factory methods.
    */
  private[seq2seq] def setModel(value: M2M100Transformer): this.type = {
    _model = Some(value)
    this
  }

  /** Returns the [[M2M100Transformer]] model, throwing [[IllegalStateException]] if
    * [[DocumentTranslator.pretrained]] has not been called yet.
    */
  def getModel: M2M100Transformer =
    _model.getOrElse(
      throw new IllegalStateException(
        s"[$uid] No M2M100Transformer model is loaded. " +
          "Use DocumentTranslator.pretrained(name, lang) to obtain a configured instance."))

  // ── M2M100-specific language parameters ───────────────────────────────────

  /** Source language code for the M2M100 model (e.g. `"en"`).
    * @group setParam
    */
  def setSrcLang(value: String): this.type = { getModel.setSrcLang(value); this }

  /** @group getParam */
  def getSrcLang: String = getModel.getOrDefault(getModel.srcLang)

  /** Target language code for the M2M100 model (e.g. `"fr"`).
    * @group setParam
    */
  def setTgtLang(value: String): this.type = { getModel.setTgtLang(value); this }

  /** @group getParam */
  def getTgtLang: String = getModel.getOrDefault(getModel.tgtLang)

  // ── Generation parameters delegated to HasGeneratorProperties ─────────────

  /** Maximum length of the input sequence.
    * @group setParam
    */
  def setMaxInputLength(value: Int): this.type = { getModel.setMaxInputLength(value); this }

  /** @group getParam */
  def getMaxInputLength: Int = getModel.getOrDefault(getModel.maxInputLength)

  /** Maximum length of the generated (translated) sequence.
    * @group setParam
    */
  def setMaxOutputLength(value: Int): this.type = { getModel.setMaxOutputLength(value); this }

  /** @group getParam */
  def getMaxOutputLength: Int = getModel.getMaxOutputLength

  /** Minimum length of the generated sequence.
    * @group setParam
    */
  def setMinOutputLength(value: Int): this.type = { getModel.setMinOutputLength(value); this }

  /** @group getParam */
  def getMinOutputLength: Int = getModel.getMinOutputLength

  /** Whether to use sampling; use greedy decoding otherwise (Default: `false`).
    * @group setParam
    */
  def setDoSample(value: Boolean): this.type = { getModel.setDoSample(value); this }

  /** @group getParam */
  def getDoSample: Boolean = getModel.getDoSample

  /** The value used to module the next token probabilities (Default: `1.0`).
    * @group setParam
    */
  def setTemperature(value: Double): this.type = { getModel.setTemperature(value); this }

  /** @group getParam */
  def getTemperature: Double = getModel.getTemperature

  /** The number of highest probability vocabulary tokens to keep for top-k-filtering (Default: `50`).
    * @group setParam
    */
  def setTopK(value: Int): this.type = { getModel.setTopK(value); this }

  /** @group getParam */
  def getTopK: Int = getModel.getTopK

  /** If set to float < `1.0`, only the most probable tokens with probabilities that add up to
    * `topP` or higher are kept for generation (Default: `1.0`).
    * @group setParam
    */
  def setTopP(value: Double): this.type = { getModel.setTopP(value); this }

  /** @group getParam */
  def getTopP: Double = getModel.getTopP

  /** The parameter for repetition penalty (Default: `1.0`). `1.0` means no penalty.
    * @group setParam
    */
  def setRepetitionPenalty(value: Double): this.type = {
    getModel.setRepetitionPenalty(value); this
  }

  /** @group getParam */
  def getRepetitionPenalty: Double = getModel.getRepetitionPenalty

  /** If set to int > `0`, all ngrams of that size can only occur once (Default: `0`).
    * @group setParam
    */
  def setNoRepeatNgramSize(value: Int): this.type = { getModel.setNoRepeatNgramSize(value); this }

  /** @group getParam */
  def getNoRepeatNgramSize: Int = getModel.getNoRepeatNgramSize

  /** A list of token ids which are ignored in the decoder's output (Default: `Array()`).
    * @group setParam
    */
  def setIgnoreTokenIds(tokenIds: Array[Int]): this.type = {
    getModel.setIgnoreTokenIds(tokenIds); this
  }

  /** @group getParam */
  def getIgnoreTokenIds: Array[Int] = getModel.getIgnoreTokenIds

  private val internalDocCol: String      = s"__${uid}_doc__"
  private val internalSentenceCol: String = s"__${uid}_sentence__"
  private val internalChunkCol: String    = s"__${uid}_chunk__"

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

  override def transform(dataset: Dataset[_]): DataFrame = {
    val spark  = dataset.sparkSession
    val reader = buildReader2Doc()

    val sentenceDetector = SentenceDetectorDLModel
      .pretrained("sentence_detector_dl", "xx")
      .setInputCols(Array(internalDocCol))
      .setOutputCol(internalSentenceCol)

    val m2m100 = getModel
      .setInputCols(Array(internalChunkCol))
      .setOutputCol(getOutputCol)

    // Step 1 & 2: read documents and split into sentences via a mini-pipeline
    val sentencePipeline = new Pipeline().setStages(Array(reader, sentenceDetector))
    val withSentences    = sentencePipeline.fit(dataset).transform(dataset)

    // Step 3: greedy sentence packing — runs on the driver, returns rows with chunk column
    val schemaWithChunks = withSentences.schema
      .add(internalChunkCol, ArrayType(Annotation.dataType), nullable = false, columnMetadata)

    val packedRows = withSentences.collect().map { row =>
      val sentences = row.getAs[Seq[org.apache.spark.sql.Row]](internalSentenceCol)
        .map(Annotation(_))
      val chunks = packSentences(sentences, $(chunkSize)).map { a =>
        org.apache.spark.sql.Row(a.annotatorType, a.begin, a.end, a.result, a.metadata, a.embeddings)
      }
      org.apache.spark.sql.Row.fromSeq(row.toSeq :+ chunks)
    }

    val withChunks = spark.createDataFrame(
      spark.sparkContext.parallelize(packedRows), schemaWithChunks)

    // Step 4: translate — M2M100Transformer reads internalChunkCol, writes getOutputCol
    val translated = m2m100.transform(withChunks)

    // Step 5: merge translated chunk annotations into one annotation per row
    val outputColIdx = translated.schema.fieldIndex(getOutputCol)
    val mergedRows = translated.collect().map { row =>
      val chunks = row.getAs[Seq[org.apache.spark.sql.Row]](getOutputCol).map(Annotation(_))
      val merged = mergeChunks(chunks)
      val mergedAsRow = org.apache.spark.sql.Row(
        merged.annotatorType, merged.begin, merged.end,
        merged.result, merged.metadata, merged.embeddings)
      val newValues = row.toSeq.zipWithIndex.map { case (v, i) =>
        if (i == outputColIdx) Seq(mergedAsRow) else v
      }
      org.apache.spark.sql.Row.fromSeq(newValues)
    }

    val finalDf = spark.createDataFrame(
      spark.sparkContext.parallelize(mergedRows), translated.schema)

    finalDf.drop(internalDocCol, internalSentenceCol, internalChunkCol)
  }

  /** Greedily packs whole sentences into chunks whose whitespace-token count stays ≤ maxTokens.
    * Sentences already over the limit are passed as-is with a warning.
    */
  private def packSentences(sentences: Seq[Annotation], maxTokens: Int): Seq[Annotation] = {
    if (sentences.isEmpty) return Seq.empty
    val result = scala.collection.mutable.ArrayBuffer.empty[Annotation]
    var texts  = scala.collection.mutable.ArrayBuffer.empty[String]
    var tokens = 0
    var begin  = sentences.head.begin

    // M2M100 uses SentencePiece subword tokenisation which produces significantly more
    // tokens than whitespace splitting. A character-based estimate (÷ 4) is a much
    // closer approximation of the actual subword token count and avoids OOM errors in
    // the ONNX decoder when chunks are larger than the model's positional embedding size.
    def tokenCount(s: String): Int = math.max(1, (s.length / 4.0).ceil.toInt)

    def flush(end: Int, meta: scala.collection.Map[String, String]): Unit =
      if (texts.nonEmpty) {
        result += Annotation(DOCUMENT, begin, end, texts.mkString(" "), meta)
        texts   = scala.collection.mutable.ArrayBuffer.empty[String]
        tokens  = 0
      }

    for (ann <- sentences) {
      val n = tokenCount(ann.result)
      if (n > maxTokens) {
        logger.warn(
          s"[$uid] Sentence with $n tokens exceeds chunkSize=$maxTokens. " +
            s"Sentence: [${ann.result}]")
        flush(ann.begin - 1, ann.metadata)
        result += Annotation(DOCUMENT, ann.begin, ann.end, ann.result, ann.metadata)
        begin   = ann.end + 1
      } else {
        if (texts.nonEmpty && tokens + n > maxTokens) {
          flush(ann.begin - 1, ann.metadata)
          begin = ann.begin
        }
        texts  += ann.result
        tokens += n
      }
    }
    flush(sentences.last.end, sentences.last.metadata)
    result
  }

  /** Merges translated chunk annotations into a single annotation by joining results with newline. */
  private def mergeChunks(chunks: Seq[Annotation]): Annotation =
    if (chunks.isEmpty) Annotation(DOCUMENT, 0, 0, "", Map.empty)
    else Annotation(
      DOCUMENT,
      chunks.head.begin,
      chunks.last.end,
      chunks.map(_.result).mkString("\n"),
      chunks.head.metadata)

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
    _model.foreach(copied.setModel)
    copied
  }
}

trait ReadablePretrainedDocumentTranslator extends DefaultParamsReadable[DocumentTranslator] {

  val defaultModelName: String = "m2m100_418M"
  val defaultLang: String = "xx"

  def pretrained(): DocumentTranslator =
    pretrained(defaultModelName, defaultLang)

  def pretrained(name: String): DocumentTranslator =
    pretrained(name, defaultLang)

  def pretrained(name: String, lang: String): DocumentTranslator = {
    val m2m100 = M2M100Transformer.pretrained(name, lang)
    new DocumentTranslator().setModel(m2m100)
  }

  def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): DocumentTranslator = {
    val m2m100 = M2M100Transformer.pretrained(name, lang, remoteLoc)
    new DocumentTranslator().setModel(m2m100)
  }
}

trait ReadDocumentTranslatorModel {

  def loadSavedModel(
      modelPath: String,
      spark: SparkSession,
      useOpenvino: Boolean = false): DocumentTranslator = {
    val m2m100 = M2M100Transformer.loadSavedModel(modelPath, spark, useOpenvino)
    new DocumentTranslator().setModel(m2m100)
  }
}

/** This is the companion object of [[DocumentTranslator]]. Please refer to that class for the
  * documentation.
  */
object DocumentTranslator
    extends ReadablePretrainedDocumentTranslator
    with ReadDocumentTranslatorModel

