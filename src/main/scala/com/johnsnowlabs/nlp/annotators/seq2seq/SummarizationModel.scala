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

import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.MPNetEmbeddings
import com.johnsnowlabs.nlp.{Annotation, CanBeLazy, ParamsAndFeaturesReadable, RawAnnotator}
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.{Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{Metadata, MetadataBuilder, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.slf4j.LoggerFactory

/** Fitted model produced by [[Summarization]].
  *
  * Orchestrates the resolved summarization delegate at the DataFrame level: prompt building,
  * long-document chunking (hierarchical map/reduce summarization), delegate inference, output
  * cleanup and transparency metadata. The delegate is one of:
  *
  *   - [[AutoGGUFModel]] for method `llm`,
  *   - [[BartTransformer]] for method `encoder_decoder`,
  *   - [[MPNetEmbeddings]] (+ rule-based sentence detection and a PacSum-style ranker) for method
  *     `extractive`.
  *
  * Saving this model persists the delegate (model weights included) under the model path, so
  * fitted pipelines reload without network access.
  *
  * Notes:
  *   - The summarization `method` is bound at fit time: this model only holds the delegate of the
  *     method it was fitted with. Changing `method` on a fitted model fails at transform with an
  *     explanatory error; fit a new [[Summarization]] stage instead. All other task parameters
  *     (lengths, style, focus, strategy, ...) can be changed freely after fit.
  *   - As a DataFrame-level orchestrator this annotator is not supported in LightPipeline.
  *   - `transform` configures the shared delegate (input/output columns, generation settings) in
  *     place, so a single model instance is not safe for concurrent `transform` calls; use one
  *     instance per concurrent caller.
  *   - Transforms cache the input row ids and the computed summaries for the lifetime of the
  *     returned DataFrame (generative inference intermediates are released eagerly). These caches
  *     back the returned DataFrame and cannot be unpersisted through this API; they are freed
  *     when the SparkSession ends or via `spark.catalog.clearCache()` (which clears all cached
  *     data). The encoder_decoder method pins inference to a single partition (the BART backend
  *     is not thread-safe); the llm and extractive methods keep the input partitioning.
  *   - For the `llm` method the document text is embedded into the prompt, so a document
  *     containing instructions can influence its own summary (prompt injection); the system
  *     prompt reduces but does not eliminate this.
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
class SummarizationModel(override val uid: String)
    extends RawAnnotator[SummarizationModel]
    with CanBeLazy
    with SummarizationParams {

  def this() = this(Identifiable.randomUID("SUMMARIZATION_MODEL"))

  /** Input Annotator Type: DOCUMENT
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)

  /** Output Annotator Type: DOCUMENT
    * @group anno
    */
  override val outputAnnotatorType: String = DOCUMENT

  private val logger = LoggerFactory.getLogger(classOf[SummarizationModel])

  /** Name of the pretrained model resolved at fit time (for transparency metadata).
    * @group param
    */
  val resolvedModel =
    new Param[String](this, "resolvedModel", "Pretrained model name resolved at fit time")

  /** @group setParam */
  def setResolvedModel(value: String): this.type = set(resolvedModel, value)

  /** @group getParam */
  def getResolvedModel: String = get(resolvedModel).getOrElse("")

  setDefault(resolvedModel -> "")

  // ------------------------------------------------------------------------------------------
  // Delegates (not Spark params; serialized via onWrite / companion reader)
  // ------------------------------------------------------------------------------------------

  private var _llmDelegate: Option[AutoGGUFModel] = None
  private var _seq2SeqDelegate: Option[BartTransformer] = None
  private var _embeddingsDelegate: Option[MPNetEmbeddings] = None

  /** @group setParam */
  def setLlmDelegate(value: AutoGGUFModel): this.type = {
    _llmDelegate = Some(value)
    this
  }

  /** @group setParam */
  def setSeq2SeqDelegate(value: BartTransformer): this.type = {
    _seq2SeqDelegate = Some(value)
    this
  }

  /** @group setParam */
  def setEmbeddingsDelegate(value: MPNetEmbeddings): this.type = {
    _embeddingsDelegate = Some(value)
    this
  }

  /** @group getParam */
  def getLlmDelegate: AutoGGUFModel =
    _llmDelegate.getOrElse(
      throw new IllegalStateException(
        "No LLM delegate set. Fit a Summarization stage with method 'llm' first."))

  /** @group getParam */
  def getSeq2SeqDelegate: BartTransformer =
    _seq2SeqDelegate.getOrElse(
      throw new IllegalStateException(
        "No encoder_decoder delegate set. Fit a Summarization stage with method " +
          "'encoder_decoder' first."))

  /** @group getParam */
  def getEmbeddingsDelegate: MPNetEmbeddings =
    _embeddingsDelegate.getOrElse(
      throw new IllegalStateException(
        "No embeddings delegate set. Fit a Summarization stage with method 'extractive' first."))

  /** Closes the llama.cpp backend if the llm delegate is loaded, freeing native resources. */
  def close(): Unit = _llmDelegate.foreach(_.close())

  // ------------------------------------------------------------------------------------------
  // Internal column names
  // ------------------------------------------------------------------------------------------

  private val rowIdCol = s"__${uid}_rowId"
  private val annIdxCol = s"__${uid}_annIdx"
  private val textCol = s"__${uid}_text"
  private val srcMetaCol = s"__${uid}_srcMeta"
  private val origLenCol = s"__${uid}_origLen"
  private val chunksCol = s"__${uid}_chunks"
  private val nChunksCol = s"__${uid}_nChunks"
  private val totalChunksCol = s"__${uid}_totalChunks"
  private val chunkIdxCol = s"__${uid}_chunkIdx"
  private val chunkTextCol = s"__${uid}_chunkText"
  private val delegateInCol = s"__${uid}_delegateIn"
  private val delegateOutCol = s"__${uid}_delegateOut"
  private val summaryTextCol = s"__${uid}_summaryText"
  private val pairsCol = s"__${uid}_pairs"
  private val sentCol = s"__${uid}_sentences"
  private val embCol = s"__${uid}_sentEmbeddings"
  private val singleDocCol = s"__${uid}_singleDoc"
  private val resultCol = s"__${uid}_resultAnns"

  private def internalColumns: Seq[String] = Seq(
    rowIdCol,
    annIdxCol,
    textCol,
    srcMetaCol,
    origLenCol,
    chunksCol,
    nChunksCol,
    totalChunksCol,
    chunkIdxCol,
    chunkTextCol,
    delegateInCol,
    delegateOutCol,
    summaryTextCol,
    pairsCol,
    sentCol,
    embCol,
    singleDocCol,
    resultCol)

  private def documentColMetadata: Metadata = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", DOCUMENT)
    metadataBuilder.build
  }

  // ------------------------------------------------------------------------------------------
  // Budgets and prompts (driver-side; UDFs only capture local immutable values)
  // ------------------------------------------------------------------------------------------

  /** Approximate chars-per-token factor used for context estimates. */
  private val charsPerToken = 4

  private def promptOverheadTokens: Int = 150

  /** Effective per-pass input budget in characters for the generative delegate. */
  private def budgetChars: Int = {
    val tokens =
      if ($(chunkSize) > 0) $(chunkSize)
      else
        $(method) match {
          case "llm" =>
            // keep prompt + generation inside the context window
            val ctx = math.max(getLlmDelegate.getNCtx, 4096)
            math.max(256, (ctx * 0.6).toInt - promptOverheadTokens - $(maxSummaryLength))
          case _ =>
            // stay safely inside the delegate's encoder length (TF-exported BART graphs are
            // traced for maxInputLength positions, default 512; exceeding it is a graph error)
            val delegate = getSeq2SeqDelegate
            val maxIn = delegate.getOrDefault(delegate.maxInputLength)
            math.max(128, maxIn - 64)
        }
    tokens * charsPerToken
  }

  private def systemPromptText: String =
    "/no_think\n" +
      "You are a professional text summarization engine. " +
      "Write only the requested summary of the user's text. " +
      "Do not include reasoning, explanations, analysis, notes, markdown headers, labels, " +
      "or <think> tags. Never refuse; always produce a summary."

  private def styleClause: String = $(summaryStyle) match {
    case "detailed" => "Write a detailed summary covering all major points."
    case "bullets" => "Write the summary as short bullet points, one key point per line."
    case _ => "Write a concise summary."
  }

  private def focusClause: String =
    if ($(focus).nonEmpty) s" Focus on: ${$(focus)}." else ""

  private def engineName: String = $(method) match {
    case "llm" => "llamacpp"
    case "encoder_decoder" => getSeq2SeqDelegate.getEngine
    case "extractive" => getEmbeddingsDelegate.getEngine
  }

  private def configureLlm(): AutoGGUFModel = {
    val delegate = getLlmDelegate
    if (delegate.getNCtx < 4096) delegate.setNCtx(4096)
    delegate
      .setInputCols(delegateInCol)
      .setOutputCol(delegateOutCol)
      .setUseChatTemplate(true)
      .setSystemPrompt(systemPromptText)
      .setTemperature($(temperature))
      .setTopP($(topP))
      .setRepeatPenalty(1.1f)
      .setRepeatLastN(64)
      .setNPredict(math.max(64, $(maxSummaryLength) * 2))
      .setNGpuLayers($(gpuLayers))
      .setReasoningBudget(0)
      .setRemoveThinkingTag("think")
  }

  private def configureSeq2Seq(): BartTransformer = {
    getSeq2SeqDelegate
      .setInputCols(delegateInCol)
      .setOutputCol(delegateOutCol)
      .setMaxOutputLength($(maxSummaryLength))
      .setMinOutputLength(math.max(0, math.min($(minSummaryLength), $(maxSummaryLength) - 1)))
      .setBeamSize($(numBeams))
      .setDoSample(false)
      .setNoRepeatNgramSize($(noRepeatNgramSize))
  }

  // ------------------------------------------------------------------------------------------
  // Transform
  // ------------------------------------------------------------------------------------------

  override def transform(dataset: Dataset[_]): DataFrame = {
    require(
      validate(dataset.schema),
      s"Wrong or missing inputCols annotators in $uid. Received inputCols: " +
        s"${getInputCols.mkString(",")}. Make sure such annotators exist in your pipeline " +
        s"and have the annotator type: ${inputAnnotatorTypes.mkString(", ")}")

    $(method) match {
      case "extractive" => transformExtractive(dataset.toDF())
      case "llm" | "encoder_decoder" => transformGenerative(dataset.toDF())
      case other => throw new IllegalArgumentException(s"Unsupported method '$other'")
    }
  }

  // ------------------------------------------------------------------------------------------
  // Generative path (llm / encoder_decoder)
  // ------------------------------------------------------------------------------------------

  private def transformGenerative(df: DataFrame): DataFrame = {
    val isLlm = $(method) == "llm"
    val delegate = if (isLlm) configureLlm() else configureSeq2Seq()
    val strategy = $(longDocumentStrategy)
    val budget = budgetChars
    val overlap = $(chunkOverlap)
    val maxRounds = if (strategy == "truncate") 1 else 3

    // local copies for UDF closures (avoid serializing the model into tasks)
    val style = styleClause
    val focusHint = focusClause
    val maxWords = $(maxSummaryLength)

    val minWords = $(minSummaryLength)

    def buildPromptLocal(text: String, isReduce: Boolean): String = {
      val src =
        if (isReduce) "the following partial summaries of a longer document"
        else "the following text"
      // task params are mutable on the fitted model, so minWords can exceed maxWords here even
      // though the estimator rejects that at fit; clamp to keep the prompt coherent
      val effMin = math.max(0, math.min(minWords, maxWords))
      val lengthClause =
        if (effMin > 0 && effMin < maxWords)
          s"The summary must be between $effMin and $maxWords words."
        else s"The summary must be at most $maxWords words."
      s"$style$focusHint $lengthClause Summarize $src:\n\n$text\n\nSummary:"
    }

    val chunkFn = (budgetC: Int, strat: String, lastRound: Boolean, overlapS: Int) =>
      udf { text: String =>
        val t = Option(text).getOrElse("")
        if (t.length <= budgetC) Seq(t)
        else
          strat match {
            case "truncate" => Seq(t.substring(0, budgetC))
            case _ if lastRound => Seq(t.substring(0, budgetC))
            case _ =>
              val sentences = ExtractiveSummarizationRanker.splitSentences(t)
              ExtractiveSummarizationRanker.packChunks(sentences, budgetC, overlapS).toSeq
          }
      }

    val mkDocUdf = udf { text: String =>
      val t = Option(text).getOrElse("")
      if (t.trim.isEmpty) Seq.empty[Annotation]
      else Seq(Annotation(DOCUMENT, 0, t.length - 1, t, Map.empty[String, String]))
    }

    val promptUdf = udf { (t: String, reduceRound: Boolean) =>
      val text = Option(t).getOrElse("")
      if (text.trim.isEmpty) ""
      else if (isLlm) buildPromptLocal(text, reduceRound)
      else text
    }

    val extractSummaryUdf = udf { anns: Seq[Annotation] =>
      val raw =
        if (anns == null || anns.isEmpty) ""
        else anns.head.result
      // remove complete <think>...</think> blocks first, then any dangling unclosed <think>
      // (Qwen and similar reasoning models can emit an opening tag whose closing tag was cut
      // off by the generation budget; strip from the opening tag to the end of the output)
      var stripped = raw.replaceAll("(?s)<think>.*?</think>", "")
      val openIdx = stripped.indexOf("<think>")
      if (openIdx >= 0) stripped = stripped.substring(0, openIdx)
      if (isLlm) {
        val withoutLabel =
          stripped.replaceAll("(?im)^\\s*(summary|here is (a|the) summary)\\s*:?\\s*", "").trim
        SummarizationModel.unwrapJsonStringLiteral(withoutLabel)
      } else stripped.trim
    }

    val joinPairsUdf = udf { pairs: Seq[(Int, String)] =>
      pairs
        .sortBy(_._1)
        .map(_._2)
        .filter(_.nonEmpty)
        .mkString(" ")
    }

    val withId = df.withColumn(rowIdCol, monotonically_increasing_id()).persist()
    withId.count() // pin row ids

    // one work row per input DOCUMENT annotation; posexplode (not posexplode_outer) so a row
    // whose input annotation array is empty/null produces no work row and, via the closing left
    // join, an empty output array (empty in -> empty out, matching the extractive path)
    var pending: DataFrame = withId
      .select(col(rowIdCol), posexplode(col(getInputCols.head)).as(Seq(annIdxCol, "__ann")))
      .select(
        col(rowIdCol),
        col(annIdxCol),
        coalesce(col("__ann.result"), lit("")).as(textCol),
        coalesce(col("__ann.metadata"), typedLit(Map.empty[String, String])).as(srcMetaCol))
      .withColumn(origLenCol, length(col(textCol)))
      .withColumn(totalChunksCol, lit(0))

    var done: Option[DataFrame] = None
    var round = 0
    // intermediate per-round caches; released once the final result is materialized
    val roundCaches = scala.collection.mutable.ArrayBuffer.empty[DataFrame]

    try {
      while (round < maxRounds && pending != null) {
        val lastRound = round == maxRounds - 1

        val chunked = pending
          .withColumn(chunksCol, chunkFn(budget, strategy, lastRound, overlap)(col(textCol)))
          .withColumn(nChunksCol, size(col(chunksCol)))

        val exploded = chunked.select(
          col(rowIdCol),
          col(annIdxCol),
          col(srcMetaCol),
          col(origLenCol),
          col(totalChunksCol),
          col(nChunksCol),
          posexplode_outer(col(chunksCol)).as(Seq(chunkIdxCol, chunkTextCol)))

        // The encoder-decoder (BART) backend keeps mutable per-instance decoder tensors that are
        // not thread-safe; running it across several Spark partitions lets concurrent tasks share
        // one broadcast model and corrupt each other's decode state (TF shape mismatches). Pin the
        // BART delegate stage to a single partition so inference is single-threaded. The llm
        // (llama.cpp) delegate has no such constraint, so it keeps the input partitioning and its
        // cluster parallelism. Lane-level parallelism (one method per JVM) still keeps the machine
        // busy in the pinned case.
        val promptedBase = exploded
          .withColumn(
            delegateInCol,
            mkDocUdf(promptUdf(col(chunkTextCol), lit(round > 0)))
              .as(delegateInCol, documentColMetadata))
        val prompted = if (isLlm) promptedBase else promptedBase.coalesce(1)

        val summarized = delegate
          .transform(prompted)
          .withColumn(summaryTextCol, extractSummaryUdf(col(delegateOutCol)))
          .persist()
        summarized.count() // materialize: avoid re-running inference on branch reuse
        roundCaches += summarized

        val perAnnotation = summarized
          .groupBy(col(rowIdCol), col(annIdxCol))
          .agg(
            // joinPairsUdf re-sorts by chunk index, so no sort_array is needed here
            collect_list(struct(col(chunkIdxCol), col(summaryTextCol))).as(pairsCol),
            first(col(srcMetaCol)).as(srcMetaCol),
            first(col(origLenCol)).as(origLenCol),
            first(col(totalChunksCol)).as(totalChunksCol),
            first(col(nChunksCol)).as(nChunksCol))
          .withColumn(summaryTextCol, joinPairsUdf(col(pairsCol)))
          .withColumn(
            totalChunksCol,
            when(col(totalChunksCol) === 0, col(nChunksCol)).otherwise(col(totalChunksCol)))
          .drop(pairsCol)

        val finishedThisRound = perAnnotation
          .filter(col(nChunksCol) === 1)
          .select(rowIdCol, annIdxCol, summaryTextCol, srcMetaCol, origLenCol, totalChunksCol)

        done = done match {
          case Some(d) => Some(d.union(finishedThisRound))
          case None => Some(finishedThisRound)
        }

        val stillPending = perAnnotation.filter(col(nChunksCol) > 1)
        // early exit: when nothing needs another reduce round (the common single-chunk case),
        // do not schedule further delegate passes over empty frames
        val hasPending = round + 1 < maxRounds && !stillPending.isEmpty
        pending =
          if (hasPending)
            stillPending.select(
              col(rowIdCol),
              col(annIdxCol),
              col(summaryTextCol).as(textCol),
              col(srcMetaCol),
              col(origLenCol),
              col(totalChunksCol))
          else null

        round += 1
      }
    } catch {
      case t: Throwable =>
        (roundCaches :+ withId).foreach(f => scala.util.Try(f.unpersist()))
        throw t
    }

    val summaries = done.get

    // build final DOCUMENT annotations and group them back per original row
    val methodName = $(method)
    val modelName = getResolvedModel
    val engine = engineName
    val strategyName = strategy
    val cpt = charsPerToken

    val mkResultUdf = udf {
      (summary: String, srcMeta: Map[String, String], origLen: Int, chunks: Int) =>
        val text = Option(summary).getOrElse("")
        // source metadata first so this stage's keys win on collision (e.g. chained summarizers)
        val metadata = Option(srcMeta).getOrElse(Map.empty[String, String]) ++ Map(
          "method" -> methodName,
          "model" -> modelName,
          "engine" -> engine,
          "longDocumentStrategy" -> strategyName,
          "numChunks" -> math.max(chunks, 1).toString,
          "originalTokensEst" -> (origLen / cpt).toString,
          "summaryTokensEst" -> (text.length / cpt).toString)
        Annotation(DOCUMENT, 0, math.max(0, text.length - 1), text, metadata)
    }

    // annotation structs contain a map field and are not orderable, so ordering by annotation
    // index must happen inside a UDF rather than via sort_array
    val sortAnnotationsUdf = udf { pairs: Seq[(Int, Annotation)] =>
      pairs.sortBy(_._1).map(_._2)
    }

    val resultPerRow = summaries
      .withColumn(
        resultCol,
        mkResultUdf(col(summaryTextCol), col(srcMetaCol), col(origLenCol), col(totalChunksCol)))
      .groupBy(col(rowIdCol))
      .agg(collect_list(struct(col(annIdxCol), col(resultCol))).as(pairsCol))
      .withColumn(resultCol, sortAnnotationsUdf(col(pairsCol)))
      .select(col(rowIdCol), col(resultCol))
      .persist()
    resultPerRow.count() // materialize so the per-round inference caches can be released

    // the summaries are now safely cached in resultPerRow; free the inference intermediates
    roundCaches.foreach(f => scala.util.Try(f.unpersist()))

    val joined = withId
      .join(resultPerRow, Seq(rowIdCol), "left")
      .withColumn(
        getOutputCol,
        wrapColumnMetadata(coalesce(col(resultCol), typedLit(Seq.empty[Annotation]))))

    joined.drop(internalColumns: _*)
  }

  // ------------------------------------------------------------------------------------------
  // Extractive path
  // ------------------------------------------------------------------------------------------

  private def transformExtractive(df: DataFrame): DataFrame = {
    val budgetWords = $(maxSummaryLength)
    val lambda = $(mmrLambda).toDouble
    val posBias = $(positionBias).toDouble
    val methodName = $(method)
    val modelName = getResolvedModel
    val engine = engineName
    val cpt = charsPerToken

    val withId = df.withColumn(rowIdCol, monotonically_increasing_id()).persist()
    withId.count() // pin row ids for the closing join

    // Rebuild each input annotation as its own single-DOCUMENT row before sentence detection.
    // Sentence detectors reset character offsets per document, so several DOCUMENT annotations in
    // one row (e.g. from MultiDocumentAssembler, where every document begins at offset 0) would
    // otherwise overlap and a begin/end containment match could pull one document's sentences
    // into another's summary. Scoping ranking to one document per row removes that ambiguity.
    val mkSingleDocUdf = udf { (text: String, meta: Map[String, String]) =>
      val t = Option(text).getOrElse("")
      Seq(
        Annotation(DOCUMENT, 0, math.max(0, t.length - 1), t, Option(meta).getOrElse(Map.empty)))
    }

    val exploded = withId
      .select(col(rowIdCol), posexplode(col(getInputCols.head)).as(Seq(annIdxCol, "__ann")))
      .withColumn(textCol, coalesce(col("__ann.result"), lit("")))
      .withColumn(
        srcMetaCol,
        coalesce(col("__ann.metadata"), typedLit(Map.empty[String, String])))
      .withColumn(
        singleDocCol,
        mkSingleDocUdf(col(textCol), col(srcMetaCol)).as(singleDocCol, documentColMetadata))
      .drop("__ann")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(singleDocCol)
      .setOutputCol(sentCol)
    val embeddings = getEmbeddingsDelegate
      .setInputCols(sentCol)
      .setOutputCol(embCol)

    val withSentences = sentenceDetector.transform(exploded)
    val withEmbeddings = embeddings.transform(withSentences)

    // Rank the sentences of exactly one document (this row) into a single summary annotation.
    // Sentence and embedding annotations are aligned 1:1 by index within the row.
    val rankUdf = udf {
      (docAnns: Seq[Annotation], sentAnns: Seq[Annotation], embAnns: Seq[Annotation]) =>
        val doc = Option(docAnns).getOrElse(Seq.empty).headOption
        val sents = Option(sentAnns).getOrElse(Seq.empty).toArray
        val embs = Option(embAnns).getOrElse(Seq.empty).toArray

        val docText = doc.map(_.result).getOrElse("")
        val docMeta = doc.map(_.metadata).getOrElse(Map.empty[String, String])
        val sentTexts = sents.map(_.result)
        val sentEmbs = sentTexts.indices.map { i =>
          if (i < embs.length) embs(i).embeddings else Array.emptyFloatArray
        }.toArray

        // source metadata first so this stage's keys win on collision (e.g. chained summarizers)
        val baseMeta = docMeta ++ Map(
          "method" -> methodName,
          "model" -> modelName,
          "engine" -> engine,
          "longDocumentStrategy" -> "native")

        if (sentTexts.isEmpty || docText.trim.isEmpty) {
          Annotation(DOCUMENT, 0, 0, "", baseMeta)
        } else {
          val scores =
            ExtractiveSummarizationRanker.centralityScores(sentEmbs, positionBias = posBias)
          val wordCounts = sentTexts.map(ExtractiveSummarizationRanker.countWords)
          val selected = ExtractiveSummarizationRanker.mmrSelect(
            sentEmbs,
            scores,
            wordCounts,
            budgetWords,
            lambda)
          val summary = selected.map(sentTexts(_)).mkString(" ")
          val metadata = baseMeta ++ Map(
            "sentencesSelected" -> selected.length.toString,
            "sentencesTotal" -> sentTexts.length.toString,
            "originalTokensEst" -> (docText.length / cpt).toString,
            "summaryTokensEst" -> (summary.length / cpt).toString)
          Annotation(DOCUMENT, 0, math.max(0, summary.length - 1), summary, metadata)
        }
    }

    val ranked = withEmbeddings
      .withColumn(resultCol, rankUdf(col(singleDocCol), col(sentCol), col(embCol)))

    // annotation structs contain a map field and are not orderable, so ordering by annotation
    // index must happen inside a UDF rather than via sort_array
    val sortAnnotationsUdf = udf { pairs: Seq[(Int, Annotation)] =>
      pairs.sortBy(_._1).map(_._2)
    }

    val resultPerRow = ranked
      .groupBy(col(rowIdCol))
      .agg(collect_list(struct(col(annIdxCol), col(resultCol))).as(pairsCol))
      .withColumn(resultCol, sortAnnotationsUdf(col(pairsCol)))
      .select(col(rowIdCol), col(resultCol))

    withId
      .join(resultPerRow, Seq(rowIdCol), "left")
      .withColumn(
        getOutputCol,
        wrapColumnMetadata(coalesce(col(resultCol), typedLit(Seq.empty[Annotation]))))
      .drop(internalColumns: _*)
  }

  // ------------------------------------------------------------------------------------------
  // Persistence
  // ------------------------------------------------------------------------------------------

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    val delegatePath = new Path(path, SummarizationModel.delegateDir).toString
    $(method) match {
      case "llm" => getLlmDelegate.write.save(delegatePath)
      case "encoder_decoder" => getSeq2SeqDelegate.write.save(delegatePath)
      case "extractive" => getEmbeddingsDelegate.write.save(delegatePath)
    }
  }

  override def copy(extra: ParamMap): SummarizationModel = {
    val copied = super.copy(extra)
    _llmDelegate.foreach(copied.setLlmDelegate)
    _seq2SeqDelegate.foreach(copied.setSeq2SeqDelegate)
    _embeddingsDelegate.foreach(copied.setEmbeddingsDelegate)
    copied
  }

  override protected def extraValidate(structType: StructType): Boolean = true
}

trait ReadSummarizationDelegate {
  this: ParamsAndFeaturesReadable[SummarizationModel] =>

  def readDelegate(instance: SummarizationModel, path: String, spark: SparkSession): Unit = {
    val delegatePath = new Path(path, SummarizationModel.delegateDir).toString
    instance.getMethod match {
      case "llm" =>
        instance.setLlmDelegate(AutoGGUFModel.load(delegatePath))
      case "encoder_decoder" =>
        instance.setSeq2SeqDelegate(BartTransformer.load(delegatePath))
      case "extractive" =>
        instance.setEmbeddingsDelegate(MPNetEmbeddings.load(delegatePath))
    }
  }

  addReader(readDelegate)
}

/** This is the companion object of [[SummarizationModel]]. Please refer to that class for the
  * documentation.
  */
object SummarizationModel
    extends ParamsAndFeaturesReadable[SummarizationModel]
    with ReadSummarizationDelegate {

  private[seq2seq] val delegateDir = "summarization_delegate"

  /** Some instruction-tuned models occasionally emit the summary as a JSON-string literal
    * (wrapped in double quotes, with `\n` written out as a literal backslash-n) instead of raw
    * text. Unwrap that shape back into plain text; text that isn't wrapped is returned unchanged.
    *
    * Quotes alone are not proof of the JSON shape (a summary may legitimately open and close with
    * a quotation mark), so unwrapping additionally requires at least one JSON escape sequence
    * inside the quotes.
    */
  private[seq2seq] def unwrapJsonStringLiteral(text: String): String = {
    val looksWrapped = text.length >= 2 && text.head == '"' && text.last == '"'
    if (!looksWrapped) return text
    val inner = text.substring(1, text.length - 1)
    val hasEscapes =
      inner.contains("\\n") || inner.contains("\\\"") || inner.contains("\\t")
    if (!hasEscapes) return text

    // decode in a single pass so an escaped backslash cannot recombine with the character
    // after it (sequential replace() calls would turn the literal \\n into a newline)
    val sb = new StringBuilder(inner.length)
    var i = 0
    while (i < inner.length) {
      if (inner(i) == '\\' && i + 1 < inner.length) {
        inner(i + 1) match {
          case 'n' => sb.append('\n'); i += 2
          case 't' => sb.append('\t'); i += 2
          case '"' => sb.append('"'); i += 2
          case '\\' => sb.append('\\'); i += 2
          case _ => sb.append(inner(i)); i += 1
        }
      } else {
        sb.append(inner(i))
        i += 1
      }
    }
    sb.toString.trim
  }
}
