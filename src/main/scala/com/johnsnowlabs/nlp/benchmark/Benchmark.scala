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

package com.johnsnowlabs.nlp.benchmark

import com.johnsnowlabs.nlp.AnnotatorType
import com.johnsnowlabs.nlp.benchmark.BenchmarkTask._
import com.johnsnowlabs.nlp.benchmark.engines.{
  DependencyAccuracyEngine,
  LabelAccuracyEngine,
  SpanF1Engine,
  TextMetric,
  TextSimilarityEngine
}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types.{ArrayType, StructField}
import org.apache.spark.storage.StorageLevel

/** Throughput and accuracy benchmarking for any fitted Spark NLP [[PipelineModel]], against data
  * supplied by the caller. See `examples/python/benchmarks/` for end-to-end recipes with
  * recommended public datasets per task.
  *
  * {{{
  * val report = Benchmark.throughput(pipelineModel, myData)
  * println(report)
  *
  * val accuracy = Benchmark.evaluate(pipelineModel, myGoldData, BenchmarkTask.NER)
  * println(accuracy)
  * }}}
  */
object Benchmark {

  // Tasks whose engine runs more than one action against the transformed DataFrame.
  private[benchmark] val tasksNeedingCache: Set[BenchmarkTask] =
    Set(POS, Classification, SpellCheck, LanguageDetection, ImageClassification)

  private val AnnotatorTypeKey = "annotatorType"

  // Matches on the whole field (name, type, and metadata), not just the name: a caller can pass
  // `goldData` that was itself produced by an earlier Spark NLP run and already carries a
  // same-named column (e.g. "category" or "document"). If the new pipeline's real prediction
  // also lands under that name, transform() overwrites it with a differently-typed/annotated
  // field -- matching by name alone would wrongly treat that fresh prediction as pre-existing
  // input and exclude it.
  private def newlyProducedColumns(
      inputSchema: Seq[StructField],
      transformed: DataFrame): Seq[StructField] = {
    val inputFields = inputSchema.toSet
    transformed.schema.fields.filterNot(inputFields.contains)
  }

  private def columnsByType(fields: Seq[StructField]): Seq[(String, String)] =
    fields.flatMap(f =>
      if (f.metadata.contains(AnnotatorTypeKey))
        Some((f.name, f.metadata.getString(AnnotatorTypeKey)))
      else None)

  private def confidenceInterval95(values: Seq[Double]): Double = {
    if (values.length < 2) 0.0
    else {
      val mean = values.sum / values.length
      val variance = values.map(v => math.pow(v - mean, 2)).sum / (values.length - 1)
      1.96 * math.sqrt(variance / values.length)
    }
  }

  /** Measures how fast `pipelineModel` processes `data`, reporting one rate per type of
    * annotation it produces (tokens/sec, sentences/sec, NER-tokens/sec, etc. — inferred from the
    * pipeline's own output, not configured by the caller).
    *
    * Runs `warmupRuns` untimed passes first (JIT/class-loading/Spark planning overhead), then
    * `trials` timed passes, reporting the mean rate per metric with a 95% confidence interval.
    * `data` is read once per pass (1 + `warmupRuns` + `trials` times in total), so if it isn't
    * already cached this persists it for the duration of the call (and unpersists it again
    * afterwards) -- otherwise a `data` sourced from an expensive or non-deterministic upstream
    * read (a file scan, a `.sample()`) would have that cost repeated on every pass, contaminating
    * the measured rate and, for a non-deterministic source, varying the row count between trials.
    *
    * @param data
    *   input data in whatever shape `pipelineModel` expects (a `text` column by default)
    * @param textCol
    *   present for documentation of the expected input column; the pipeline's own stages
    *   determine which columns are actually read
    */
  def throughput(
      pipelineModel: PipelineModel,
      data: DataFrame,
      textCol: String = "text",
      warmupRuns: Int = 1,
      trials: Int = 5): ThroughputReport = {
    require(warmupRuns >= 0, "warmupRuns must be >= 0")
    require(trials >= 1, "trials must be >= 1")
    require(data.schema.fieldNames.contains(textCol), s"data must contain a '$textCol' column")

    // Only persist here if the caller hasn't already -- leaves caller-managed caching untouched
    // instead of unpersisting a DataFrame we didn't cache ourselves.
    val alreadyCached = data.storageLevel != StorageLevel.NONE
    if (!alreadyCached) data.persist()
    try {
      val producedTypes = columnsByType(
        newlyProducedColumns(data.schema.fields, pipelineModel.transform(data)))
      require(
        producedTypes.nonEmpty,
        "The pipeline did not add any annotator-typed output columns to the input data.")

      def countCols = producedTypes.map { case (name, _) => sum(size(col(name))).alias(name) }

      // Must run the same aggregation as the timed trials -- `.count()` alone gets column-pruned
      // and never executes the annotators.
      (0 until warmupRuns).foreach { _ =>
        pipelineModel.transform(data).select(countCols: _*).head()
      }

      val trialResults: Seq[(Double, IndexedSeq[Long])] = (0 until trials).map { _ =>
        val t0 = System.nanoTime()
        val out = pipelineModel.transform(data)
        val row = out.select(countCols: _*).head()
        val elapsed = (System.nanoTime() - t0) / 1e9
        val counts =
          producedTypes.indices.map(i => Option(row.getAs[Long](i)).getOrElse(0L)).toIndexedSeq
        (elapsed, counts)
      }

      val rates = producedTypes.zipWithIndex.map { case ((name, annotatorType), i) =>
        val perTrialRates = trialResults.map { case (elapsed, counts) =>
          counts(i).toDouble / elapsed
        }
        val mean = perTrialRates.sum / perTrialRates.length
        val totalItems = trialResults.map(_._2(i)).sum
        MetricRate(annotatorType, name, totalItems, mean, confidenceInterval95(perTrialRates))
      }

      ThroughputReport(rates, trialResults.map(_._1))
    } finally {
      if (!alreadyCached) data.unpersist()
    }
  }

  // Last matching column wins, not the first: several stages can share an annotator type (e.g.
  // DocumentAssembler and MarianTransformer both emit `document`), and schema order is stage
  // order.
  private def findOutputColumn(
      task: BenchmarkTask,
      annotatorType: String,
      goldInputSchema: Seq[StructField],
      transformed: DataFrame,
      explicitCol: Option[String],
      overrideParamName: String): String =
    explicitCol.getOrElse {
      columnsByType(newlyProducedColumns(goldInputSchema, transformed))
        .collect { case (name, t) if t == annotatorType => name }
        .lastOption
        .getOrElse(throw new IllegalArgumentException(
          s"Could not find an output column of type '$annotatorType' produced by this pipeline " +
            s"for task ${task.name}. Check that the pipeline actually produces this annotation, " +
            s"or pass an explicit $overrideParamName."))
    }

  private def resultArrayCol(df: DataFrame, annotationCol: String, alias: String): DataFrame =
    df.withColumn(alias, expr(s"transform(`$annotationCol`, x -> x.result)"))

  /** Scores `pipelineModel`'s predictions on `goldData` against `task`'s accuracy metric.
    *
    * The expected shape of `goldData` depends on `task`:
    *   - `NER`, `POS`, `WordSegmentation`: `textCol` (raw text) + `labelCol` (`array<string>` of
    *     gold tags/segment boundaries, aligned by position to the pipeline's own tokenization —
    *     for `WordSegmentation` each entry is a `"begin:end"` character-offset string, inclusive
    *     on both ends (e.g. a 2-character token starting at 0 is `"0:1"`)
    *   - `Classification`, `SpellCheck`, `LanguageDetection`: `textCol` + `labelCol` (a single
    *     gold label string per row)
    *   - `ImageClassification`: `textCol` names the pipeline's image input column + `labelCol`
    *     (single gold class label per row). Reports `accuracy` for the model's top-1 label like
    *     every other task, plus an extra `top${topK}Accuracy` metric.
    *   - `DependencyParsing`: `textCol` + `labelCol` (`array<string>` of `"headIndex:label"` per
    *     token)
    *   - `QuestionAnswering`: `textCol` + `labelCol`, either a single reference answer per row or
    *     an `array<string>` of every acceptable answer for that question — with an array, each
    *     prediction is scored against its best-matching reference, matching the official SQuAD
    *     eval script.
    *   - `SpeechRecognition`, `Translation`, `Summarization`: `textCol` + `labelCol` (single
    *     reference text per row: expected transcript/translation/summary)
    *
    * `SpeechRecognition`'s word error rate is case-sensitive, matching
    * [[https://github.com/jitsi/jiwer jiwer]]'s default.
    *
    * `Summarization`'s ROUGE never stems (matches
    * [[https://github.com/google-research/google-research/tree/master/rouge rouge_score]]'s
    * `use_stemmer=False` default) -- published ROUGE numbers for benchmarks like
    * XSum/CNN-DailyMail typically use Porter stemming, so a real model's score here will read
    * lower than those published figures on stem-sensitive text for reasons that have nothing to
    * do with the model.
    *
    * @param predictedCol
    *   which of the pipeline's output columns to score. Optional: by default `evaluate` takes the
    *   last column of the type this task expects. Pass this when a pipeline has several stages
    *   sharing that type and the default picks the wrong one. Reported back on
    *   [[AccuracyReport.scoredColumns]]. For `DependencyParsing` this overrides only the
    *   `dependency`-typed column; use `predictedLabeledDependencyCol` for the
    *   `labeled_dependency`-typed one.
    * @param predictedLabeledDependencyCol
    *   override for the `labeled_dependency`-typed output column, used only by
    *   `DependencyParsing`.
    * @param topK
    *   size of the top-k window for `ImageClassification`'s extra `top${topK}Accuracy` metric;
    *   ignored by every other task.
    */
  def evaluate(
      pipelineModel: PipelineModel,
      goldData: DataFrame,
      task: BenchmarkTask,
      textCol: String = "text",
      labelCol: String = "label",
      predictedCol: Option[String] = None,
      predictedLabeledDependencyCol: Option[String] = None,
      topK: Int = 5): AccuracyReport = {

    require(topK >= 1, "topK must be >= 1")
    require(
      goldData.schema.fieldNames.contains(textCol),
      s"goldData must contain a '$textCol' column")
    require(
      goldData.schema.fieldNames.contains(labelCol),
      s"goldData must contain a '$labelCol' column")

    val goldInputSchema = goldData.schema.fields
    // Only cached for tasks that actually reuse it (see tasksNeedingCache) -- caching the whole
    // annotated DataFrame for a single-pass task costs real memory for no benefit.
    val predicted =
      if (Benchmark.tasksNeedingCache(task)) pipelineModel.transform(goldData).persist()
      else pipelineModel.transform(goldData)

    val resolved = scala.collection.mutable.ListBuffer.empty[String]
    def resolve(annotatorType: String, explicit: Option[String], paramName: String): String = {
      val name =
        findOutputColumn(task, annotatorType, goldInputSchema, predicted, explicit, paramName)
      if (!resolved.contains(name)) resolved += name
      name
    }
    def col1(annotatorType: String): String = resolve(annotatorType, predictedCol, "predictedCol")
    def colLabeledDependency(): String =
      resolve(
        AnnotatorType.LABELED_DEPENDENCY,
        predictedLabeledDependencyCol,
        "predictedLabeledDependencyCol")

    try
      evaluateAgainst(task, predicted, goldData, labelCol, col1, colLabeledDependency, topK)
        .copy(scoredColumns = resolved.toList)
    finally predicted.unpersist()
  }

  private def evaluateAgainst(
      task: BenchmarkTask,
      predicted: DataFrame,
      goldData: DataFrame,
      labelCol: String,
      col1: String => String,
      colLabeledDependency: () => String,
      topK: Int): AccuracyReport = {
    task match {
      case NER =>
        val predCol = col1(AnnotatorType.NAMED_ENTITY)
        val withPred = resultArrayCol(predicted, predCol, "__pred")
        val rdd = withPred
          .select(col("__pred"), col(labelCol))
          .rdd
          .map(r => (r.getAs[Seq[String]](0), r.getAs[Seq[String]](1)))
        SpanF1Engine.evaluate(task, rdd)

      case WordSegmentation =>
        val predCol = col1(AnnotatorType.TOKEN)
        val withPred =
          predicted.withColumn(
            "__predBoundaries",
            expr(s"transform(`$predCol`, x -> array(x.begin, x.end))"))
        val rdd = withPred
          .select(col("__predBoundaries"), col(labelCol))
          .rdd
          .map { r =>
            val predBoundaries =
              r.getAs[Seq[Seq[Int]]](0).map(b => (b.head, b(1))).toSet
            val goldBoundaries = r.getAs[Seq[String]](1).map(parseIntPair).toSet
            (predBoundaries, goldBoundaries)
          }
        SpanF1Engine.evaluateBoundaries(task, rdd)

      case POS =>
        val predCol = col1(AnnotatorType.POS)
        labelAccuracy(
          task,
          resultArrayCol(predicted, predCol, "__pred"),
          "__pred",
          labelCol,
          flatten = true)

      case Classification =>
        labelAccuracy(
          task,
          predicted,
          singleResultExpr(col1(AnnotatorType.CATEGORY), s"'$NoLabelPrediction'"),
          labelCol)

      case SpellCheck =>
        labelAccuracy(
          task,
          predicted,
          singleResultExpr(col1(AnnotatorType.TOKEN), s"'$NoLabelPrediction'"),
          labelCol)

      case LanguageDetection =>
        labelAccuracy(
          task,
          predicted,
          singleResultExpr(col1(AnnotatorType.LANGUAGE), s"'$NoLabelPrediction'"),
          labelCol)

      case ImageClassification =>
        val predCol = col1(AnnotatorType.CATEGORY)
        val rdd = predicted
          .select(col(predCol), col(labelCol))
          .rdd
          .map { r =>
            val annotations = r.getAs[Seq[Row]](0)
            val ranked = annotations.headOption
              .map(a => rankedLabels(a.getAs[Map[String, String]](4)))
              .getOrElse(Seq.empty)
            (ranked, r.getString(1))
          }
        LabelAccuracyEngine.evaluateTopK(task, rdd, k = topK)

      case DependencyParsing =>
        val depCol = col1(AnnotatorType.DEPENDENCY)
        val labeledDepCol = colLabeledDependency()
        val withPred = predicted
          .withColumn(
            "__head",
            expr(s"transform(`$depCol`, x -> CAST(element_at(x.metadata, 'head') AS INT))"))
          .withColumn("__label", expr(s"transform(`$labeledDepCol`, x -> x.result)"))
        val rdd = withPred
          .select(col("__head"), col("__label"), col(labelCol))
          .rdd
          .map { r =>
            val heads = r.getAs[Seq[Int]](0)
            val labels = r.getAs[Seq[String]](1)
            val pred = heads.zip(labels)
            val gold = r.getAs[Seq[String]](2).map(parseHeadLabel)
            (pred, gold)
          }
        DependencyAccuracyEngine.evaluate(task, rdd)

      case QuestionAnswering =>
        val predCol = col1(AnnotatorType.CHUNK)
        val multiRef = goldData.schema(labelCol).dataType.isInstanceOf[ArrayType]
        val rdd = predicted
          .select(expr(singleResultExpr(predCol, "''")).alias("__pred"), col(labelCol))
          .rdd
          .map { r =>
            val golds =
              if (multiRef) r.getAs[Seq[String]](1) else Option(r.getString(1)).toSeq
            (r.getString(0), golds)
          }
        TextSimilarityEngine.evaluateSquad(task, rdd)
      case SpeechRecognition =>
        textSimilarity(task, predicted, col1(AnnotatorType.DOCUMENT), labelCol, TextMetric.Wer)
      case Translation =>
        textSimilarity(task, predicted, col1(AnnotatorType.DOCUMENT), labelCol, TextMetric.Bleu)
      case Summarization =>
        textSimilarity(task, predicted, col1(AnnotatorType.DOCUMENT), labelCol, TextMetric.Rouge)
    }
  }

  private val NoLabelPrediction = "<<no_prediction>>"

  // `IF`/`size` guard against ArrayIndexOutOfBoundsException under Spark's ANSI SQL mode on an
  // empty annotation array.
  private def singleResultExpr(annotationCol: String, onMissing: String): String =
    s"COALESCE(IF(size(`$annotationCol`) > 0, element_at(`$annotationCol`, 1).result, NULL), $onMissing)"

  private def labelAccuracy(
      task: BenchmarkTask,
      df: DataFrame,
      predictedExpr: String,
      labelCol: String,
      flatten: Boolean = false): AccuracyReport = {
    if (!flatten) {
      val rdd = df
        .select(expr(predictedExpr).alias("__pred"), col(labelCol))
        .rdd
        .map(r => (r.getString(0), r.getString(1)))
      LabelAccuracyEngine.evaluate(task, rdd)
    } else {
      // Captured as a String: BenchmarkTask case objects have no no-arg constructor, so shipping
      // `task` itself into the closure fails Java serialization.
      val taskName = task.name
      val rdd = df
        .select(col(predictedExpr), col(labelCol))
        .rdd
        .flatMap { r =>
          val pred = r.getAs[Seq[String]](0)
          val gold = r.getAs[Seq[String]](1)
          require(
            pred.length == gold.length,
            s"Benchmark.evaluate(task = $taskName): predicted and gold sequences have " +
              s"different lengths (${pred.length} vs ${gold.length}) for one row. labelCol must " +
              "align one-to-one with the pipeline's own tokenization for this task.")
          pred.zip(gold)
        }
      LabelAccuracyEngine.evaluate(task, rdd)
    }
  }

  private def textSimilarity(
      task: BenchmarkTask,
      df: DataFrame,
      predCol: String,
      labelCol: String,
      metric: TextMetric): AccuracyReport = {
    val rdd = df
      .select(expr(singleResultExpr(predCol, "''")).alias("__pred"), col(labelCol))
      .rdd
      .map(r => (r.getString(0), r.getString(1)))
    TextSimilarityEngine.evaluate(task, rdd, metric)
  }

  private val reservedMetadataKeys =
    Set("sentence", "image", "chunk", "score", "height", "width", "nChannels", "mode", "origin")

  // Some classifiers (e.g. ViTForImageClassification, see ViTClassifier.scala:184) build metadata
  // keys via `Option(...).toString` without ever unwrapping the Option first, so EVERY class
  // name key arrives as the literal string "Some(damselfly)" rather than "damselfly", and a
  // lookup miss arrives as bare "None". This is a bug in that annotator, not a rare edge case --
  // confirmed against a real `image_classifier_vit_base_patch16_224` prediction, where all 15
  // candidate keys were "Some(...)"-wrapped and only the miss case was bare "None". Filtering
  // "Some(...)" out entirely (rather than unwrapping it) discards every real class on every row
  // and silently reports 0% accuracy regardless of how good the model is. The fix is to unwrap
  // "Some(x)" to its inner label `x` -- still rejecting bare "None" (a genuine no-match) and
  // empty "Some()" -- while passing through any key that was never wrapped in the first place, so
  // other classifiers whose metadata keys are already clean labels keep working unchanged.
  private[benchmark] def unwrapLabelKey(key: String): Option[String] =
    if (key == "None") None
    else if (key.startsWith("Some(") && key.endsWith(")")) {
      val inner = key.substring(5, key.length - 1)
      if (inner.isEmpty) None else Some(inner)
    } else Some(key)

  private[benchmark] def rankedLabels(metadata: Map[String, String]): Seq[String] =
    metadata.iterator
      .filterNot { case (k, _) => reservedMetadataKeys.contains(k) }
      .flatMap { case (k, v) =>
        for {
          label <- unwrapLabelKey(k)
          score <- scala.util.Try(v.toDouble).toOption
        } yield (label, score)
      }
      .toSeq
      .sortBy(-_._2)
      .map(_._1)

  private def parseIntPair(s: String): (Int, Int) = {
    val parts = s.split(":")
    (parts(0).toInt, parts(1).toInt)
  }

  private def parseHeadLabel(s: String): (Int, String) = {
    val idx = s.indexOf(':')
    (s.substring(0, idx).toInt, s.substring(idx + 1))
  }
}
