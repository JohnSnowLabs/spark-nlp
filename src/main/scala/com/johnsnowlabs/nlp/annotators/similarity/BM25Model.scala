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

package com.johnsnowlabs.nlp.annotators.similarity

import com.johnsnowlabs.nlp.AnnotatorType.{BM25_RANKINGS, TOKEN}
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.{
  Annotation,
  AnnotatorModel,
  AnnotatorType,
  HasSimpleAnnotate,
  ParamsAndFeaturesReadable,
  ParamsAndFeaturesWritable
}
import org.apache.spark.ml.param.{
  BooleanParam,
  DoubleParam,
  LongParam,
  Param,
  ParamValidators,
  StringArrayParam
}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

/** Fitted model produced by [[BM25Approach]]. It holds the corpus-level statistics (IDF map,
  * average document length and document count) and scores every document in a dataset against a
  * user-provided query using the Okapi BM25 ranking function.
  *
  * The query is provided at transform time, so the same fitted model can be reused for many
  * different queries ("fit once, query many times"). Provide it either as a raw string with
  * `setQuery(...)` (the model splits it on non-word characters) or — recommended when the corpus
  * was analyzed by a non-trivial pipeline — as already-analyzed tokens with
  * `setQueryTokens(...)`, so the query and the documents are tokenized/normalized identically
  * (see the analyzer-symmetry note on the `query` parameter). For every input document the model
  * emits a single `BM25_RANKINGS` annotation whose `result` is the BM25 score and whose metadata
  * contains:
  *
  *   - `bm25_score` — the BM25 relevance score of the document for the current query
  *   - `num_query_terms_matched` — how many distinct query terms occur in the document
  *   - `query` — the query the document was scored against
  *   - `doc_len` — the number of tokens in the document
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.{StopWordsCleaner, Tokenizer}
  * import com.johnsnowlabs.nlp.annotators.similarity.{BM25Approach, BM25Model}
  * import org.apache.spark.ml.Pipeline
  * import org.apache.spark.sql.functions.{col, explode}
  *
  * val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
  * val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token")
  * val stopWords = new StopWordsCleaner().setInputCols("token").setOutputCol("clean_token")
  * val bm25 = new BM25Approach().setInputCols("clean_token").setOutputCol("bm25_rankings")
  *
  * val model = new Pipeline()
  *   .setStages(Array(documentAssembler, tokenizer, stopWords, bm25))
  *   .fit(corpus)
  *
  * val bm25Model = model.stages.last.asInstanceOf[BM25Model]
  * bm25Model.setQuery("vitamin C health benefits fruits")
  *
  * model.transform(corpus)
  *   .select(explode(col("bm25_rankings")).alias("ranking"))
  *   .select(col("ranking.metadata")("bm25_score").alias("bm25_score"))
  *   .show(false)
  * }}}
  *
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio setParam  4
  * @groupprio getParam  5
  */
class BM25Model(override val uid: String)
    extends AnnotatorModel[BM25Model]
    with HasSimpleAnnotate[BM25Model]
    with ParamsAndFeaturesWritable {

  def this() = this(Identifiable.randomUID("BM25Model"))

  /** Input annotator type: TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  /** Output annotator type: BM25_RANKINGS
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = BM25_RANKINGS

  /** Learned inverse document frequency for every vocabulary term.
    *
    * @group param
    */
  val idf: MapFeature[String, Double] = new MapFeature(this, "idf")

  /** @group setParam */
  def setIdf(value: Map[String, Double]): this.type = set(idf, value)

  /** @group getParam */
  def getIdf: Map[String, Double] = $$(idf)

  /** Average document length (in tokens) of the training corpus.
    *
    * @group param
    */
  val avgDocLength = new DoubleParam(
    this,
    "avgDocLength",
    "Average document length (in tokens) of the training corpus")

  /** @group setParam */
  def setAvgDocLength(value: Double): this.type = set(avgDocLength, value)

  /** @group getParam */
  def getAvgDocLength: Double = $(avgDocLength)

  /** Total number of documents in the training corpus.
    *
    * @group param
    */
  val numDocuments =
    new LongParam(this, "numDocuments", "Total number of documents in the training corpus")

  /** @group setParam */
  def setNumDocuments(value: Long): this.type = set(numDocuments, value)

  /** @group getParam */
  def getNumDocuments: Long = $(numDocuments)

  /** Term-frequency saturation parameter `k1` (carried over from [[BM25Approach]]).
    *
    * @group param
    */
  val k1 = new DoubleParam(
    this,
    "k1",
    "BM25 term-frequency saturation parameter (typical range [1.0, 2.0])",
    ParamValidators.gtEq(0.0))

  /** @group setParam */
  def setK1(value: Double): this.type = set(k1, value)

  /** @group getParam */
  def getK1: Double = $(k1)

  /** Length-normalization parameter `b` (carried over from [[BM25Approach]]).
    *
    * @group param
    */
  val b = new DoubleParam(
    this,
    "b",
    "BM25 length-normalization parameter (range [0.0, 1.0])",
    ParamValidators.inRange(0.0, 1.0))

  /** @group setParam */
  def setB(value: Double): this.type = set(b, value)

  /** @group getParam */
  def getB: Double = $(b)

  /** Whether tokens are treated case-sensitively. This is '''fixed when the corpus statistics are
    * computed''' by [[BM25Approach]] and carried onto the model: the IDF vocabulary keys are
    * stored with that exact case handling. Changing it on a fitted model would desynchronize the
    * query/document terms from the stored vocabulary and silently corrupt the scores, so it is
    * deliberately '''read-only''' here — there is no `setCaseSensitive` on the model. The query
    * is normalized with the same setting before scoring.
    *
    * @group param
    */
  val caseSensitive = new BooleanParam(
    this,
    "caseSensitive",
    "Whether to treat tokens case-sensitively when scoring (read-only; fixed at fit time)")

  /** @group getParam */
  def getCaseSensitive: Boolean = $(caseSensitive)

  /** The query that documents are scored against, as a raw string. This is a '''convenience''':
    * the model tokenizes it itself by splitting on non-word characters (`\W+`) and applying the
    * model's case handling — it does '''not''' run the query through the same annotator pipeline
    * used for the corpus.
    *
    * '''Analyzer-symmetry warning.''' BM25 only scores a query term when it matches a key in the
    * learned IDF vocabulary, and those keys were produced by the pipeline placed in front of
    * [[BM25Approach]] (e.g. [[com.johnsnowlabs.nlp.annotators.Tokenizer Tokenizer]],
    * [[com.johnsnowlabs.nlp.annotators.Normalizer Normalizer]], a stemmer/lemmatizer, ...). If
    * that pipeline ''transforms'' tokens (stemming, lemmatization, punctuation stripping, ...), a
    * raw-string query analyzed only by `\W+` + lowercasing can fail to match and silently
    * contribute nothing to the score. For anything beyond plain tokenization, analyze the query
    * with the '''same''' pipeline and pass the resulting tokens via [[setQueryTokens]].
    *
    * Either `query` or `queryTokens` must be set before `transform`; when both are set,
    * `queryTokens` takes precedence (Default: empty).
    *
    * @group param
    */
  val query = new Param[String](this, "query", "The query to score every document against")

  /** @group setParam */
  def setQuery(value: String): this.type = set(query, value)

  /** @group getParam */
  def getQuery: String = $(query)

  /** The query as a list of '''already-analyzed''' terms. This is the recommended way to query
    * when the corpus was built with a non-trivial pipeline: run the query string through the very
    * same stages used for the documents (for example with a
    * [[com.johnsnowlabs.nlp.LightPipeline LightPipeline]]) and pass the resulting tokens here, so
    * the query and the documents are analyzed identically. When non-empty, `queryTokens`
    * overrides [[query]]. The model still applies its (read-only) case handling to these tokens
    * so they line up with the stored IDF keys (Default: empty).
    *
    * @group param
    */
  val queryTokens = new StringArrayParam(
    this,
    "queryTokens",
    "Pre-analyzed query terms; when non-empty they override the raw query string")

  /** @group setParam */
  def setQueryTokens(value: Array[String]): this.type = set(queryTokens, value)

  /** @group getParam */
  def getQueryTokens: Array[String] = $(queryTokens)

  setDefault(
    inputCols -> Array(TOKEN),
    outputCol -> BM25_RANKINGS,
    k1 -> 1.2,
    b -> 0.75,
    caseSensitive -> false,
    query -> "",
    queryTokens -> Array.empty[String])

  /** Resolves the effective query terms, applying the same case handling that was used when the
    * corpus statistics were computed so the terms line up with the stored IDF keys. Pre-analyzed
    * [[queryTokens]] take precedence when set; otherwise the raw [[query]] string is split on
    * non-word characters. The `(?U)` flag makes `\W` Unicode-aware so accented and other
    * non-ASCII letters are kept as part of a term rather than used as delimiters.
    */
  private def queryTerms: Array[String] = {
    val rawTerms =
      if ($(queryTokens).nonEmpty) $(queryTokens)
      else $(query).split("(?U)\\W+")
    val nonEmpty = rawTerms.filter(_.nonEmpty)
    if ($(caseSensitive)) nonEmpty else nonEmpty.map(_.toLowerCase)
  }

  override def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    require(
      idf.isSet && isDefined(avgDocLength),
      "BM25Model is missing its learned statistics. Produce the model with BM25Approach.fit(...) " +
        "or load a previously fitted model before calling transform.")
    require(
      $(query).trim.nonEmpty || $(queryTokens).exists(_.trim.nonEmpty),
      "BM25Model requires a query. Set one with setQuery(\"...\") for a quick raw-string query, or " +
        "setQueryTokens(Array(...)) with the query analyzed by the same pipeline as the corpus " +
        "(recommended) before calling transform.")
    dataset
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val isCaseSensitive = $(caseSensitive)
    val docTerms = annotations.map { annotation =>
      if (isCaseSensitive) annotation.result else annotation.result.toLowerCase
    }

    val docLength = docTerms.length
    val termFrequencies = docTerms.groupBy(identity).map { case (term, occ) =>
      term -> occ.length
    }

    val idfMap = $$(idf)
    val avgdl = $(avgDocLength)
    val k1Value = $(k1)
    val bValue = $(b)

    // Avoid division by zero on a degenerate (empty) corpus.
    val safeAvgdl = if (avgdl > 0.0) avgdl else 1.0
    val lengthNorm = k1Value * (1.0 - bValue + bValue * (docLength.toDouble / safeAvgdl))

    val distinctQueryTerms = queryTerms.distinct

    var score = 0.0
    var matched = 0
    distinctQueryTerms.foreach { term =>
      val tf = termFrequencies.getOrElse(term, 0)
      // Only count a query term as matched when it is both present in the document and part of
      // the learned vocabulary, i.e. when it actually contributes to the score. Terms pruned by
      // minDocFreq or never seen in the training corpus do not count.
      if (tf > 0) {
        idfMap.get(term).foreach { termIdf =>
          matched += 1
          score += termIdf * (tf * (k1Value + 1.0)) / (tf + lengthNorm)
        }
      }
    }

    // Report whichever query representation was actually used for scoring.
    val effectiveQuery =
      if ($(queryTokens).nonEmpty) $(queryTokens).mkString(" ") else $(query)

    val begin = annotations.headOption.map(_.begin).getOrElse(0)
    val end = annotations.lastOption.map(_.end).getOrElse(0)

    Seq(
      Annotation(
        annotatorType = outputAnnotatorType,
        begin = begin,
        end = end,
        result = score.toString,
        metadata = Map(
          "bm25_score" -> score.toString,
          "num_query_terms_matched" -> matched.toString,
          "query" -> effectiveQuery,
          "doc_len" -> docLength.toString),
        embeddings = Array.emptyFloatArray))
  }
}

trait ReadableBM25Model extends ParamsAndFeaturesReadable[BM25Model]

/** This is the companion object of [[BM25Model]]. Please refer to that class for the
  * documentation.
  */
object BM25Model extends ReadableBM25Model
