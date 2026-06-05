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
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, LongParam, Param, ParamValidators}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

/** Fitted model produced by [[BM25Approach]]. It holds the corpus-level statistics (IDF map,
  * average document length and document count) and scores every document in a dataset against a
  * user-provided query using the Okapi BM25 ranking function.
  *
  * The query is provided at transform time with `setQuery(...)`, so the same fitted model can be
  * reused for many different queries ("fit once, query many times"). For every input document the
  * model emits a single `BM25_RANKINGS` annotation whose `result` is the BM25 score and whose
  * metadata contains:
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

  /** Whether tokens are treated case-sensitively (carried over from [[BM25Approach]]). The query
    * is normalized with the same setting before scoring.
    *
    * @group param
    */
  val caseSensitive = new BooleanParam(
    this,
    "caseSensitive",
    "Whether to treat tokens case-sensitively when scoring")

  /** @group setParam */
  def setCaseSensitive(value: Boolean): this.type = set(caseSensitive, value)

  /** @group getParam */
  def getCaseSensitive: Boolean = $(caseSensitive)

  /** The query that documents are scored against. Set this at query time; the same fitted model
    * can be re-queried by calling `setQuery` again (Default: empty).
    *
    * @group param
    */
  val query = new Param[String](this, "query", "The query to score every document against")

  /** @group setParam */
  def setQuery(value: String): this.type = set(query, value)

  /** @group getParam */
  def getQuery: String = $(query)

  setDefault(
    inputCols -> Array(TOKEN),
    outputCol -> BM25_RANKINGS,
    k1 -> 1.2,
    b -> 0.75,
    caseSensitive -> false,
    query -> "")

  /** Splits a raw query string into normalized terms, applying the same case handling that was
    * used when the corpus statistics were computed. The `(?U)` flag makes `\W` Unicode-aware so
    * accented and other non-ASCII letters are kept as part of a term rather than used as
    * delimiters.
    */
  private def queryTerms: Array[String] = {
    val rawTerms = $(query).split("(?U)\\W+").filter(_.nonEmpty)
    if ($(caseSensitive)) rawTerms else rawTerms.map(_.toLowerCase)
  }

  override def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    require(
      idf.isSet && isDefined(avgDocLength),
      "BM25Model is missing its learned statistics. Produce the model with BM25Approach.fit(...) " +
        "or load a previously fitted model before calling transform.")
    require(
      $(query).trim.nonEmpty,
      "BM25Model requires a query. Set one with setQuery(\"...\") before calling transform.")
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
          "query" -> $(query),
          "doc_len" -> docLength.toString),
        embeddings = Array.emptyFloatArray))
  }
}

trait ReadableBM25Model extends ParamsAndFeaturesReadable[BM25Model]

/** This is the companion object of [[BM25Model]]. Please refer to that class for the
  * documentation.
  */
object BM25Model extends ReadableBM25Model
