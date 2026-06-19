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
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, AnnotatorType}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, IntParam, ParamValidators}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.{Dataset, Row}

/** Trains a BM25 (Okapi BM25) lexical ranker over a corpus of tokenized documents.
  *
  * BM25 is a bag-of-words retrieval function that ranks documents against a query based on the
  * query terms appearing in each document. Because the score of a document depends on
  * corpus-level statistics (how many documents contain a term, and the average document length),
  * BM25 has to be implemented as a two-phase Estimator/Model pair:
  *
  *   - [[BM25Approach]] (this class) scans the full corpus once during `fit()` and learns:
  *     - the total document count `N`
  *     - the document frequency `df(t)` of every vocabulary term
  *     - the average document length `avgdl`
  *     - the inverse document frequency `idf(t)` of every term
  *   - [[BM25Model]] reuses those statistics at query time to score every document against a
  *     user-provided query, emitting a `BM25_RANKINGS` annotation with the relevance score.
  *
  * The IDF uses the non-negative (Lucene / Elasticsearch) variant:
  * {{{
  *   idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
  * }}}
  * and each document is scored as:
  * {{{
  *   score(D, Q) = sum over t in Q of
  *     idf(t) * (tf(t, D) * (k1 + 1)) / (tf(t, D) + k1 * (1 - b + b * |D| / avgdl))
  * }}}
  *
  * The input is a column of `TOKEN` annotations, so BM25 is normally placed after a
  * [[com.johnsnowlabs.nlp.annotators.Tokenizer Tokenizer]] (optionally followed by a
  * [[com.johnsnowlabs.nlp.annotators.Normalizer Normalizer]] and/or
  * [[com.johnsnowlabs.nlp.annotators.StopWordsCleaner StopWordsCleaner]]). For the produced model
  * and usage examples see [[BM25Model]].
  *
  * The learned vocabulary (document frequencies and IDF) is collected to the driver during
  * `fit()`. For corpora with a very large number of distinct terms, raise [[minDocFreq]] to prune
  * rare terms and keep the driver-side vocabulary bounded.
  *
  * ==Example==
  * {{{
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.{StopWordsCleaner, Tokenizer}
  * import com.johnsnowlabs.nlp.annotators.similarity.BM25Approach
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols("document")
  *   .setOutputCol("token")
  *
  * val stopWords = new StopWordsCleaner()
  *   .setInputCols("token")
  *   .setOutputCol("clean_token")
  *   .setCaseSensitive(false)
  *
  * val bm25 = new BM25Approach()
  *   .setInputCols("clean_token")
  *   .setOutputCol("bm25_rankings")
  *   .setK1(1.2)
  *   .setB(0.75)
  *   .setMinDocFreq(1)
  *   .setCaseSensitive(false)
  *
  * val pipeline = new Pipeline().setStages(
  *   Array(documentAssembler, tokenizer, stopWords, bm25))
  *
  * val model = pipeline.fit(corpus)
  * model.stages.last.asInstanceOf[BM25Model].setQuery("vitamin C health benefits fruits")
  * model.transform(corpus).selectExpr("explode(bm25_rankings) as ranking").show(false)
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
class BM25Approach(override val uid: String) extends AnnotatorApproach[BM25Model] {

  def this() = this(Identifiable.randomUID("BM25Approach"))

  override val description: String = "BM25 lexical document ranker"

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

  /** Term-frequency saturation parameter `k1`. Higher values let the score keep growing with term
    * frequency; lower values saturate faster. Typical range `[1.0, 2.0]` (Default: `1.2`).
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

  /** Length-normalization parameter `b`. `0.0` disables document-length normalization, `1.0`
    * applies it fully. Range `[0.0, 1.0]` (Default: `0.75`).
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

  /** Minimum document frequency for a term to be kept in the vocabulary. Terms appearing in fewer
    * than `minDocFreq` documents are dropped from the IDF map (Default: `1`).
    *
    * @group param
    */
  val minDocFreq = new IntParam(
    this,
    "minDocFreq",
    "Drop terms that appear in fewer than this many documents",
    ParamValidators.gtEq(1))

  /** @group setParam */
  def setMinDocFreq(value: Int): this.type = set(minDocFreq, value)

  /** @group getParam */
  def getMinDocFreq: Int = $(minDocFreq)

  /** Whether to treat tokens case-sensitively. When `false` (Default), terms are lowercased
    * before the corpus statistics are computed (and again when a query is scored).
    *
    * @group param
    */
  val caseSensitive = new BooleanParam(
    this,
    "caseSensitive",
    "Whether to treat tokens case-sensitively when computing statistics")

  /** @group setParam */
  def setCaseSensitive(value: Boolean): this.type = set(caseSensitive, value)

  /** @group getParam */
  def getCaseSensitive: Boolean = $(caseSensitive)

  setDefault(k1 -> 1.2, b -> 0.75, minDocFreq -> 1, caseSensitive -> false)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): BM25Model = {

    val inputCol = getInputCols.head
    val isCaseSensitive = $(caseSensitive)

    // Extract the normalized term list for every document from its TOKEN annotations. Empty
    // documents are kept on purpose: BM25's corpus statistics (document count N and average
    // document length) are defined over *all* documents, and a zero-length document naturally
    // contributes nothing to the document frequencies.
    val docTerms = dataset
      .select(inputCol)
      .rdd
      .map { row =>
        Option(row.getAs[Seq[Row]](0)).getOrElse(Seq.empty).map { annotationRow =>
          val term = Annotation(annotationRow).result
          if (isCaseSensitive) term else term.toLowerCase
        }
      }

    // Cache: we scan the corpus twice (statistics, then document frequencies).
    docTerms.persist()

    try {
      // Single pass for both the document count and the total token count.
      val (numDocuments, totalLength) = docTerms.aggregate((0L, 0L))(
        (acc, terms) => (acc._1 + 1L, acc._2 + terms.length),
        (a, b) => (a._1 + b._1, a._2 + b._2))

      require(
        numDocuments > 0,
        "BM25Approach received an empty corpus. Make sure the input token column is populated.")

      val avgDocLength = totalLength.toDouble / numDocuments.toDouble

      // Document frequency: count each distinct term once per document.
      val docFreq = docTerms
        .flatMap(_.distinct.map(term => (term, 1L)))
        .reduceByKey(_ + _)
        .filter { case (_, df) => df >= $(minDocFreq) }
        .collectAsMap()

      val n = numDocuments.toDouble
      val idf = docFreq.map { case (term, df) =>
        // Non-negative (Lucene/Elasticsearch) IDF variant.
        term -> math.log(1.0 + (n - df + 0.5) / (df + 0.5))
      }.toMap

      new BM25Model()
        .setIdf(idf)
        .setAvgDocLength(avgDocLength)
        .setNumDocuments(numDocuments)
    } finally {
      docTerms.unpersist()
    }
  }
}

/** This is the companion object of [[BM25Approach]]. Please refer to that class for the
  * documentation.
  */
object BM25Approach extends DefaultParamsReadable[BM25Approach]
