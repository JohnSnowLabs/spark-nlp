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

import com.johnsnowlabs.nlp.LightPipeline
import com.johnsnowlabs.nlp.annotators.{Stemmer, StopWordsCleaner, Tokenizer}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.commons.io.FileUtils
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.functions.{col, explode}
import org.scalatest.flatspec.AnyFlatSpec

import java.nio.file.Files

class BM25TestSpec extends AnyFlatSpec {

  import ResourceHelper.spark.implicits._

  private val corpus = Seq(
    (1, "Apples are a great source of dietary fiber and vitamin C."),
    (2, "Bananas are rich in potassium and natural sugars."),
    (3, "Machine learning is a subset of artificial intelligence."),
    (4, "Deep learning uses neural networks with many layers."),
    (5, "Florence in Italy is one of the most beautiful cities in Europe."),
    (6, "The French Riviera is a warm coastal region in southern France."),
    (7, "Vitamin C deficiency can lead to scurvy and immune system problems."),
    (8, "Neural networks are inspired by the human brain's structure."),
    (9, "Potassium is an essential mineral for heart and muscle function."),
    (10, "Italy is home to some of the world's greatest Renaissance art."))
    .toDF("id", "text")

  private val documentAssembler =
    new DocumentAssembler().setInputCol("text").setOutputCol("document")

  private val tokenizer = new Tokenizer().setInputCols("document").setOutputCol("token")

  private val stopWordsCleaner = new StopWordsCleaner()
    .setInputCols("token")
    .setOutputCol("clean_token")
    .setCaseSensitive(false)

  private def fitPipeline(): PipelineModel = {
    val bm25 = new BM25Approach()
      .setInputCols("clean_token")
      .setOutputCol("bm25_rankings")
      .setK1(1.2)
      .setB(0.75)
      .setMinDocFreq(1)
      .setCaseSensitive(false)

    new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, stopWordsCleaner, bm25))
      .fit(corpus)
  }

  /** Score the corpus and return rows of (id, bm25_score, terms_matched) ordered by score desc.
    */
  private def rankedResults(model: PipelineModel): Array[(Int, Double, Int)] = {
    model
      .transform(corpus)
      .select(col("id"), explode(col("bm25_rankings")).alias("ranking"))
      .select(
        col("id"),
        col("ranking.metadata")("bm25_score").cast("double").alias("bm25_score"),
        col("ranking.metadata")("num_query_terms_matched").cast("int").alias("terms_matched"))
      .orderBy(col("bm25_score").desc)
      .as[(Int, Double, Int)]
      .collect()
  }

  "BM25Approach" should "learn corpus statistics during fit" taggedAs FastTest in {
    val pipelineModel = fitPipeline()
    val bm25Model = pipelineModel.stages.last.asInstanceOf[BM25Model]

    assert(bm25Model.getNumDocuments == 10L)
    assert(bm25Model.getAvgDocLength > 0.0)
    assert(bm25Model.getIdf.nonEmpty)
    // All IDF weights are non-negative with the Lucene variant.
    assert(bm25Model.getIdf.values.forall(_ >= 0.0))
    // "vitamin" appears in two documents, so it must be in the vocabulary.
    assert(bm25Model.getIdf.contains("vitamin"))
  }

  it should "rank documents by lexical relevance to the query" taggedAs FastTest in {
    val pipelineModel = fitPipeline()
    pipelineModel.stages.last
      .asInstanceOf[BM25Model]
      .setQuery("vitamin C health benefits fruits")

    val ranked = rankedResults(pipelineModel)
    ranked.foreach { case (id, score, matched) =>
      println(s"id=$id score=$score terms_matched=$matched")
    }

    val topTwoIds = ranked.take(2).map(_._1).toSet
    // Documents 1 and 7 are the only ones mentioning "vitamin"/"C".
    assert(topTwoIds == Set(1, 7))

    // Every score is non-negative and documents with no query term score exactly 0.
    assert(ranked.forall(_._2 >= 0.0))
    val unrelated = ranked.filter(_._1 == 3).head
    assert(unrelated._2 == 0.0)
    assert(unrelated._3 == 0)

    // num_query_terms_matched is reported for the matching documents.
    assert(ranked.filter(_._1 == 1).head._3 >= 1)
  }

  it should "reuse the same fitted statistics across different queries" taggedAs FastTest in {
    val pipelineModel = fitPipeline()
    val bm25Model = pipelineModel.stages.last.asInstanceOf[BM25Model]

    bm25Model.setQuery("neural networks deep learning")
    val aiTop = rankedResults(pipelineModel).head._1
    assert(Set(4, 8).contains(aiTop))

    bm25Model.setQuery("Italy France Europe travel")
    val travelTop = rankedResults(pipelineModel).head._1
    assert(Set(5, 6, 10).contains(travelTop))
  }

  it should "require a query before transform" taggedAs FastTest in {
    val pipelineModel = fitPipeline()
    // query defaults to empty -> transform must fail fast with a helpful message.
    val thrown = intercept[IllegalArgumentException] {
      pipelineModel.transform(corpus).collect()
    }
    assert(thrown.getMessage.contains("query"))
  }

  "BM25Model" should "be saved and loaded and produce identical rankings" taggedAs FastTest in {
    val pipelineModel = fitPipeline()
    val bm25Model = pipelineModel.stages.last.asInstanceOf[BM25Model]
    bm25Model.setQuery("vitamin C health benefits fruits")
    val before = rankedResults(pipelineModel)

    val modelPath = Files.createTempDirectory("bm25_model").toFile
    try {
      bm25Model.write.overwrite().save(modelPath.getAbsolutePath)
      val loaded = BM25Model.load(modelPath.getAbsolutePath)

      assert(loaded.getNumDocuments == bm25Model.getNumDocuments)
      assert(loaded.getAvgDocLength == bm25Model.getAvgDocLength)
      assert(loaded.getIdf == bm25Model.getIdf)

      loaded.setQuery("vitamin C health benefits fruits")
      val inferPipeline = new Pipeline()
        .setStages(Array(documentAssembler, tokenizer, stopWordsCleaner, loaded))
        .fit(corpus)
      val after = inferPipeline
        .transform(corpus)
        .select(
          col("id"),
          col("bm25_rankings.metadata")(0)("bm25_score").cast("double").alias("bm25_score"))
        .as[(Int, Double)]
        .collect()
        .toMap

      before.foreach { case (id, score, _) =>
        assert(math.abs(after(id) - score) < 1e-9)
      }
    } finally {
      FileUtils.deleteDirectory(modelPath)
    }
  }

  "BM25" should "compute exact scores and statistics on a controlled corpus" taggedAs FastTest in {
    // A tiny corpus with no stop words, so the tokens are fully predictable and the BM25
    // statistics can be derived by hand.
    val tiny =
      Seq((1, "apple apple banana"), (2, "banana cherry"), (3, "cherry")).toDF("id", "text")

    val bm25 = new BM25Approach()
      .setInputCols("token")
      .setOutputCol("bm25_rankings")

    val model = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, bm25))
      .fit(tiny)
    val bm25Model = model.stages.last.asInstanceOf[BM25Model]

    // N = 3 documents, lengths 3 + 2 + 1 = 6 -> avgdl = 2.0
    assert(bm25Model.getNumDocuments == 3L)
    assert(bm25Model.getAvgDocLength == 2.0)

    // df(apple)=1, df(banana)=2, df(cherry)=2 ; idf(t)=ln(1+(N-df+0.5)/(df+0.5))
    assert(math.abs(bm25Model.getIdf("apple") - math.log(1.0 + 2.5 / 1.5)) < 1e-9)
    assert(math.abs(bm25Model.getIdf("banana") - math.log(1.6)) < 1e-9)

    bm25Model.setQuery("apple banana")
    val scored = model
      .transform(tiny)
      .select(
        col("id"),
        col("bm25_rankings.metadata")(0)("bm25_score").cast("double").alias("score"),
        col("bm25_rankings.metadata")(0)("num_query_terms_matched").cast("int").alias("matched"))
      .as[(Int, Double, Int)]
      .collect()
      .map(r => r._1 -> (r._2, r._3))
      .toMap

    // Hand-computed BM25 for document 1 ("apple apple banana", len 3, avgdl 2, k1=1.2, b=0.75):
    //   lengthNorm = 1.2 * (1 - 0.75 + 0.75 * 3/2) = 1.65
    //   apple : idf=ln(2.6667) tf=2 -> idf * (2*2.2)/(2+1.65)
    //   banana: idf=ln(1.6)    tf=1 -> idf * (1*2.2)/(1+1.65)
    val lengthNorm = 1.2 * (1.0 - 0.75 + 0.75 * 3.0 / 2.0)
    val expectedDoc1 =
      math.log(1.0 + 2.5 / 1.5) * (2 * 2.2) / (2 + lengthNorm) +
        math.log(1.6) * (1 * 2.2) / (1 + lengthNorm)
    assert(math.abs(scored(1)._1 - expectedDoc1) < 1e-9)
    assert(scored(1)._2 == 2) // both query terms matched in document 1
    assert(scored(3)._1 == 0.0) // document 3 ("cherry") has neither query term
    assert(scored(3)._2 == 0)
  }

  it should "honor non-default parameters and prune by minDocFreq" taggedAs FastTest in {
    val tiny =
      Seq((1, "apple apple banana"), (2, "banana cherry"), (3, "cherry")).toDF("id", "text")

    val bm25 = new BM25Approach()
      .setInputCols("token")
      .setOutputCol("bm25_rankings")
      .setK1(2.0)
      .setB(0.5)
      .setMinDocFreq(2)
      .setCaseSensitive(true)

    val model = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, bm25))
      .fit(tiny)
    val bm25Model = model.stages.last.asInstanceOf[BM25Model]

    // Non-default tuning params must be copied from the approach onto the fitted model.
    assert(bm25Model.getK1 == 2.0)
    assert(bm25Model.getB == 0.5)
    assert(bm25Model.getCaseSensitive)

    // minDocFreq=2 drops "apple" (df=1) but keeps "banana"/"cherry" (df=2).
    assert(!bm25Model.getIdf.contains("apple"))
    assert(bm25Model.getIdf.contains("banana"))

    // "apple" is present in document 1 but pruned from the vocabulary, so it must NOT count as a
    // matched term, and only "banana" contributes to the score.
    bm25Model.setQuery("apple banana")
    val doc1 = model
      .transform(tiny)
      .where(col("id") === 1)
      .select(
        col("bm25_rankings.metadata")(0)("bm25_score").cast("double").alias("score"),
        col("bm25_rankings.metadata")(0)("num_query_terms_matched").cast("int").alias("matched"))
      .as[(Double, Int)]
      .head()
    assert(doc1._2 == 1)
    assert(doc1._1 > 0.0)
  }

  it should "reject out-of-range parameters" taggedAs FastTest in {
    intercept[IllegalArgumentException](new BM25Approach().setB(2.0))
    intercept[IllegalArgumentException](new BM25Approach().setK1(-1.0))
    intercept[IllegalArgumentException](new BM25Approach().setMinDocFreq(0))
  }

  "BM25 with a token-transforming pipeline" should
    "match an analyzed query that a raw-string query would miss" taggedAs FastTest in {
      // The corpus is stemmed, so the IDF vocabulary only ever contains stems. A raw-string query
      // is NOT stemmed by the model -- this is exactly the analyzer-asymmetry trap. Pre-analyzing
      // the query with the same pipeline (setQueryTokens) is the cure.
      val docs = Seq((1, "running runs"), (2, "jumping jumps")).toDF("id", "text")

      val stemmer = new Stemmer().setInputCols("token").setOutputCol("stem")

      // Analyze the query with exactly the stages used for the documents.
      val analyzer = new Pipeline()
        .setStages(Array(documentAssembler, tokenizer, stemmer))
        .fit(docs)
      val analyzedQuery = new LightPipeline(analyzer).annotate("running")("stem").toArray
      // The stemmer actually transforms the surface form, which is what creates the asymmetry.
      assert(analyzedQuery.nonEmpty && !analyzedQuery.contains("running"))

      val bm25 = new BM25Approach().setInputCols("stem").setOutputCol("bm25_rankings")
      val model = new Pipeline()
        .setStages(Array(documentAssembler, tokenizer, stemmer, bm25))
        .fit(docs)
      val bm25Model = model.stages.last.asInstanceOf[BM25Model]

      // The vocabulary is built from stems, so the surface form "running" is never a key, but the
      // analyzed query terms are.
      assert(analyzedQuery.forall(bm25Model.getIdf.contains))
      assert(!bm25Model.getIdf.contains("running"))

      def scoreOfDoc1(): Double =
        model
          .transform(docs)
          .where(col("id") === 1)
          .select(col("bm25_rankings.metadata")(0)("bm25_score").cast("double"))
          .as[Double]
          .head()

      // Raw-string query bypasses the stemmer -> "running" misses the vocabulary -> 0.
      bm25Model.setQuery("running")
      assert(scoreOfDoc1() == 0.0)

      // Pre-analyzed query (same pipeline) -> matches the stemmed vocabulary -> > 0. queryTokens
      // also takes precedence over the still-set raw query.
      bm25Model.setQueryTokens(analyzedQuery)
      assert(scoreOfDoc1() > 0.0)
    }

  it should "score identically whether the query is a raw string or pre-analyzed tokens" taggedAs FastTest in {
    val tiny =
      Seq((1, "apple apple banana"), (2, "banana cherry"), (3, "cherry")).toDF("id", "text")
    val bm25 = new BM25Approach().setInputCols("token").setOutputCol("bm25_rankings")
    val model = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, bm25))
      .fit(tiny)
    val bm25Model = model.stages.last.asInstanceOf[BM25Model]

    def scores(): Map[Int, Double] =
      model
        .transform(tiny)
        .select(
          col("id"),
          col("bm25_rankings.metadata")(0)("bm25_score").cast("double").alias("s"))
        .as[(Int, Double)]
        .collect()
        .toMap

    bm25Model.setQuery("apple banana")
    val viaString = scores()
    // The same terms supplied as pre-analyzed tokens must reproduce the scores exactly.
    bm25Model.setQueryTokens(Array("apple", "banana"))
    val viaTokens = scores()
    viaString.foreach { case (id, s) => assert(math.abs(viaTokens(id) - s) < 1e-9) }
  }
}
