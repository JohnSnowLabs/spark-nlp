package com.johnsnowlabs.nlp.similarity

import com.johnsnowlabs.nlp.AnnotatorType.DOC_SIMILARITY_RANKINGS
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.similarity.DocumentSimilarityRankerApproach
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings
import com.johnsnowlabs.nlp.finisher.DocumentSimilarityRankerFinisher
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{AnnotatorBuilder, EmbeddingsFinisher}
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.scalatest.flatspec.AnyFlatSpec

class DocumentSimilarityRankerTestSpec extends AnyFlatSpec {
  val spark: SparkSession = ResourceHelper.spark

  "DocumentSimilarityRanker" should "should rank document similarity" taggedAs SlowTest in {

    val smallCorpus = spark
      .createDataFrame(
        List(
          "First document, this is my first sentence. This is my second sentence.",
          "Second document, this is my second sentence. This is my second sentence.",
          "Third document, climate change is arguably one of the most pressing problems of our time.",
          "Fourth document, climate change is definitely one of the most pressing problems of our time.",
          "Fifth document, Florence in Italy, is among the most beautiful cities in Europe.",
          "Sixth document, Florence in Italy, is a very beautiful city in Europe like Lyon in France.",
          "Seventh document, the French Riviera is the Mediterranean coastline of the southeast corner of France.",
          "Eighth document, the warmest place in France is the French Riviera coast in Southern France.")
          .map(Tuple1(_)))
      .toDF("text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = AnnotatorBuilder
      .getGLoveEmbeddings(smallCorpus)
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val embeddingsSentence = new SentenceEmbeddings()
      .setInputCols(Array("document", "embeddings"))
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val sentenceFinisher = new EmbeddingsFinisher()
      .setInputCols("sentence_embeddings")
      .setOutputCols("finished_sentence_embeddings")
      .setCleanAnnotations(false)

    val docSimilarityRanker = new DocumentSimilarityRankerApproach()
      .setInputCols("sentence_embeddings")
      .setOutputCol(DOC_SIMILARITY_RANKINGS)
      .setSimilarityMethod("brp")
      .setNumberOfNeighbours(3)
      .setVisibleDistances(true)
      .setIdentityRanking(false)

    val documentSimilarityFinisher = new DocumentSimilarityRankerFinisher()
      .setInputCols("doc_similarity_rankings")
      .setOutputCols(
        "finished_doc_similarity_rankings_id",
        "finished_doc_similarity_rankings_neighbors")
      .setExtractNearestNeighbor(true)

    val pipeline = new Pipeline()
      .setStages(
        Array(
          documentAssembler,
          sentence,
          tokenizer,
          embeddings,
          embeddingsSentence,
          sentenceFinisher,
          docSimilarityRanker,
          documentSimilarityFinisher))

    val transformed = pipeline.fit(smallCorpus).transform(smallCorpus)

    transformed.printSchema
    transformed
      .select(
        "text",
        "finished_sentence_embeddings",
        "finished_doc_similarity_rankings_id",
        "nearest_neighbor_id",
        "nearest_neighbor_distance")
      .show(false)

    // correct if not empty as inclusive query points are at distance 0.0 from themselves
    assert(!transformed.where(col("nearest_neighbor_distance") === 0.0).rdd.isEmpty() == true)
  }
}
