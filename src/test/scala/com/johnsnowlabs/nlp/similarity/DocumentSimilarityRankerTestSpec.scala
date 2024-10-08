package com.johnsnowlabs.nlp.similarity

import com.johnsnowlabs.nlp.AnnotatorType.DOC_SIMILARITY_RANKINGS
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.similarity.DocumentSimilarityRankerApproach
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.{
  AlbertEmbeddings,
  BertSentenceEmbeddings,
  SentenceEmbeddings
}
import com.johnsnowlabs.nlp.finisher.DocumentSimilarityRankerFinisher
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, element_at, size}
import org.scalatest.flatspec.AnyFlatSpec

class DocumentSimilarityRankerTestSpec extends AnyFlatSpec {
  val spark: SparkSession = ResourceHelper.spark

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

  "DocumentSimilarityRanker" should "should use brp to rank document similarity" taggedAs SlowTest in {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = AlbertEmbeddings
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")

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
      .setIdentityRanking(true)

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

    val trainedPipelineModel = pipeline.fit(smallCorpus)

    val pipelineModelLoc = "./tmp_doc_sim_ranker_brp_pipeline"
    trainedPipelineModel.write.overwrite().save(pipelineModelLoc)
    val pipelineModel = PipelineModel.load(pipelineModelLoc)

    val transformed = pipelineModel.transform(smallCorpus)

    transformed.select("text", "finished_sentence_embeddings").show()

    // correct if not empty as inclusive asRetrieverQuery points are at distance 0.0 from themselves
    assert(!transformed.where(col("nearest_neighbor_distance") === 0.0).rdd.isEmpty() == true)
  }

  "DocumentSimilarityRanker" should "should use min hash to rank document similarity" taggedAs SlowTest in {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = AlbertEmbeddings
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")

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
      .setSimilarityMethod("mh")
      .setNumberOfNeighbours(3)
      .setVisibleDistances(true)
      .setIdentityRanking(true)

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

    val trainedPipelineModel = pipeline.fit(smallCorpus)

    val pipelineModelLoc = "./tmp_doc_sim_ranker_mh_pipeline"
    trainedPipelineModel.write.overwrite().save(pipelineModelLoc)
    val pipelineModel = PipelineModel.load(pipelineModelLoc)

    val transformed = pipelineModel.transform(smallCorpus)

    // correct if not empty as inclusive asRetrieverQuery points are at distance 0.0 from themselves
    assert(!transformed.where(col("nearest_neighbor_distance") === 0.0).rdd.isEmpty() == true)
  }

  "Databricks pipeline" should "should use min hash to rank document similarity" taggedAs SlowTest in {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = AlbertEmbeddings
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")

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
      .setIdentityRanking(true)

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

    transformed
      .select("text", "sentence_embeddings.embeddings")
      .withColumn("extracted_embeddings", element_at(col("embeddings"), 1))
      .withColumn("embeddings_size", size(col("extracted_embeddings")))
      .show(10, false)
  }

  "Pipeline" should "should use rank document similarity as retriever for nearest 3 docs" taggedAs SlowTest in {
    val nbOfNeighbors = 3

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = AlbertEmbeddings
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")

    val embeddingsSentence = new SentenceEmbeddings()
      .setInputCols(Array("document", "embeddings"))
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val sentenceFinisher = new EmbeddingsFinisher()
      .setInputCols("sentence_embeddings")
      .setOutputCols("finished_sentence_embeddings")
      .setCleanAnnotations(false)

    val query = "Fifth document, Florence in Italy, is among the most beautiful cities in Europe."

    val docSimilarityRanker = new DocumentSimilarityRankerApproach()
      .setInputCols("sentence_embeddings")
      .setOutputCol(DOC_SIMILARITY_RANKINGS)
      .setSimilarityMethod("brp")
      .setNumberOfNeighbours(nbOfNeighbors)
      .setVisibleDistances(true)
      .setIdentityRanking(true)
      .asRetriever(query)

    val documentSimilarityFinisher = new DocumentSimilarityRankerFinisher()
      .setInputCols("doc_similarity_rankings")
      .setOutputCols(
        "finished_doc_similarity_rankings_id",
        "finished_doc_similarity_rankings_neighbors")

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

    transformed.show(false)

    assert(transformed.count() === 3)
    assert(transformed.columns.contains("nearest_neighbor_id"))
    assert(transformed.columns.contains("nearest_neighbor_distance"))
  }

  it should "work when setting aggregation method" taggedAs SlowTest in {
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val embeddings = BertSentenceEmbeddings
      .pretrained("sent_biobert_clinical_base_cased", "en")
      .setInputCols("sentence")
      .setOutputCol("sentence_embeddings")

    val document_similarity_ranker = new DocumentSimilarityRankerApproach()
      .setInputCols("sentence_embeddings")
      .setOutputCol("doc_similarity_rankings")
      .setSimilarityMethod("brp")
      .setNumberOfNeighbours(1)
      .setBucketLength(2.0)
      .setNumHashTables(3)
      .setVisibleDistances(true)
      .setIdentityRanking(false)
      .setAggregationMethod("MAX")

    val document_similarity_ranker_finisher = new DocumentSimilarityRankerFinisher()
      .setInputCols("doc_similarity_rankings")
      .setOutputCols(
        "finished_doc_similarity_rankings_id",
        "finished_doc_similarity_rankings_neighbors")
      .setExtractNearestNeighbor(true)

    val pipeline = new Pipeline()
      .setStages(
        Array(
          documentAssembler,
          sentenceDetector,
          embeddings,
          document_similarity_ranker,
          document_similarity_ranker_finisher))

    val transformed = pipeline.fit(smallCorpus).transform(smallCorpus)

    transformed
      .select(
        "doc_similarity_rankings",
        "finished_doc_similarity_rankings_id",
        "finished_doc_similarity_rankings_neighbors")
      .show(10, false)
  }

  "Pipeline" should "should not fail if I use the outputCol and inputCols feature" taggedAs SlowTest in {
    val nbOfNeighbors = 3

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = AlbertEmbeddings
      .pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")

    val embeddingsSentence = new SentenceEmbeddings()
      .setInputCols(Array("document", "embeddings"))
      .setOutputCol("my_sentence_emb")
      .setPoolingStrategy("AVERAGE")

    val sentenceFinisher = new EmbeddingsFinisher()
      .setInputCols("my_sentence_emb")
      .setOutputCols("finished_sentence_embeddings")
      .setCleanAnnotations(false)

    val query = "Fifth document, Florence in Italy, is among the most beautiful cities in Europe."

    val docSimilarityRanker = new DocumentSimilarityRankerApproach()
      .setInputCols("my_sentence_emb")
      .setOutputCol(DOC_SIMILARITY_RANKINGS)
      .setSimilarityMethod("brp")
      .setNumberOfNeighbours(nbOfNeighbors)
      .setVisibleDistances(true)
      .setIdentityRanking(true)
      .asRetriever(query)

    val documentSimilarityFinisher = new DocumentSimilarityRankerFinisher()
      .setInputCols("doc_similarity_rankings")
      .setOutputCols(
        "finished_doc_similarity_rankings_id",
        "finished_doc_similarity_rankings_neighbors")

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

    transformed.show(false)

    assert(transformed.count() === 3)
    assert(transformed.columns.contains("nearest_neighbor_id"))
    assert(transformed.columns.contains("nearest_neighbor_distance"))
  }

}
