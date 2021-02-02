package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.annotator.{Chunker, PerceptronModel}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.{NGramGenerator, StopWordsCleaner, Tokenizer}
import com.johnsnowlabs.nlp.base.{DocumentAssembler, RecursivePipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{AnnotatorBuilder, EmbeddingsFinisher, Finisher}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.Pipeline
import org.scalatest._

class ChunkEmbeddingsTestSpec extends FlatSpec {

  "ChunkEmbeddings" should "correctly calculate chunk embeddings from Chunker" taggedAs FastTest in {

    val smallCorpus = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val posTagger = PerceptronModel.pretrained()
      .setInputCols("sentence", "token")
      .setOutputCol("pos")

    val chunker= new Chunker()
      .setInputCols(Array("sentence", "pos"))
      .setOutputCol("chunk")
      .setRegexParsers(Array("<DT>?<JJ>*<NN>+"))

    val embeddings = AnnotatorBuilder.getGLoveEmbeddings(smallCorpus)
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val chunkEmbeddings = new ChunkEmbeddings()
      .setInputCols(Array("chunk", "embeddings"))
      .setOutputCol("chunk_embeddings")
      .setPoolingStrategy("AVERAGE")

    val sentenceEmbeddingsChunk = new SentenceEmbeddings()
      .setInputCols(Array("document", "chunk_embeddings"))
      .setOutputCol("sentence_embeddings_chunks")
      .setPoolingStrategy("AVERAGE")

    val finisher = new EmbeddingsFinisher()
      .setInputCols("chunk_embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsVector(true)
      .setCleanAnnotations(false)

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        posTagger,
        chunker,
        embeddings,
        chunkEmbeddings,
        sentenceEmbeddingsChunk,
        finisher
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)

//    pipelineDF.select("chunk.metadata").show(2)
//    pipelineDF.select("chunk.result").show(2)
//
//    pipelineDF.select("embeddings.metadata").show(2)
//    pipelineDF.select("embeddings.embeddings").show(2)
//    pipelineDF.select("embeddings.result").show(2)
//
//    pipelineDF.select("chunk_embeddings").show(2)
//    println("Chunk Embeddings")
//    pipelineDF.select("chunk_embeddings.embeddings").show(2)
//    pipelineDF.select(size(pipelineDF("chunk_embeddings.embeddings")).as("chunk_embeddings_size")).show
//
//    pipelineDF.select("sentence_embeddings_chunks.embeddings").show(2)
//    pipelineDF.select(size(pipelineDF("sentence_embeddings_chunks.embeddings")).as("chunk_embeddings_size")).show
//
//    pipelineDF.select("finished_embeddings").show(2)
//    pipelineDF.select(size(pipelineDF("finished_embeddings")).as("chunk_embeddings_size")).show

  }

  "ChunkEmbeddings" should "correctly calculate chunk embeddings from NGramGenerator" taggedAs FastTest in {

    val smallCorpus = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val nGrams = new NGramGenerator()
      .setInputCols("token")
      .setOutputCol("chunk")
      .setN(2)

    val embeddings = AnnotatorBuilder.getGLoveEmbeddings(smallCorpus)
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val chunkEmbeddings = new ChunkEmbeddings()
      .setInputCols(Array("chunk", "embeddings"))
      .setOutputCol("chunk_embeddings")
      .setPoolingStrategy("AVERAGE")

    val finisher = new Finisher()
      .setInputCols("chunk_embeddings")
      .setOutputCols("finished_embeddings")
      .setOutputAsArray(true)
      .setCleanAnnotations(false)

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        nGrams,
        embeddings,
        chunkEmbeddings,
        finisher
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)

//    pipelineDF.select("token.metadata").show(2)
//    pipelineDF.select("token.result").show(2)
//
//    pipelineDF.select("chunk.metadata").show(2)
//    pipelineDF.select("chunk.result").show(2)
//
//    pipelineDF.select("embeddings.metadata").show(2)
//    pipelineDF.select("embeddings.embeddings").show(2)
//    pipelineDF.select("embeddings.result").show(2)
//
//    pipelineDF.select("chunk_embeddings").show(2)
//
//    pipelineDF.select("chunk_embeddings.embeddings").show(2)
//    pipelineDF.select(size(pipelineDF("chunk_embeddings.embeddings")).as("chunk_embeddings_size")).show
//
//    pipelineDF.select("finished_embeddings").show(2)
//    pipelineDF.select(size(pipelineDF("finished_embeddings")).as("chunk_embeddings_size")).show

    assert(pipelineDF.selectExpr("explode(chunk_embeddings.metadata) as meta").select("meta.chunk").distinct().count() > 1)
  }

  "ChunkEmbeddings" should "correctly work with empty tokens" taggedAs FastTest in {

    val smallCorpus = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val stopWordsCleaner = new StopWordsCleaner()
      .setInputCols("token")
      .setOutputCol("cleanTokens")
      .setStopWords(Array("this", "is", "my", "document", "sentence", "second", "first", ",", "."))
      .setCaseSensitive(false)

    val posTagger = PerceptronModel.pretrained()
      .setInputCols("sentence", "cleanTokens")
      .setOutputCol("pos")

    val chunker= new Chunker()
      .setInputCols(Array("sentence", "pos"))
      .setOutputCol("chunk")
      .setRegexParsers(Array("<DT>?<JJ>*<NN>+"))

    val embeddings = AnnotatorBuilder.getGLoveEmbeddings(smallCorpus)
      .setInputCols("sentence", "cleanTokens")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    val chunkEmbeddings = new ChunkEmbeddings()
      .setInputCols(Array("chunk", "embeddings"))
      .setOutputCol("chunk_embeddings")
      .setPoolingStrategy("AVERAGE")
      .setSkipOOV(true)

    val embeddingsSentence = new SentenceEmbeddings()
      .setInputCols(Array("sentence", "chunk_embeddings"))
      .setOutputCol("sentence_embeddings")
      .setPoolingStrategy("AVERAGE")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        stopWordsCleaner,
        posTagger,
        chunker,
        embeddings,
        chunkEmbeddings,
        embeddingsSentence
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    println(pipelineDF.count())
    pipelineDF.show()
    pipelineDF.select("chunk").show(1)
    pipelineDF.select("embeddings").show(1)
    pipelineDF.select("sentence_embeddings").show(1)

  }

}
