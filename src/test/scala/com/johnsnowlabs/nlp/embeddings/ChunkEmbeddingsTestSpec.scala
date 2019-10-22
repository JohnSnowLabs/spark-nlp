package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.Finisher
import com.johnsnowlabs.nlp.annotator.{Chunker, PerceptronModel}
import com.johnsnowlabs.nlp.annotators.{NGramGenerator, Tokenizer}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.base.{DocumentAssembler, RecursivePipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.sql.functions.size
import org.scalatest._

class ChunkEmbeddingsTestSpec extends FlatSpec {

  "ChunkEmbeddings" should "correctly calculate chunk embeddings from Chunker" in {

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

    val embeddings = WordEmbeddingsModel.pretrained()
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
        posTagger,
        chunker,
        embeddings,
        chunkEmbeddings,
        finisher
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)

    pipelineDF.select("chunk.metadata").show(2 ,truncate = 500)
    pipelineDF.select("chunk.result").show(2 ,truncate = 500)

    pipelineDF.select("embeddings.metadata").show(2 ,truncate = 500)
    pipelineDF.select("embeddings.embeddings").show(2 ,truncate = 500)
    pipelineDF.select("embeddings.result").show(2 ,truncate = 500)

    pipelineDF.select("chunk_embeddings").show(2 ,truncate = 500)
    println("Chunk Embeddings")
    pipelineDF.select("chunk_embeddings.embeddings").show(2 ,false)
    pipelineDF.select(size(pipelineDF("chunk_embeddings.embeddings")).as("chunk_embeddings_size")).show

    pipelineDF.select("finished_embeddings").show(2 ,false)
    pipelineDF.select(size(pipelineDF("finished_embeddings")).as("chunk_embeddings_size")).show

  }

  "ChunkEmbeddings" should "correctly calculate chunk embeddings from NGramGenerator" in {

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

    val embeddings = WordEmbeddingsModel.pretrained()
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

    pipelineDF.select("token.metadata").show(2 ,truncate = 500)
    pipelineDF.select("token.result").show(2 ,truncate = 500)

    pipelineDF.select("chunk.metadata").show(2 ,truncate = 500)
    pipelineDF.select("chunk.result").show(2 ,truncate = 500)

    pipelineDF.select("embeddings.metadata").show(2 ,truncate = 500)
    pipelineDF.select("embeddings.embeddings").show(2 ,truncate = 500)
    pipelineDF.select("embeddings.result").show(2 ,truncate = 500)

    pipelineDF.select("chunk_embeddings").show(2 ,truncate = 500)
    println("Chunk Embeddings")
    pipelineDF.select("chunk_embeddings.embeddings").show(2 ,false)
    pipelineDF.select(size(pipelineDF("chunk_embeddings.embeddings")).as("chunk_embeddings_size")).show

    pipelineDF.select("finished_embeddings").show(2 ,false)
    pipelineDF.select(size(pipelineDF("finished_embeddings")).as("chunk_embeddings_size")).show

  }

}
