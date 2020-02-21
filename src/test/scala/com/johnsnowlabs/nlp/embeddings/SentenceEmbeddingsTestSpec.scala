package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.{AnnotatorBuilder, EmbeddingsFinisher, Finisher}
import com.johnsnowlabs.nlp.annotators.{StopWordsCleaner, Tokenizer}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.base.{DocumentAssembler, RecursivePipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.scalatest._
import org.apache.spark.sql.functions.size

class SentenceEmbeddingsTestSpec extends FlatSpec {

  "SentenceEmbeddings" should "correctly calculate sentence embeddings in WordEmbeddings" in {

    val smallCorpus = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = AnnotatorBuilder.getGLoveEmbeddings(smallCorpus)
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

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        embeddings,
        embeddingsSentence,
        sentenceFinisher
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    pipelineDF.printSchema()
    pipelineDF.select("embeddings.metadata").show(2)
    pipelineDF.select("embeddings.embeddings").show(2)
    pipelineDF.select("embeddings.result").show(2)

    pipelineDF.select("sentence_embeddings").show(2)
    pipelineDF.select("sentence_embeddings.embeddings").show(1)
    pipelineDF.select(size(pipelineDF("sentence_embeddings.embeddings")).as("sentence_embeddings_size")).show

    pipelineDF.select("finished_sentence_embeddings").show(1)
    pipelineDF.select(size(pipelineDF("finished_sentence_embeddings")).as("sentence_embeddings_size")).show

  }

  "SentenceEmbeddings" should "correctly calculate sentence embeddings in BertEmbeddings" in {

    val smallCorpus = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = BertEmbeddings.pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setPoolingLayer(0)

    val embeddingsSentence = new SentenceEmbeddings()
      .setInputCols(Array("document", "embeddings"))
      .setOutputCol("sentence_embeddings")

    val finisher = new Finisher()
      .setInputCols("sentence_embeddings")
      .setOutputCols("finished_embeddings")
      .setCleanAnnotations(false)

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        embeddings,
        embeddingsSentence,
        finisher
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    pipelineDF.printSchema()
    pipelineDF.select("embeddings.metadata").show(2)
    pipelineDF.select("embeddings.embeddings").show(2)
    pipelineDF.select("embeddings.result").show(2)

    pipelineDF.select("sentence_embeddings").show(2)
    pipelineDF.select("sentence_embeddings.embeddings").show(1)
    pipelineDF.select(size(pipelineDF("sentence_embeddings.embeddings")).as("sentence_embeddings_size")).show

    pipelineDF.select("finished_embeddings").show(1)
    pipelineDF.select(size(pipelineDF("finished_embeddings")).as("sentence_embeddings_size")).show

  }

  "SentenceEmbeddings" should "not crash on empty embeddings" in {

    val smallCorpus = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val stopWordsCleaner = new StopWordsCleaner()
      .setInputCols("token")
      .setOutputCol("cleanTokens")
      .setStopWords(Array("this", "is", "my", "document", "sentence", "second", "first", ",", "."))
      .setCaseSensitive(false)

    val embeddings = AnnotatorBuilder.getGLoveEmbeddings(smallCorpus)
      .setInputCols("document", "cleanTokens")
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

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        tokenizer,
        stopWordsCleaner,
        embeddings,
        embeddingsSentence,
        sentenceFinisher
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)
    pipelineDF.show(2)
  }

}
