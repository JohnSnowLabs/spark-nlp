package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.scalatest._

class TokenAssemblerTestSpec extends FlatSpec {

  "TokenAssembler" should "correctly turn tokens into orignal document" in {

    val smallCorpus = ResourceHelper.spark.read.option("header","true")
      .csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val token = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("tokens")

    val tokenAssem = new TokenAssembler()
      .setInputCols("tokens")
      .setOutputCol("newDocs")

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        token,
        tokenAssem
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)

    pipelineDF.show()
    pipelineDF.select("document").show(2, false)
    pipelineDF.select("sentence").show(2, false)
    pipelineDF.select("newDocs").show(2, false)
    pipelineDF.select("tokens").show(2, false)
  }

  "TokenAssembler" should "correctly merge tokens after StopWordsCleaner" in {

    val smallCorpus = ResourceHelper.spark.read.option("header","true")
      .csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val token = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("tokens")

    val stop = new StopWordsCleaner()
      .setInputCols("tokens")
      .setOutputCol("cleaned")

    val tokenAssem = new TokenAssembler()
      .setInputCols("cleaned")
      .setOutputCol("newDocs")

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        token,
        stop,
        tokenAssem
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)

    pipelineDF.show()
    pipelineDF.select("document").show(2, false)
    pipelineDF.select("sentence").show(2, false)
    pipelineDF.select("tokens").show(2, false)
    pipelineDF.select("cleaned").show(2, false)
    pipelineDF.select("newDocs").show(2, false)

  }

}
