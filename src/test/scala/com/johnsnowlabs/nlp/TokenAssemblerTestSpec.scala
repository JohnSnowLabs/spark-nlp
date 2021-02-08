package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.DataFrame
import org.scalatest._

import scala.collection.mutable

class TokenAssemblerTestSpec extends FlatSpec {

  def createPipeline(corpus: DataFrame): DataFrame = {

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
      .setInputCols(Array("sentence", "tokens"))
      .setOutputCol("newDocs")
      .setPreservePosition(true)

    val finisher = new Finisher()
      .setInputCols("newDocs")
      .setOutputAsArray(true)
      .setCleanAnnotations(false)
      .setOutputCols("output")

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        token,
        tokenAssem,
        finisher
      ))

    val pipelineDF = pipeline.fit(corpus).transform(corpus)

    pipelineDF
  }

  "TokenAssembler" should "correctly turn tokens into original document in simple example" taggedAs FastTest in {

    val smallCorpus = ResourceHelper.spark.read.option("header", "true")
      .csv("src/test/resources/embeddings/sentence_embeddings.csv")

    val result = createPipeline(smallCorpus).select("output")
    val corpusFirst = smallCorpus.first.get(0).toString
    val assemFirst = result.first.getAs[mutable.WrappedArray[String]](0).mkString(" ")

    assert(
      corpusFirst.length == assemFirst.length,
              s"because result sentence length differ: " +
                s"\nresult was \n${assemFirst.length} \nexpected is: \n${corpusFirst.length}")
  }

  "TokenAssembler" should "correctly turn tokens into original document in sentence with line breaks" taggedAs FastTest in {

    import ResourceHelper.spark.implicits._

    var rawData = Seq("Test 1  Number 1 lives\nTest 2  Number 2 |ives\nTest 3  number 3 1ives\nTest 4 number 4 llves\nTest 5 number 5 liwes\nTest 6 Number 6 owner\nTest 7 Number 7 ovner\nTest 8 Number 8  orannnge", "test test")

    val df = rawData.toDF("text")
    val result = createPipeline(df).select("output")

    val assemFirst = result.first.getAs[mutable.WrappedArray[String]](0).mkString(" ")

    assert(
      rawData(0).length == assemFirst.length,
      s"because result sentence length differ: " +
        s"\nresult was \n${assemFirst.length} \nexpected is: \n${rawData.length}")
  }

  "TokenAssembler" should "correctly merge tokens after StopWordsCleaner" taggedAs FastTest in {

    val smallCorpus = ResourceHelper.spark.read.option("header", "true")
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
      .setInputCols(Array("sentence", "cleaned"))
      .setOutputCol("newDocs")
      .setPreservePosition(true)

    val finisher = new Finisher()
      .setInputCols("newDocs")
      .setOutputAsArray(true)
      .setCleanAnnotations(false)
      .setOutputCols("output")

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        token,
        stop,
        tokenAssem,
        finisher
      ))

    val pipelineDF = pipeline.fit(smallCorpus).transform(smallCorpus)

    val corpusFirst = smallCorpus.first.get(0).toString
    val assemFirst = pipelineDF.select("output").first.getAs[mutable.WrappedArray[String]](0).mkString(" ")

    assert(
      corpusFirst.length > assemFirst.length,
      s"because result sentence length is not less, then init sentence length: " +
        s"\nresult was \n${assemFirst.length} \nexpected less then: \n${corpusFirst.length}")

  }

}
