package com.johnsnowlabs.nlp.embeddings

import org.apache.spark.sql.types._
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.{DocumentAssembler, RecursivePipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.scalatest._

class WordEmbeddingsTestSpec extends FlatSpec {

  "Word Embeddings" should "correctly embed clinical words not embed non-existent words" in {


    val words = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/clinical_words.txt")
    val notWords = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/not_words.txt")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("word")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
      .setInputCols("document", "token")
      .setOutputCol("embeddings")

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        embeddings
      ))

    val wordsP = pipeline.fit(words).transform(words).cache()
    val notWordsP = pipeline.fit(notWords).transform(notWords).cache()

    val wordsCoverage = WordEmbeddingsModel.withCoverageColumn(wordsP, "embeddings", "cov_embeddings")
    val notWordsCoverage = WordEmbeddingsModel.withCoverageColumn(notWordsP, "embeddings", "cov_embeddings")

    wordsCoverage.select("word","cov_embeddings").show()
    notWordsCoverage.select("word","cov_embeddings").show()

    val wordsOverallCoverage = WordEmbeddingsModel.overallCoverage(wordsCoverage,"embeddings").percentage
    val notWordsOverallCoverage = WordEmbeddingsModel.overallCoverage(notWordsCoverage,"embeddings").percentage

    ResourceHelper.spark.createDataFrame(
      Seq(
        ("Words", wordsOverallCoverage),("Not Words", notWordsOverallCoverage)
      )
    ).toDF("Dataset", "OverallCoverage").show()

    assert(wordsOverallCoverage == 1)
    assert(notWordsOverallCoverage == 0)
  }

}
