package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.{DocumentAssembler, RecursivePipeline}
import com.johnsnowlabs.nlp.util.io.{ReadAs, ResourceHelper}
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest._

class WordEmbeddingsTestSpec extends FlatSpec {

  "Word Embeddings" should "correctly embed clinical words not embed non-existent words" taggedAs SlowTest in {

    val words = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/clinical_words.txt")
    val notWords = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/not_words.txt")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("word")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = WordEmbeddingsModel.pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

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

    wordsCoverage.select("word","cov_embeddings").show(1)
    notWordsCoverage.select("word","cov_embeddings").show(1)

    val wordsOverallCoverage = WordEmbeddingsModel.overallCoverage(wordsCoverage,"embeddings").percentage
    val notWordsOverallCoverage = WordEmbeddingsModel.overallCoverage(notWordsCoverage,"embeddings").percentage

    ResourceHelper.spark.createDataFrame(
      Seq(
        ("Words", wordsOverallCoverage),("Not Words", notWordsOverallCoverage)
      )
    ).toDF("Dataset", "OverallCoverage").show(1)

    assert(wordsOverallCoverage == 1)
    assert(notWordsOverallCoverage == 0)
  }

  "Word Embeddings" should "store and load from disk" taggedAs FastTest in {

    val data =
      ResourceHelper.spark.read.option("header","true").csv("src/test/resources/embeddings/clinical_words.txt")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("word")
      .setOutputCol("document")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val embeddings = new WordEmbeddings()
      .setStoragePath("src/test/resources/random_embeddings_dim4.txt", ReadAs.TEXT)
      .setDimension(4)
      .setStorageRef("glove_4d")
      .setInputCols("document", "token")
      .setOutputCol("embeddings")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        embeddings
      ))

    val model = pipeline.fit(data)

    model.write.overwrite().save("./tmp_embeddings_pipeline")

    model.transform(data).show(1)

    val loadedPipeline1 = PipelineModel.load("./tmp_embeddings_pipeline")

    loadedPipeline1.transform(data).show(1)

    val loadedPipeline2 = PipelineModel.load("./tmp_embeddings_pipeline")

    loadedPipeline2.transform(data).show(1)
  }

}
