package com.johnsnowlabs.nlp.annotators.sda.vivekn

import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.{Normalizer, Tokenizer}
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.Row
import org.scalatest._

class ViveknSentimentTestSpec extends FlatSpec {

  "A ViveknSentiment" should "should be trained by DataFrame" in {
    import SparkAccessor.spark.implicits._

    val trainingDataDF = Seq(
      ("amazing voice acting", "positive"),
      ("horrible staff", "negative"),
      ("very bad", "negative"),
      ("simply fantastic", "positive"),
      ("incredible!!", "positive")
    ).toDF("text", "sentiment_label")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val sentimentDetector = new ViveknSentimentApproach()
      .setInputCols(Array("token", "sentence"))
      .setOutputCol("vivekn")
      .setSentimentCol("sentiment_label")
      .setCorpusPrune(0)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        sentimentDetector
      ))

    // Train ViveknSentimentApproach inside Pipeline by using DataFrame
    val model = pipeline.fit(trainingDataDF)

    // Use the same Pipeline to predict a new DataFrame
    val testDataDF = Seq(
      "amazing voice acting",
      "horrible staff",
      "very bad",
      "simply fantastic",
      "incredible!!",
      "I think this movie is horrible."
    ).toDF("text")

    model.transform(testDataDF).show(20)
    succeed
  }

  "A ViveknSentiment" should "work under a pipeline framework" in {

    import SparkAccessor.spark.implicits._

    val trainingDataDF = Seq(
      ("amazing voice acting", "positive"),
      ("horrible staff", "negative"),
      ("very bad", "negative"),
      ("simply fantastic", "positive"),
      ("incredible!!", "positive")
    ).toDF("text", "sentiment_label")

    val testDataDF = ContentProvider.parquetData.limit(1000)

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val normalizer = new Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalized")

    val spellChecker = new NorvigSweetingApproach()
      .setInputCols(Array("normalized"))
      .setOutputCol("spell")
      .setDictionary("src/test/resources/spell/words.txt")

    val sentimentDetector = new ViveknSentimentApproach()
      .setInputCols(Array("spell", "sentence"))
      .setOutputCol("vivekn")
      .setSentimentCol("sentiment_label")
      .setCorpusPrune(0)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        normalizer,
        spellChecker,
        sentimentDetector
      ))

    val model = pipeline.fit(trainingDataDF)

    model.transform(testDataDF).show(1)

    val PIPE_PATH = "./tmp_pipeline"
    model.write.overwrite().save(PIPE_PATH)
    val loadedPipeline = PipelineModel.read.load(PIPE_PATH)
    loadedPipeline.transform(testDataDF).show(20)

    succeed
  }

  "an spark vivekn sentiment analysis annotator" should "process a dataframe successfully" in {

    import SparkAccessor.spark.implicits._

    val trainingDataDF = Seq(
      ("amazing voice acting", "positive"),
      ("horrible staff", "negative"),
      ("very bad", "negative"),
      ("simply fantastic", "positive"),
      ("incredible!!", "positive")
    ).toDF("text", "sentiment_label")

    val testDataset = Map(
      "amazing voice acting" -> "positive",
      "horrible staff" -> "negative",
      "very bad" -> "negative",
      "simply fantastic" -> "positive",
      "incredible!!" -> "positive"
    )

    AnnotatorBuilder.withViveknSentimentAnalysis(trainingDataDF)
      .select("text", "vivekn")
      .collect().foreach {
      row => {
        val content = row.getString(0)
        val sentiments = row.getSeq[Row](1).map(Annotation(_).result)
        assert(sentiments.length == 1, "because sentiments per sentence returned more or less than one result?")
        assert(sentiments.head == testDataset(content), s"because text $content returned ${sentiments.head} when it was ${testDataset(content)}")
      }
    }
  }


}
