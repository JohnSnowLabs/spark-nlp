package com.johnsnowlabs.nlp.annotators.sda.vivekn

import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetectorModel
import com.johnsnowlabs.nlp.annotators.{Normalizer, RegexTokenizer}
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingApproach
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.Row
import org.scalatest._

class ViveknSentimentTestSpec extends FlatSpec {

  "an spark vivekn sentiment analysis annotator" should "process a dataframe successfully" in {
    val results = Map(
      "amazing voice acting" -> "positive",
      "horrible staff" -> "negative",
      "very bad" -> "negative",
      "simply fantastic" -> "positive",
      "incredible!!" -> "positive"
    )
    val data = DataBuilder.basicDataBuild(
      results.keys.toSeq: _*
    )

    AnnotatorBuilder.withViveknSentimentAnalysis(data)
      .select("text", "vivekn")
      .collect().foreach { row => {
      val content = row.getString(0)
      val sentiments = row.getSeq[Row](1).map(Annotation(_).result)
      assert(sentiments.length == 1, "because sentiments per sentence returned more or less than one result?")
      assert(sentiments.head == results(content), s"because text $content returned ${sentiments.head} when it was ${results(content)}")
    }
    }
  }

  "A ViveknSentiment" should "work under a pipeline framework" in {

    val data = ContentProvider.parquetData.limit(1000)

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetectorModel()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new RegexTokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val normalizer = new Normalizer()
      .setInputCols(Array("token"))
      .setOutputCol("normalized")

    val spellChecker = new NorvigSweetingApproach()
      .setInputCols(Array("normalized"))
      .setOutputCol("spell")

    val sentimentDetector = new ViveknSentimentApproach()
      .setInputCols(Array("spell", "sentence"))
      .setOutputCol("vivekn")
      .setPositiveSourcePath("/vivekn/positive/1.txt")
      .setNegativeSourcePath("/vivekn/negative/1.txt")
      .setCorpusPrune(false)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        normalizer,
        spellChecker,
        sentimentDetector
      ))

    val model = pipeline.fit(data)
    model.transform(data).show()

    val PIPE_PATH = "./tmp_pipeline"
    model.write.overwrite().save(PIPE_PATH)
    val loadedPipeline = PipelineModel.read.load(PIPE_PATH)
    loadedPipeline.transform(data).show

    succeed
  }
}
