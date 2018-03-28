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
      .setPositiveSource(ExternalResource("src/test/resources/vivekn/positive/1.txt", ReadAs.LINE_BY_LINE, Map("tokenPattern" -> "\\S+")))
      .setNegativeSource(ExternalResource("src/test/resources/vivekn/negative/1.txt", ReadAs.LINE_BY_LINE, Map("tokenPattern" -> "\\S+")))
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

    val model = pipeline.fit(data)
    model.transform(data).show(1)

    val PIPE_PATH = "./tmp_pipeline"
    model.write.overwrite().save(PIPE_PATH)
    val loadedPipeline = PipelineModel.read.load(PIPE_PATH)
    loadedPipeline.transform(data).show(20)

    succeed
  }
}
