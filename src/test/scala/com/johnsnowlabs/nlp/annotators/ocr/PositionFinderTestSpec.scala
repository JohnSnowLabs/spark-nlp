package com.johnsnowlabs.nlp.annotators.ocr

import com.johnsnowlabs.nlp.annotators.ocr.schema.{Mapping, PageMatrix}
import com.johnsnowlabs.nlp.{DocumentAssembler, SparkAccessor}
import com.johnsnowlabs.nlp.annotators.{TextMatcher, Tokenizer}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.util.io.ReadAs
import org.apache.spark.ml.Pipeline
import org.scalatest.FlatSpec
import org.apache.spark.sql.functions._

class PositionFinderTestSpec extends FlatSpec {

  def generateRandomPageMatrix(text: String): Seq[PageMatrix] = {
    val initialp = 0
    var initialx = 39.2844F
    val initialy = 12.2313F
    val initialw = 5.1221F
    val initialh = 61.31F
    val mapping = text.map(c => {
      val m = Mapping(c.toString, 0, initialx, initialy, initialw, initialh)
      initialx += 1F
      m
    }).toArray
    Seq(PageMatrix(mapping))
  }

  "a PositionFinder" should "correctly identify chunk coordinates" in {

    import SparkAccessor.spark.implicits._

    val texts = Seq(
      "Hello world, my name is Michael, I am an artist and I work at Benezar. Robert, an engineer from Farendell, graduated last year. The other one, Lucas, graduated last week."
    )
    val data = texts.toDS.toDF("text")
      .withColumn("positions", typedLit[Seq[PageMatrix]](generateRandomPageMatrix(texts.head)))

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols(Array("document"))
      .setOutputCol("sentence")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

    val entityExtractor = new TextMatcher()
      .setInputCols("sentence", "token")
      .setEntities("src/test/resources/entity-extractor/test-chunks.txt", ReadAs.LINE_BY_LINE)
      .setOutputCol("entity")

    val positionFinder = new PositionFinder()
      .setInputCols("entity")
      .setOutputCol("coordinates")
      .setPageMatrixCol("positions")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        entityExtractor,
        positionFinder
      ))

    val result = pipeline.fit(data).transform(data)

    result.select("positions").show(truncate=true)
    result.select("entity.result").show(truncate=false)
    result.select("coordinates").show(truncate=false)

    succeed

  }

}
