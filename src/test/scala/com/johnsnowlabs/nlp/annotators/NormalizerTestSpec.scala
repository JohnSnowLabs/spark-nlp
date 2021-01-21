package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.SparkAccessor.spark.implicits._
import com.johnsnowlabs.nlp.{AnnotatorType, ContentProvider, DataBuilder, _}
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._


class NormalizerTestSpec extends FlatSpec with NormalizerBehaviors {

  "A normalizer" should s"be of type ${AnnotatorType.TOKEN}" taggedAs FastTest in {
    val normalizer = new Normalizer
    assert(normalizer.outputAnnotatorType == AnnotatorType.TOKEN)
  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  "A full Normalizer pipeline with latin content" should behave like fullNormalizerPipeline(latinBodyData)
  "A Normalizer pipeline with latin content and disabled lowercasing" should behave like lowercasingNormalizerPipeline(latinBodyData)

  private var data = Seq(
    ("lol", "laugh@out@loud"),
    ("gr8", "great"),
     ("b4", "before"),
    ("4", "for"),
    ("Yo dude whatsup?", "hey@friend@how@are@you")
  ).toDS.toDF("text", "normalized_gt")

  "an isolated normalizer " should behave like testCorrectSlangs(data)

  data = Seq(
    ("test-ing", "testing"),
    ("test-ingX", "testing")
  ).toDS.toDF("text", "normalized_gt")

  "an isolated normalizer " should behave like testMultipleRegexPatterns(data)

}
