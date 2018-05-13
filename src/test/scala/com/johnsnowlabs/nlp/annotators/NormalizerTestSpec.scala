package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{AnnotatorType, ContentProvider, DataBuilder}
import org.apache.spark.sql.{Dataset, Row}
import com.johnsnowlabs.nlp._
import org.scalatest._

import SparkAccessor.spark.implicits._

/**
  * Created by saif on 02/05/17.
  */
class NormalizerTestSpec extends FlatSpec with NormalizerBehaviors {

  /*"A normalizer" should s"be of type ${AnnotatorType.TOKEN}" in {
    val normalizer = new Normalizer
    assert(normalizer.annotatorType == AnnotatorType.TOKEN)
  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  "A full Normalizer pipeline with latin content" should behave like fullNormalizerPipeline(latinBodyData)
  "A Normalizer pipeline with latin content and disabled lowercasing" should behave like lowercasingNormalizerPipeline(latinBodyData)


  var data = Seq(
    ("lol", "laugh out loud"),
    ("gr8", "great"),
     ("b4", "before"),
    ("4", "for")
  ).toDS.toDF("text", "normalized_gt")

  "an isolated normalizer " should behave like testCorrectSlangs(data)

  data = Seq(
    ("test-ing", "testing"),
    ("test-ingX", "testing")
  ).toDS.toDF("text", "normalized_gt")

  "an isolated normalizer " should behave like testMultipleRegexPatterns(data)*/

  "a loaded model " should behave like testLoadModel()

}
