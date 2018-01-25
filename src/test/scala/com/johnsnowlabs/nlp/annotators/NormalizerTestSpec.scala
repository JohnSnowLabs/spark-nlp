package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{AnnotatorType, ContentProvider, DataBuilder}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by saif on 02/05/17.
  */
class NormalizerTestSpec extends FlatSpec with NormalizerBehaviors {

  "A normalizer" should s"be of type ${AnnotatorType.TOKEN}" in {
    val normalizer = new Normalizer
    assert(normalizer.annotatorType == AnnotatorType.TOKEN)
  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  "A full Normalizer pipeline with latin content" should behave like fullNormalizerPipeline(latinBodyData)
  "A Normalizer pipeline with latin content and disabled lowercasing" should behave like lowercasingNormalizerPipeline(latinBodyData)
}
