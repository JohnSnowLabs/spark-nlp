package com.jsl.nlp.annotators

import com.jsl.nlp.{ContentProvider, DataBuilder}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by saif on 02/05/17.
  */
class EntityExtractorTestSpec extends FlatSpec with EntityExtractorBehaviors {

  "An EntityExtractor" should s"be of type ${EntityExtractor.aType}" in {
    val entityExtractor = new EntityExtractor
    assert(entityExtractor.aType == EntityExtractor.aType)
  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  "A full Normalizer pipeline with latin content" should behave like fullEntityExtractorPipeline(latinBodyData)

}
