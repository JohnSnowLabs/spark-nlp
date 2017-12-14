package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._


class EntityExtractorTestSpec extends FlatSpec with EntityExtractorBehaviors {

  "An EntityExtractor" should s"be of type ${ENTITY}" in {
    val entityExtractor = new EntityExtractor
    assert(entityExtractor.annotatorType == ENTITY)
  }

  "An EntityExtractor" should "extracts entities" in {
    val dataset = DataBuilder.basicDataBuild("Hello dolore magna aliqua Lorem ipsum dolor sit in laborum")
    val result = AnnotatorBuilder.withFullEntityExtractor(dataset)
    val extracted = Annotation.collect(result, "entity").flatten.toSeq

    val expected = Seq(
      Annotation(ENTITY, 6, 24, "dolor magna aliqua", Map()),
      Annotation(ENTITY, 26, 46, "lorem ipsum dolor sit", Map()),
      Annotation(ENTITY, 51, 57, "laborum", Map())
    )

    assert(extracted == expected)
  }

  "An Entity Extractor" should "search inside sentences" in {
    val dataset = DataBuilder.basicDataBuild("Hello dolore magna. Aliqua")
    val result = AnnotatorBuilder.withFullEntityExtractor(dataset)
    val extracted = Annotation.collect(result, "entity").flatten.toSeq

    assert(extracted == Seq.empty[Annotation])
  }

  "An Entity Extractor" should "search in all text" in {
    val dataset = DataBuilder.basicDataBuild("Hello dolore magna. Aliqua")
    val result = AnnotatorBuilder.withFullEntityExtractor(dataset, false)
    val extracted = Annotation.collect(result, "entity").flatten.toSeq

    assert(extracted.length == 1)
  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  "A full Normalizer pipeline with latin content" should behave like fullEntityExtractorPipeline(latinBodyData)

}
