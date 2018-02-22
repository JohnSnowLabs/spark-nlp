package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._


class EntityExtractorTestSpec extends FlatSpec with EntityExtractorBehaviors {

  "An EntityExtractor" should s"be of type $ENTITY" in {
    val entityExtractor = new EntityExtractorModel
    assert(entityExtractor.annotatorType == ENTITY)
  }

  "An EntityExtractor" should "extract entities with and without sentences" in {
    val dataset = DataBuilder.basicDataBuild("Hello dolore magna aliqua Lorem ipsum dolor sit in laborum")
    val result = AnnotatorBuilder.withFullEntityExtractor(dataset)
    val resultNoSentence = AnnotatorBuilder.withFullEntityExtractor(dataset, sbd = false)
    val extracted = Annotation.collect(result, "entity").flatten.toSeq
    val extractedNoSentence = Annotation.collect(resultNoSentence, "entity").flatten.toSeq

    val expected = Seq(
      Annotation(ENTITY, 6, 24, "dolore magna aliqua", Map()),
      Annotation(ENTITY, 26, 46, "lorem ipsum dolor sit", Map()),
      Annotation(ENTITY, 51, 57, "laborum", Map())
    )

    assert(extracted == expected)
    assert(extractedNoSentence == expected)
  }

  "An Entity Extractor" should "search inside sentences" in {
    val dataset = DataBuilder.basicDataBuild("Hello dolore magna. Aliqua")
    val result = AnnotatorBuilder.withFullEntityExtractor(dataset, lowerCase = false)
    val extracted = Annotation.collect(result, "entity").flatten.toSeq

    assert(extracted == Seq.empty[Annotation])
  }

  "A Recursive Pipeline EntityExtractor" should "extract entities from dataset" in {
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

    val entityExtractor = new EntityExtractor()
      .setInputCols("token")
      .setEntities("/entity-extractor/test-phrases.txt", ReadAs.LINE_BY_LINE, Map.empty[String, String])
      .setOutputCol("entity")

    val finisher = new Finisher()
      .setInputCols("entity")

    val recursivePipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        entityExtractor,
        finisher
      ))

    assert(recursivePipeline.fit(data).transform(data).filter("finished_entity == ''").count > 0)
  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  "A full Normalizer pipeline with latin content" should behave like fullEntityExtractor(latinBodyData)

}
