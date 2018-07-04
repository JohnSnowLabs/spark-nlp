package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._


class TextMatcherTestSpec extends FlatSpec with TextMatcherBehaviors {

  "An TextMatcher" should s"be of type $ENTITY" in {
    val entityExtractor = new TextMatcherModel
    assert(entityExtractor.annotatorType == ENTITY)
  }

  "An TextMatcher" should "extract entities with and without sentences" in {
    val dataset = DataBuilder.basicDataBuild("Hello dolore magna aliqua Lorem ipsum dolor sit in laborum")
    val result = AnnotatorBuilder.withFullTextMatcher(dataset)
    val resultNoSentence = AnnotatorBuilder.withFullTextMatcher(dataset, sbd = false)
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
    val result = AnnotatorBuilder.withFullTextMatcher(dataset, lowerCase = false)
    val extracted = Annotation.collect(result, "entity").flatten.toSeq

    assert(extracted == Seq.empty[Annotation])
  }

  "A Recursive Pipeline TextMatcher" should "extract entities from dataset" in {
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

    val entityExtractor = new TextMatcher()
      .setInputCols("token")
      .setEntities("src/test/resources/entity-extractor/test-phrases.txt", ReadAs.LINE_BY_LINE)
      .setOutputCol("entity")

    val finisher = new Finisher()
      .setInputCols("entity")
      .setOutputAsArray(false)
      .setAnnotationSplitSymbol("@")
      .setValueSplitSymbol("#")

    val recursivePipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        entityExtractor,
        finisher
      ))

    recursivePipeline.fit(data).transform(data).show
    assert(recursivePipeline.fit(data).transform(data).filter("finished_entity == ''").count > 0)
  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  "A full Normalizer pipeline with latin content" should behave like fullTextMatcher(latinBodyData)

}
