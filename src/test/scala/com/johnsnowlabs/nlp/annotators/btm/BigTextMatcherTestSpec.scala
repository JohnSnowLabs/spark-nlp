package com.johnsnowlabs.nlp.annotators.btm

import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.util.io.ReadAs
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._


class BigTextMatcherTestSpec extends FlatSpec with BigTextMatcherBehaviors {

  "An BigTextMatcher" should s"be of type $CHUNK" taggedAs FastTest in {
    val entityExtractor = new BigTextMatcherModel
    assert(entityExtractor.outputAnnotatorType == CHUNK)
  }

  "A BigTextMatcher" should "extract entities with and without sentences" taggedAs FastTest in {
    val dataset = DataBuilder.basicDataBuild("Hello dolore magna aliqua. Lorem ipsum dolor. sit in laborum")
    val result = AnnotatorBuilder.withFullBigTextMatcher(dataset)
    val resultNoSentence = AnnotatorBuilder.withFullBigTextMatcher(dataset, sbd = false)
    val resultNoSentenceNoCase = AnnotatorBuilder.withFullBigTextMatcher(dataset, sbd = false, caseSensitive = false)
    val extractedSentenced = Annotation.collect(result, "entity").flatten.toSeq
    val extractedNoSentence = Annotation.collect(resultNoSentence, "entity").flatten.toSeq
    val extractedNoSentenceNoCase = Annotation.collect(resultNoSentenceNoCase, "entity").flatten.toSeq

    val expectedSentenced = Seq(
      Annotation(CHUNK, 6, 24, "dolore magna aliqua", Map("sentence" -> "0", "chunk" -> "0")),
      Annotation(CHUNK, 53, 59, "laborum", Map("sentence" -> "2", "chunk" -> "1"))
    )

    val expectedNoSentence = Seq(
      Annotation(CHUNK, 6, 24, "dolore magna aliqua", Map("sentence" -> "0", "chunk" -> "0")),
      Annotation(CHUNK, 53, 59, "laborum", Map("sentence" -> "0", "chunk" -> "1"))
    )

    val expectedNoSentenceNoCase = Seq(
      Annotation(CHUNK, 6, 24, "dolore magna aliqua", Map("sentence" -> "0", "chunk" -> "0")),
      Annotation(CHUNK, 27, 48, "Lorem ipsum dolor. sit", Map("sentence" -> "0", "chunk" -> "1")),
      Annotation(CHUNK, 53, 59, "laborum", Map("sentence" -> "0", "chunk" -> "2"))
    )

    assert(extractedSentenced == expectedSentenced)
    assert(extractedNoSentence == expectedNoSentence)
    assert(extractedNoSentenceNoCase == expectedNoSentenceNoCase)
  }

  "An Entity Extractor" should "search inside sentences" taggedAs FastTest in {
    val dataset = DataBuilder.basicDataBuild("Hello dolore magna. Aliqua")
    val result = AnnotatorBuilder.withFullBigTextMatcher(dataset, caseSensitive = false)
    val extracted = Annotation.collect(result, "entity").flatten.toSeq

    assert(extracted == Seq.empty[Annotation])
  }

  "A Recursive Pipeline BigTextMatcher" should "extract entities from dataset" taggedAs FastTest in {
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

    val entityExtractor = new BigTextMatcher()
      .setInputCols("sentence", "token")
      .setStoragePath("src/test/resources/entity-extractor/test-phrases.txt", ReadAs.TEXT)
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

    val m = recursivePipeline.fit(data)
    m.write.overwrite().save("./tmp_bigtm")
    m.transform(data).show(1,false)
    assert(recursivePipeline.fit(data).transform(data).filter("finished_entity == ''").count > 0)
  }

  "A big text matcher pipeline" should "work fine" taggedAs FastTest in {
    val m = PipelineModel.load("./tmp_bigtm")
    val dataset = DataBuilder.basicDataBuild("Hello dolore magna. Aliqua")
    m.transform(dataset).show(1,false)
  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  "A full Normalizer pipeline with latin content" should behave like fullBigTextMatcher(latinBodyData)

}
