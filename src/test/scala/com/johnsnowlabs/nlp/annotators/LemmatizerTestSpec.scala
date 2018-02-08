package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp._
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.types.ArrayType
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by saif on 02/05/17.
  */
class LemmatizerTestSpec extends FlatSpec with LemmatizerBehaviors {

  require(Some(SparkAccessor).isDefined)

  val lemmatizer = new Lemmatizer
  "a lemmatizer" should s"be of type ${AnnotatorType.TOKEN}" in {
    assert(lemmatizer.annotatorType == AnnotatorType.TOKEN)
  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  "A full Normalizer pipeline with latin content" should behave like fullLemmatizerPipeline(latinBodyData)

  "A lemmatizer" should "be readable and writable" taggedAs Tag("LinuxOnly") in {
    val lemmatizer = new Lemmatizer()
    val path = "./test-output-tmp/lemmatizer"
    try {
      lemmatizer.write.overwrite.save(path)
      val lemmatizerRead = Lemmatizer.read.load(path)
      assert(lemmatizer.getDictionary.path == lemmatizerRead.getDictionary.path)
    } catch {
      case _: java.io.IOException => succeed
    }
  }

  "A lemmatizer" should "work under a pipeline framework" in {

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

    val lemmatizer = new Lemmatizer()
      .setInputCols(Array("token"))
      .setOutputCol("lemma")

    val finisher = new Finisher()
      .setInputCols("lemma")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        lemmatizer,
        finisher
      ))

    val recursivePipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        lemmatizer,
        finisher
      ))

    val model = pipeline.fit(data)
    model.transform(data).show()

    val PIPE_PATH = "./tmp_pipeline"

    model.write.overwrite().save(PIPE_PATH)
    val loadedPipeline = PipelineModel.read.load(PIPE_PATH)
    loadedPipeline.transform(data).show

    val recursiveModel = recursivePipeline.fit(data)
    recursiveModel.transform(data).show()

    recursiveModel.write.overwrite().save(PIPE_PATH)
    val loadedRecPipeline = PipelineModel.read.load(PIPE_PATH)
    loadedRecPipeline.transform(data).show

    succeed
  }

}
