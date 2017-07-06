package com.jsl.nlp.annotators

import com.jsl.nlp.util.io.ResourceHelper
import com.jsl.nlp.{ContentProvider, DataBuilder, DocumentAssembler, SparkAccessor}
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by saif on 02/05/17.
  */
class LemmatizerTestSpec extends FlatSpec with LemmatizerBehaviors {

  val lemmatizer = new Lemmatizer
  "a lemmatizer" should s"be of type ${Lemmatizer.annotatorType}" in {
    assert(lemmatizer.annotatorType == Lemmatizer.annotatorType)
  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  "A full Normalizer pipeline with latin content" should behave like fullLemmatizerPipeline(latinBodyData)

  "A lemmatizer" should "be readable and writable" taggedAs Tag("LinuxOnly") in {
    val lemmatizer = new Lemmatizer().setLemmaDict(ResourceHelper.retrieveLemmaDict)
    val path = "./test-output-tmp/lemmatizer"
    try {
      lemmatizer.write.overwrite.save(path)
      val lemmatizerRead = Lemmatizer.read.load(path)
      assert(lemmatizer.getLemmaDict.head == lemmatizerRead.getLemmaDict.head)
    } catch {
      case _: java.io.IOException => succeed
    }
  }

  "A lemmatizer" should "work under a pipeline framework" in {

    val data = ContentProvider.parquetData.limit(1000)

    val documentAssembler = new DocumentAssembler()
      .setInputColName("text")
      .setOutputColName("document")

    val tokenizer = new RegexTokenizer()
      .setDocumentCol("document")

    val lemmatizer = new Lemmatizer()
      .setDocumentCol("document")
      .setLemmaDict(ResourceHelper.retrieveLemmaDict)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        tokenizer,
        lemmatizer)
      )

    val model = pipeline.fit(data)
    model.transform(data).show()

    val PIPE_PATH = "./tmp_pipeline"
    model.write.overwrite().save(PIPE_PATH)
    val loadedPipeline = PipelineModel.read.load(PIPE_PATH)
    loadedPipeline.transform(data).show

    succeed
  }

}
