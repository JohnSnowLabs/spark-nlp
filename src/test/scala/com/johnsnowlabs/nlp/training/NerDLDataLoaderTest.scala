package com.johnsnowlabs.nlp.training

import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.annotators.common.{NerTagged, WordpieceEmbeddingsSentence}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.FastTest
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should._

class NerDLDataLoaderTest extends AnyFlatSpec with SparkSessionTest with Matchers {

  lazy val textCols: Array[String] = Array("token", "sentence")
  lazy val labelCol = "label"
  lazy val data: Dataset[Row] =
    CoNLL()
      .readDataset(ResourceHelper.spark, "src/test/resources/ner-corpus/test_ner_dataset.txt")
      .limit(100)
      .select(labelCol, textCols: _*)

  val batchSize = 16

  behavior of "NerDLDataLoader"

  it should "create same batches as non threaded iterator" taggedAs FastTest in {

    val expectedData: Array[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]] = NerTagged
      .iterateOnDataframe(
        data,
        textCols,
        labelCol,
        batchSize = batchSize,
        shuffleInPartition = false)
      .toArray

    val nerDLDataLoader = NerDLDataLoader.iterateOnDataframe(
      data,
      textCols,
      labelCol,
      batchSize = batchSize,
      prefetchBatches = 10,
      shuffleInPartition = false)
    val loaderData = nerDLDataLoader.toArray

    loaderData.length shouldBe expectedData.length
    loaderData should contain theSameElementsInOrderAs expectedData
  }
}
