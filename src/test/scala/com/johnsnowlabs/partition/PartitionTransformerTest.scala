package com.johnsnowlabs.partition

import com.johnsnowlabs.nlp.annotator.MarianTransformer
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.annotators.cleaners.Cleaner
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class PartitionTransformerTest extends AnyFlatSpec with SparkSessionTest {

  val wordDirectory = "src/test/resources/reader/doc"

  "PartitionTransformer" should "work in a RAG pipeline" taggedAs SlowTest in {
    val partition = new PartitionTransformer()
      .setContentPath(s"$wordDirectory/fake_table.docx")
      .setOutputCol("partition")

    val marian = MarianTransformer
      .pretrained()
      .setInputCols("partition")
      .setOutputCol("translation")
      .setMaxInputLength(30)

    val pipeline = new Pipeline()
      .setStages(Array(partition, marian))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(emptyDataSet)
    resultDf.select("doc", "partition", "translation").show(truncate = false)
  }

  it should "work with a Document input" taggedAs FastTest in {
    import spark.implicits._
    val testDataSet = Seq("An example with DocumentAssembler annotator").toDS.toDF("text")

    val partition = new PartitionTransformer()
      .setInputCols("document")
      .setOutputCol("partition")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, partition))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(testDataSet)
    resultDf.show(truncate = false)
  }

  it should "work with a Cleaner input" taggedAs FastTest in {
    import spark.implicits._
    val testDf = Seq("\\x88This text contains ®non-ascii characters!●").toDS.toDF("text")
    testDf.show(truncate = false)

    val cleaner = new Cleaner()
      .setInputCols("document")
      .setOutputCol("cleaned")
      .setCleanerMode("clean_non_ascii_chars")

    val partition = new PartitionTransformer()
      .setInputCols("cleaned")
      .setOutputCol("partition")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, cleaner, partition))

    val pipelineModel = pipeline.fit(emptyDataSet)
    val resultDf = pipelineModel.transform(testDf)
    resultDf.show(truncate = false)
  }

}
