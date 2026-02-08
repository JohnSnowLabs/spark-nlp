package com.johnsnowlabs.reader

import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class LayoutAlignerTest extends AnyFlatSpec with SparkSessionTest {

  val docDirectory = "src/test/resources/reader/doc"

  "LayoutAligner" should "work in a pipeline" in {

    val reader = new ReaderAssembler()
      .setContentType("application/msword")
      .setContentPath(s"$docDirectory/contains-pictures.docx")
      .setOutputAsDocument(false)
      .setOutputCol("data")
//      .setExplodeDocs(true)

    val layoutAligner = new LayoutAligner()
        .setInputCols("data_text", "data_image")
        .setOutputCol("aligned_data")
        .setExplodeDocs(true)
//        .setMergeImagesPerChunk(false)

    val pipeline = new Pipeline().setStages(Array(reader, layoutAligner))
    val resultDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)
    //AutoGUUF

//    resultDf.select("data_text.result", "data_text.metadata", "data_image.metadata", "aligned_data.result", "aligned_data.metadata").show(truncate = false)
    resultDf.select("aligned_data").show(truncate = false)
    resultDf.select("aligned_data.result", "aligned_data.metadata").show(truncate = false)

    resultDf.printSchema()

//    val explodedDf = resultDf.withColumn("aligned_exploded", explode(col("aligned_data")))
//    explodedDf.select("aligned_exploded.result", "aligned_exploded.metadata").show(truncate = false)
  }

  "LayoutAligner" should "work from a dataframe" in {

    val reader = new ReaderAssembler()
      .setContentType("application/msword")
      .setContentPath(s"$docDirectory/contains-pictures.docx")
      .setOutputAsDocument(false)
      .setOutputCol("data")
      .setExplodeDocs(true)

    val layoutAligner = new LayoutAligner()
      .setInputCols("data_text", "data_image")
      .setOutputCol("aligned_data")
      .setExplodeDocs(true)

    val pipeline = new Pipeline().setStages(Array(reader))
    val resultDf = pipeline.fit(emptyDataSet).transform(emptyDataSet)

    resultDf.show()

    val layoutPipeline = new Pipeline().setStages(Array(layoutAligner))
    val layoutResultDf = layoutPipeline.fit(emptyDataSet).transform(resultDf)
    layoutResultDf.show()
  }

}
