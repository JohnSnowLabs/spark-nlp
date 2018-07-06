package com.johnsnowlabs.nlp.util.io

import com.johnsnowlabs.nlp.DocumentAssembler
import org.apache.spark.sql.SparkSession
import org.scalatest._

class OcrExample extends FlatSpec {

  "OcrExample with Spark" should "successfully create a dataset" in {

    val spark: SparkSession = SparkSession.builder()
      .appName("SparkNLP-OCR-Default-Spark")
      .master("local[*]")
      .config("spark.driver.memory", "4G")
      .config("spark.driver.maxResultSize", "2G")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "500m")
      .getOrCreate()

    val data = OcrHelper.createDataset(spark, "/home/saif/IdeaProjects/spark-nlp/python/example/ocr/pdfs/immortal_text.pdf")

    data.show(10)

    val documentAssembler = new DocumentAssembler().setInputCol("region").setMetadataCol("metadata")

    documentAssembler.transform(data).show()

    succeed

  }

}
