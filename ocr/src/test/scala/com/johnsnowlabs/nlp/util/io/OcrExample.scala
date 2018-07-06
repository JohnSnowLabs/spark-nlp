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

    // point to test/resources/pdfs
    val data = OcrHelper.createDataset(spark, "ocr/src/test/resources/pdfs")

    data.show(10)

    val documentAssembler = new DocumentAssembler().setInputCol("region").setMetadataCol("metadata")

    documentAssembler.transform(data).show()

    succeed

  }

}
