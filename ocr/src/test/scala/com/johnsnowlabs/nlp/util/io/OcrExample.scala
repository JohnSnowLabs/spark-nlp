package com.johnsnowlabs.nlp.util.io

import com.johnsnowlabs.nlp.{DocumentAssembler, LightPipeline}
import org.apache.spark.ml.Pipeline
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

    import spark.implicits._

    // point to test/resources/pdfs
    val data = OcrHelper.createDataset(spark, "ocr/src/test/resources/pdfs/", "region", "metadata")

    data.show(10)

    val documentAssembler = new DocumentAssembler().setInputCol("region").setMetadataCol("metadata")

    documentAssembler.transform(data).show()

    val raw = OcrHelper.createMap("ocr/src/test/resources/pdfs/")

    val pipeline = new LightPipeline(new Pipeline().setStages(Array(documentAssembler)).fit(Seq.empty[String].toDF("region")))

    val result = pipeline.annotate(raw.values.toArray)

    assert(raw.size == 2 && result.nonEmpty)

    println(result.mkString(","))

    succeed

  }

}
