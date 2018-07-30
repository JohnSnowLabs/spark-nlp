package com.johnsnowlabs.nlp.pretrained
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{AssertionDLModel}
import com.johnsnowlabs.nlp.annotators.assertion.logreg.NegexDatasetReader
import org.apache.spark.ml.{Pipeline}
import org.apache.spark.sql.SparkSession
/**
  * Combine a downloaded tokenizer with a locally created document assembler
  *
  * Created by jose on 22/02/18.
  */
object ModelDownloadSpec extends App {

  implicit val spark = SparkSession.builder().appName("i2b2 logreg").master("local[1]").getOrCreate

  val datasetPath = "rsAnnotations-1-120-random.txt"
  val embeddingsDims = 100

  val reader = new NegexDatasetReader()

  val dataset = "rsAnnotations-1-120-random.txt"

  val ds = reader.readDataframe(datasetPath).cache

  val documentAssembler = new DocumentAssembler()
    .setInputCol("sentence")
    .setOutputCol("document")

  val assertion = ResourceDownloader.downloadModel(AssertionDLModel,"as_full_dl", Some("en"))
  //assertion.setIndexPath("cache_pretrained/as_fast_dl_en_1.5_2_1523537970256/embeddings")

  val pipeline = new Pipeline().setStages(Array(documentAssembler, assertion))

  val pipelineModel = pipeline.fit(ds.cache)
  val result = pipelineModel.transform(ds.cache).collect

  result.foreach { x=>

    println(x)
  }


}
