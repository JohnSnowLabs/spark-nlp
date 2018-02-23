package com.johnsnowlabs.pretrained

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.{DocumentAssembler, annotators}
import com.johnsnowlabs.pretrained.en.models.Tokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
import com.johnsnowlabs.util.Version
import com.johnsnowlabs.pretrained.en.models._

/**
  * Created by jose on 22/02/18.
  */
object PipelineDownloadSpec extends App{

  implicit val spark = SparkSession.builder().appName("Remote Pipelines Test").master("local[1]").getOrCreate
  import spark.implicits._

  val dataset = Seq("Songs are to be sung", "Dances are to be danced").toDF("text")

  val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

  val pipeline = new Pipeline().setStages(
    Array(
      documentAssembler,
      tokenizer)
  )

  val fittedPipeline = pipeline.fit(dataset)
  val transformed1 = fittedPipeline.transform(dataset)

  val libVersion = Some(Version(1))
  val sparkVersion = Some(Version.parse(spark.version).take(1))
  val folder = "pipeline"

  //TrainingHelper.saveModel("pipeline_std", Some("en"), libVersion, sparkVersion, fittedPipeline.write, folder)

  // now some magic brings this to S3 bucket ... and we can download it
  val downloadedPipeline = ResourceDownloader.downloadPipeline("pipeline_std", Some("en"))
  val transformed2 = downloadedPipeline.transform(dataset)

  assert(transformed1.collect == transformed2.collect)


}
