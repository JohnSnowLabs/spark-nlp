package com.johnsnowlabs.util
import com.johnsnowlabs.nlp.functions
import com.johnsnowlabs.nlp.annotator.WordEmbeddingsModel
import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.nlp.base.{DocumentAssembler, Finisher, RecursivePipeline}
import com.johnsnowlabs.nlp.functions.CoverageResult
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.sql.{Row, SparkSession}

object CoverageSample extends App {

  val spark = SparkSession
    .builder()
    .master("local[*]")
    .appName("NGramApp")
    .getOrCreate()

  val pretrainedEmbeddingsName: String = "embeddings_clinical"
  val emptyDataset = PipelineModels.dummyDataset

  //val icd10 = ResourceHelper.spark.read.option("header", "true").csv("icd10cm.csv")
  val test = ResourceHelper.spark.read.option("header", "true").csv("icd10cm.csv").limit(10)

  val documentAssembler = new DocumentAssembler()
    .setInputCol("description")
    .setOutputCol("document")

  val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

  val embeddings = WordEmbeddingsModel.pretrained(pretrainedEmbeddingsName, "en", "clinical/models")
    .setInputCols("document", "token")
    .setOutputCol("embeddings")

  val pipeline = new RecursivePipeline()
    .setStages(Array(
      documentAssembler,
      tokenizer,
      embeddings
    ))

  val readyData = pipeline.fit(test).transform(test).cache()

  val wcov1 = functions.EmbeddingsCoverage(readyData).withCoverageColumn("embeddings", "cov_embeddings1")
  val overallCoverageApprox_ = wcov1.select("cov_embeddings1")
  val blah = overallCoverageApprox_.rdd.flatMap(x => Seq(x)).collect()
  val overallCoverageApprox = overallCoverageApprox_.rdd.flatMap(x => Seq(x.getAs[Row]("cov_embeddings1").getAs[Float]("percentage"))).mean()

  println(overallCoverageApprox)
  println(functions.EmbeddingsCoverage(wcov1).overallCoverage("embeddings").percentage)

}
