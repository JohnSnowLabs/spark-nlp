package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.{col, udf}
import org.scalatest._

class MultiClassifierDLTestSpec extends FlatSpec {

  val spark = ResourceHelper.getActiveSparkSession

  "MultiClassifierDL" should "correctly train E2E Challenge" taggedAs SlowTest in {
    def splitAndTrim = udf { labels: String =>
      labels.split(", ").map(x=>x.trim)
    }

    val smallCorpus = spark.read
      .option("header", true)
      .option("inferSchema", true)
      .option("mode", "DROPMALFORMED")
      .csv("src/test/resources/classifier/e2e.csv")
      .withColumn("labels", splitAndTrim(col("mr")))
      .drop("mr")

    println("count of training dataset: ", smallCorpus.count)
    smallCorpus.select("labels").show()
    smallCorpus.printSchema()

    val documentAssembler = new DocumentAssembler()
      .setInputCol("ref")
      .setOutputCol("document")
      .setCleanupMode("shrink")

    val embeddings = UniversalSentenceEncoder.pretrained()
      .setInputCols("document")
      .setOutputCol("embeddings")

    val docClassifier = new MultiClassifierDLApproach()
      .setInputCols("embeddings")
      .setOutputCol("category")
      .setLabelColumn("labels")
      .setBatchSize(128)
      .setMaxEpochs(10)
      .setLr(1e-3f)
      .setThreshold(0.5f)
      .setValidationSplit(0.1f)

    val pipeline = new Pipeline()
      .setStages(
        Array(
          documentAssembler,
          embeddings,
          docClassifier
        )
      )

    val pipelineModel = pipeline.fit(smallCorpus)

  }

}