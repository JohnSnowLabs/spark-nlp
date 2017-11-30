package com.johnsnowlabs.ml.logreg

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Created by jose on 24/11/17.
  */
object NegexDatasetLogRegTest extends App {

  /* local Spark for test */
  implicit val spark = SparkSession.builder().appName("DataFrame-UDF").master("local[4]").getOrCreate()
  val datasetPath = "rsAnnotations-1-120-random.txt.csv"

  val embeddingsDims = 200
  val embeddingsFile = s"/home/jose/Downloads/bio_nlp_vec/PubMed-shuffle-win-2.bin"
  val reader = new NegexDatasetReader(embeddingsFile, embeddingsDims)

  def train(datasetPath: String) = {
    import spark.implicits._
    val ds = reader.readNegexDataset(datasetPath)

    val lr = new LogisticRegression()
      .setMaxIter(8)
      .setRegParam(0.01)
      .setElasticNetParam(0.8)
    lr.fit(ds)
  }

  val model = train(datasetPath)

  // test on train data, just as a 'smoke test'
  val ds = reader.readNegexDataset(datasetPath)
  val result = model.transform(ds)
  val total = result.count
  val correct = result.filter(r => r.getAs[Double]("prediction") == r.getAs[Double]("label")).count

  println("Accuracy: " + correct.toDouble / total.toDouble)
  println(s"Coefficients: ${model.coefficients} Intercept: ${model.intercept}")
}
