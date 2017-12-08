
package com.johnsnowlabs.ml.logreg

import com.johnsnowlabs.nlp.annotators.assertion.logreg.Windowing
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.{DataFrame, SparkSession}



object I2b2DatasetLogRegTest extends App with Windowing {

  override val before = 10
  override val after = 14

  implicit val spark = SparkSession.builder().appName("i2b2 logreg").master("local[2]").getOrCreate()

  val trainDatasetPath = Seq("/home/jose/Downloads/i2b2/concept_assertion_relation_training_data/partners",
  "/home/jose/Downloads/i2b2/concept_assertion_relation_training_data/beth")

  val testDatasetPath = Seq("/home/jose/Downloads/i2b2/test_data")

  val embeddingsDims = 200
  val embeddingsFile = s"/home/jose/Downloads/bio_nlp_vec/PubMed-shuffle-win-2.bin"
  val reader = new I2b2DatasetReader(embeddingsFile)

  import spark.implicits._
  val trainDataset = reader.readDataFrame(trainDatasetPath).
    withColumn("features", applyWindowUdf(reader.wordVectors.get)($"text", $"target"))
    .select($"features", $"label")

  println("trainDsSize: " +  trainDataset.count)
  val testDataset = reader.readDataFrame(testDatasetPath).
    withColumn("features", applyWindowUdf(reader.wordVectors.get)($"text", $"target"))
    .select($"features", $"label")

  println("testDsSize: " +  testDataset.count)

  val model = train(trainDataset)
  val result = model.transform(testDataset)

  import spark.implicits._
  case class TpFnFp(tp: Int, fn: Int, fp: Int)
  val tpTnFp = result.map ({ r =>
    if (r.getAs[Double]("prediction") == r.getAs[Double]("label")) TpFnFp(1, 0, 0)
    else TpFnFp(0, 1, 1)
  }).collect().reduce((t1, t2) => TpFnFp(t1.tp + t2.tp, t1.fn + t2.fn, t1.fp + t2.fp))

  println(calcStat(tpTnFp.tp + tpTnFp.fn, tpTnFp.tp + tpTnFp.fp, tpTnFp.tp))

  def train(dataFrame: DataFrame) = {
    import spark.implicits._
    val lr = new LogisticRegression()
      .setMaxIter(20)
      .setRegParam(0.002)
      .setElasticNetParam(0.8)
    lr.fit(dataFrame)
  }

  /* TODO put in a common place */
  def calcStat(correct: Long, predicted: Long, predictedCorrect: Long): (Float, Float, Float) = {
    // prec = (predicted & correct) / predicted
    // rec = (predicted & correct) / correct
    val prec = predictedCorrect.toFloat / predicted
    val rec = predictedCorrect.toFloat / correct
    val f1 = 2 * prec * rec / (prec + rec)
    (prec, rec, f1)
  }

  def confusionMatrix[T](predicted: Seq[T], gold: Seq[T]) = {
    val labels = gold.distinct
    import scala.collection.mutable.{Map => MutableMap}
    val matrix : Map[T, MutableMap[T, Int]] =
      labels.map(label => (label, MutableMap(labels.zip(Array.fill(labels.size)(0)): _*))).toMap

    predicted.zip(gold).foreach { case (p, g) =>
        matrix.get(p).get(g) += 1
    }

    /* sanity check */
    if(predicted.length ==matrix.map(map => map._2.values.sum).sum)
      println("looks good")

    matrix
  }


}
