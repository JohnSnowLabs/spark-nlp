package com.johnsnowlabs.ml.logreg


import com.johnsnowlabs.nlp.annotators.assertion.logreg.{SimpleTokenizer, Tokenizer, Windowing}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{ColumnName, DataFrame, SparkSession}



object I2b2DatasetLogRegTest extends App with Windowing {

  override val before = 11
  override val after = 13
  override val tokenizer: Tokenizer = new SimpleTokenizer

  implicit val spark = SparkSession.builder().appName("i2b2 logreg").master("local[2]").getOrCreate()
  import spark.implicits._

  // directory of the i2b2 dataset
  val i2b2Dir = "/home/jose/Downloads/i2b2"

  val trainDatasetPath = Seq(s"${i2b2Dir}/concept_assertion_relation_training_data/partners"
  , s"${i2b2Dir}/concept_assertion_relation_training_data/beth")

  val testDatasetPath = Seq("/home/jose/Downloads/i2b2/test_data")

  val embeddingsDims = 200
  val embeddingsFile = s"/home/jose/Downloads/bio_nlp_vec/PubMed-shuffle-win-2.bin"
  val reader = new I2b2DatasetReader(embeddingsFile)

  val trainDataset = reader.readDataFrame(trainDatasetPath)
    .withColumn("features", applyWindowUdf(reader.wordVectors.get)($"text", $"target", $"start", $"end"))
    .select($"features", $"label")

  println("trainDsSize: " +  trainDataset.count)
  val testDataset = reader.readDataFrame(testDatasetPath)
    .withColumn("features", applyWindowUdf(reader.wordVectors.get)($"text", $"target", $"start", $"end"))
    .select($"features", $"label", $"text", $"target")

  println("testDsSize: " +  testDataset.count)

  val model = train(trainDataset)
  case class TpFnFp(tp: Int, fn: Int, fp: Int)

  // Compute raw scores on the test set.
  val result = model.transform(testDataset.cache())

  val tpTnFp = result.map ({ r =>
    if (r.getAs[Double]("prediction") == r.getAs[Double]("label")) TpFnFp(1, 0, 0)
    else TpFnFp(0, 1, 1)
  }).collect().reduce((t1, t2) => TpFnFp(t1.tp + t2.tp, t1.fn + t2.fn, t1.fp + t2.fp))

  println(calcStat(tpTnFp.tp + tpTnFp.fn, tpTnFp.tp + tpTnFp.fp, tpTnFp.tp))

  val badGuys = result.filter(r => r.getAs[Double]("prediction") != r.getAs[Double]("label")).collect()
  println(badGuys)

  val pred = result.select($"prediction").collect.map{ r =>
    r.getAs[Double]("prediction")
  }

  val gold = result.select($"label").collect.map{ r =>
    r.getAs[Double]("label")
  }


  println(confusionMatrix(pred, gold))

  def train(dataFrame: DataFrame) = {
    val lr = new LogisticRegression()
      .setMaxIter(26)
      .setRegParam(0.00192)
      .setElasticNetParam(0.9)

    lr.fit(dataFrame)
  }

  def calcStat(correct: Long, predicted: Long, predictedCorrect: Long): (Float, Float, Float) = {
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

    predicted.zip(gold).foreach { case (p, g) => matrix.get(p).get(g) += 1}

    /* sanity check, the confusion matrix should contain as many elements as there were used during training / prediction */
    assert(predicted.length ==matrix.map(map => map._2.values.sum).sum)
    matrix
  }

  // produces a org.apache.spark.ml.linalg.Vector
  def convertToVectorUdf = udf {(array: Array[Double]) =>
      val tmp = Vectors.dense(array)
      tmp
  }


}
