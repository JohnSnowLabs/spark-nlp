package com.johnsnowlabs.ml.logreg


import com.johnsnowlabs.nlp.annotators.assertion.logreg.Windowing
import org.apache.spark.ml.classification.{GBTClassifier, LogisticRegression, MultilayerPerceptronClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.{DataFrame, SparkSession}

//shitty spark!
import org.apache.spark.mllib.linalg.{Vectors => MlLibVectors}
import org.apache.spark.ml.linalg.{Vector => MlVector}
import org.apache.spark.mllib.linalg.{Vector => MlLibVector}


object I2b2DatasetLogRegTest extends App with Windowing {

  override val before = 12
  override val after = 12

  implicit val spark = SparkSession.builder().appName("i2b2 logreg").master("local[2]").getOrCreate()

  // directory of the i2b2 dataset
  val i2b2Dir = "/home/jose/Downloads/i2b2"

  val trainDatasetPath = Seq(s"${i2b2Dir}/concept_assertion_relation_training_data/partners"
  , s"${i2b2Dir}/concept_assertion_relation_training_data/beth")

  val testDatasetPath = Seq("/home/jose/Downloads/i2b2/test_data")

  val embeddingsDims = 200
  val embeddingsFile = s"/home/jose/Downloads/bio_nlp_vec/PubMed-shuffle-win-2.bin"
  //val embeddingsFile = s"/home/jose/embeddings/pubmed_i2b2.bin"
  val reader = new I2b2DatasetReader(embeddingsFile)

  import spark.implicits._
  val trainDataset = reader.readDataFrame(trainDatasetPath)
    .withColumn("features", applyWindowUdf(reader.wordVectors.get)($"text", $"target", $"start", $"end"))
    .select($"features", $"label")

  println("trainDsSize: " +  trainDataset.count)
  val testDataset = reader.readDataFrame(testDatasetPath).
    withColumn("features", applyWindowUdf(reader.wordVectors.get)($"text", $"target", $"start", $"end"))
    .select($"features", $"label", $"text", $"target")

  println("testDsSize: " +  testDataset.count)

  val model = train(trainDataset)
  case class TpFnFp(tp: Int, fn: Int, fp: Int)
  import org.apache.spark.mllib.util.MLUtils

  val test = testDataset
    .rdd.map(r => LabeledPoint(r.getAs[Double]("label"),
    r.getAs[MlLibVector]("features")))

  // Compute raw scores on the test set.
  val predictionAndLabels = test.map { case LabeledPoint(label, features) =>
    val prediction = model.predict(features)
    (prediction, label)
  }


  val tpTnFp = predictionAndLabels.map ({ case (pred, label) =>
    if (pred == label) TpFnFp(1, 0, 0)
    else TpFnFp(0, 1, 1)
  }).collect().reduce((t1, t2) => TpFnFp(t1.tp + t2.tp, t1.fn + t2.fn, t1.fp + t2.fp))

  println(calcStat(tpTnFp.tp + tpTnFp.fn, tpTnFp.tp + tpTnFp.fp, tpTnFp.tp))


  /*
  val result = model.transform(testDataset.cache())

  val tpTnFp = result.map ({ r =>
    if (r.getAs[Double]("prediction") == r.getAs[Double]("label")) TpFnFp(1, 0, 0)
    else TpFnFp(0, 1, 1)
  }).collect().reduce((t1, t2) => TpFnFp(t1.tp + t2.tp, t1.fn + t2.fn, t1.fp + t2.fp))

  println(calcStat(tpTnFp.tp + tpTnFp.fn, tpTnFp.tp + tpTnFp.fp, tpTnFp.tp))

  val evaluator = new MulticlassClassificationEvaluator("f1").setMetricName("f1")
  println("Test set f1 = " + evaluator.evaluate(result))

  val badGuys = result.filter(r => r.getAs[Double]("prediction") != r.getAs[Double]("label")).collect()
  println(badGuys)


  val pred = result.select($"prediction").collect.map{ r =>
    r.getAs[Double]("prediction")
  }

  val gold = result.select($"label").collect.map{ r =>
    r.getAs[Double]("label")
  }


  println(confusionMatrix(pred, gold)) */

  def train(dataFrame: DataFrame) = {
/*
    import spark.implicits._
    val lr = new LogisticRegression()
      .setMaxIter(20) //20
      .setRegParam(0.00135) //0.0012
      .setElasticNetParam(0.8) //0.8
      .setTol(1.0)
      .setStandardization(false)

    lr.fit(dataFrame) */

    // Run training algorithm to build the model
    val model = new LogisticRegressionWithLBFGS()
      model.optimizer.setRegParam(0.0013)
      model.setNumClasses(6)

    model.run(dataFrame.rdd
      .map(r => LabeledPoint(r.getAs[Double]("label"),
        r.getAs[MlLibVector]("features"))))

/*
    val layers = Array[Int](5630, 6)

    // 5078, 6 -> 0.8878302

    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100) // 30
      .setTol(1E-6) //1E-5

    trainer.fit(dataFrame.cache())
*/


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
