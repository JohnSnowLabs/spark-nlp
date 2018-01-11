package com.johnsnowlabs.ml.logreg


import com.johnsnowlabs.ml.common.EvaluationMetrics
import com.johnsnowlabs.nlp.annotators.assertion.logreg.{RegexTokenizer, SimpleTokenizer, Tokenizer, Windowing}
import com.johnsnowlabs.nlp.annotators.assertion.logreg.mllib.vectors
import com.johnsnowlabs.nlp.embeddings.WordEmbeddings
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, ColumnName, DataFrame, SparkSession}

object I2b2DatasetLogRegTest extends App with Windowing with EvaluationMetrics {

  override val before = 11
  override val after = 13
  override val tokenizer: Tokenizer = new RegexTokenizer
  override lazy val wordVectors: Option[WordEmbeddings] = reader.wordVectors

  implicit val spark = SparkSession.builder().appName("i2b2 logreg").master("local[2]").getOrCreate()
  import spark.implicits._

  val mappings = Map("hypothetical" -> 0.0,
    "present" -> 1.0, "absent" -> 2.0, "possible" -> 3.0,
    "conditional"-> 4.0, "associated_with_someone_else" -> 5.0)

  // directory of the i2b2 dataset
  val i2b2Dir = "/home/jose/Downloads/i2b2"

  val trainDatasetPath = Seq(s"${i2b2Dir}/concept_assertion_relation_training_data/partners"
  , s"${i2b2Dir}/concept_assertion_relation_training_data/beth")

  val testDatasetPath = Seq(s"$i2b2Dir/test_data")

  val embeddingsDims = 200
  // word embeddings location
  val embeddingsFile = s"/home/jose/Downloads/bio_nlp_vec/PubMed-shuffle-win-2.bin"
  //val embeddingsFile = s"/home/jose/wordembeddings/pubmed_i2b2.bin"
  val reader = new I2b2DatasetReader(wordEmbeddingsFile = embeddingsFile, targetLengthLimit = 8)


  val windowUDF = applyWindowUdf(vectors)
  val trainDataset = reader.readDataFrame(trainDatasetPath)
    .withColumn("features", windowUDF($"text", $"target", $"start", $"end"))
    .withColumn("label", labelToNumber($"label"))
    .select($"features", $"label")

  println("trainDsSize: " +  trainDataset.count)
  val testDataset = reader.readDataFrame(testDatasetPath)
    .withColumn("features", windowUDF($"text", $"target", $"start", $"end"))
    .withColumn("label", labelToNumber($"label"))
    .select($"features", $"label", $"text", $"target")

  println("testDsSize: " +  testDataset.count)

  val model = train(trainDataset.cache())

  // Compute raw scores on the test set.
  val pred = model.predict(testDataset.rdd.cache().map(r => r.getAs[Vector]("features"))).collect
  val gold = testDataset.select($"label").map(r => r.getAs[Double]("label")).collect
  println(calcStat(pred, gold))
  println(confusionMatrix(pred, gold))

  def train(dataFrame: DataFrame) = {
    val lr = new LogisticRegressionWithLBFGS().setNumClasses(6)
    lr.optimizer.setRegParam(0.00192)
    val lblpt = toLabeledPoint(dataFrame.rdd.cache())
    lr.run(lblpt)
  }


  def labelToNumber = udf { label:String  => mappings.get(label)}

  /* TODO improve this */
  import org.apache.spark.sql.Row
  def toLabeledPoint(rdd: RDD[Row]) = rdd.map(r => LabeledPoint(r.getAs[Double]("label"),
    r.getAs[Vector]("features")))
}
