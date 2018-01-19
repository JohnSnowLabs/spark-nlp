package com.johnsnowlabs.ml.logreg

import com.johnsnowlabs.ml.common.EvaluationMetrics
import com.johnsnowlabs.nlp.annotators.assertion.logreg.{SimpleTokenizer, Tokenizer, Windowing}
import com.johnsnowlabs.nlp.embeddings.WordEmbeddings
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import smile.classification.SVM
import smile.classification.SVM.Multiclass
import smile.math.kernel.GaussianKernel

object I2b2DatasetSVMTestSmile extends App with Windowing with EvaluationMetrics {

  override val before = 11
  override val after = 13
  override val tokenizer: Tokenizer = new SimpleTokenizer
  override lazy val wordVectors: Option[WordEmbeddings] = reader.wordVectors

  implicit val spark = SparkSession.builder().appName("i2b2 logreg").master("local[2]").getOrCreate()
  import spark.implicits._

  val mappings = Map("hypothetical" -> 0.0,
    "present" -> 1.0, "absent" -> 2.0, "possible" -> 3.0,
    "conditional"-> 4.0, "associated_with_someone_else" -> 5.0)

  // directory of the i2b2 dataset
  val i2b2Dir = "/home/jose/Downloads/i2b2"

  val trainDatasetPath = Seq(s"${i2b2Dir}/concept_assertion_relation_training_data/partners",
    s"${i2b2Dir}/concept_assertion_relation_training_data/beth")

  val testDatasetPath = Seq(s"$i2b2Dir/test_data")

  val embeddingsDims = 200
  // word embeddings location
  val embeddingsFile = s"/home/jose/Downloads/bio_nlp_vec/PubMed-shuffle-win-2.bin"
  //val embeddingsFile = s"/home/jose/wordembeddings/pubmed_i2b2.bin"
  val reader = new I2b2DatasetReader(wordEmbeddingsFile = embeddingsFile, targetLengthLimit = 8)
  val trainAnnotations = reader.read(trainDatasetPath)
  var trainDataset = trainAnnotations.map(applyWindow)
  val trainLabels = trainAnnotations.map(ann => mappings.get(ann.label).get.toInt)
  println("trainDsSize: " +  trainDataset.size)


  val granges:List[Double] = List(1.0, 10.0)
  val cranges:List[Double] = List(1.0)
  for (gamma <- granges; c <-cranges) {
    val model = train(trainDataset.toArray, trainLabels.toArray, gamma, c)
    val testAnnotations = reader.read(testDatasetPath)
    val testDataset = testAnnotations.map(applyWindow)
    var testLabels = testAnnotations.map(ann => mappings.get(ann.label).get.toInt)
    println("testDsSize: " + testDataset.size)

    // Compute raw scores on the test set.
    val pred: Seq[Int] = testDataset.par.map(model.predict).toList
    val gold = testLabels
    println(calcStat(pred, gold))
    println(confusionMatrix(pred, gold))
  }

  def train(dataset: Array[Array[Double]], labels: Array[Int], gamma:Double, C:Double) = {
    println(gamma, C)
    val svm = new SVM(new GaussianKernel(gamma), C, 6, Multiclass.ONE_VS_ONE)
    1 to 7 foreach{_ =>
      svm.learn(dataset, labels)
    }
    svm.finish()
    svm
  }

  def labelToNumber = udf { label:String  => mappings.get(label)}

  /* TODO improve this */
  import org.apache.spark.sql.Row
  def toFeatureVector(rdd: RDD[Row]):Array[Array[Double]] = rdd.map(r =>
    r.getAs[Vector]("features").toArray).collect

  def toLabels(rdd: RDD[Row]): Array[Int] = rdd.map(r => r.getAs[Double]("label").toInt).collect
}
