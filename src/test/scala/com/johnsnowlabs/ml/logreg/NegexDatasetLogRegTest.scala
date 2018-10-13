package com.johnsnowlabs.ml.logreg

import java.io.File

import com.johnsnowlabs.ml.common.EvaluationMetrics
import com.johnsnowlabs.nlp.annotators.assertion.logreg.{NegexDatasetReader, SimpleTokenizer, Tokenizer, Windowing}
import com.johnsnowlabs.nlp.embeddings.{WordEmbeddingsRetriever, WordEmbeddingsIndexer}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Test on simple dataset from NegEx, can be obtained from,
  * https://raw.githubusercontent.com/mongoose54/negex/master/genConText/rsAnnotations-1-120-random.txt
  * Word Embeddings can be obtained from,
  * https://github.com/cambridgeltl/BioNLP-2016
  */
object NegexDatasetLogRegTest extends App with Windowing with EvaluationMetrics {

  override val before: Int = 10
  override val after: Int = 10
  override val tokenizer: Tokenizer = new SimpleTokenizer

  /* local Spark for test */
  implicit val spark = SparkSession.builder().
    appName("Simple Assertion Status text on Negex dataset").
    master("local[4]").getOrCreate()

  import spark.implicits._
  val datasetPath = "rsAnnotations-1-120-random.txt"

  val embeddingsDims = 200
  val embeddingsFile = s"PubMed-shuffle-win-2.bin"
  val fileDb = embeddingsFile + ".db"

  override lazy val wordVectors: WordEmbeddingsRetriever = Option(embeddingsFile).map {
    wordEmbeddingsFile =>
      require(new File(embeddingsFile).exists())
      val fileDb = wordEmbeddingsFile + ".db"
      if (!new File(fileDb).exists())
        WordEmbeddingsIndexer.indexBinary(wordEmbeddingsFile, fileDb)
  }.filter(_ => new File(fileDb).exists())
    .map(_ => WordEmbeddingsRetriever(fileDb, embeddingsDims, caseSensitive = true))
    .getOrElse(throw new Exception("Embeddings file not found"))

  val mappings = Map("Affirmed" -> 0.0, "Negated" -> 1.0)
  val reader = new NegexDatasetReader()

  val ds = reader.readDataframe(datasetPath)
    .withColumn("features", applyWindowUdf($"sentence", $"target", $"start", $"end"))
    .withColumn("label", labelToNumber($"label"))
    .select($"features", $"label").cache()

  // Split the data into training and test sets (30% held out for testing).
  val Array(trainingData, testData) = ds.randomSplit(Array(0.7, 0.3))

  val model = train(trainingData)
  val result = model.transform(testData)

  val pred = result.map(r => r.getAs[Double]("prediction")).collect
  val gold = result.map(r => r.getAs[Double]("label")).collect

  println(calcStat(pred, gold))
  println(confusionMatrix(pred, gold))

  def train(dataFrame: DataFrame) = {
    import spark.implicits._
    val lr = new LogisticRegression()
      .setMaxIter(8)
      .setRegParam(0.01)
      .setElasticNetParam(0.8)
    lr.fit(dataFrame)
  }

  def labelToNumber = udf { label:String  => mappings.get(label)}

}