package com.johnsnowlabs.ml.logreg

import com.johnsnowlabs.ml.logreg.I2b2DatasetLogRegTest.calcStat
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.RegexTokenizer
import com.johnsnowlabs.nlp.annotators.assertion.logreg.AssertionLogRegApproach
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronApproach
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetectorModel
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.SparkSession

object I2b2DatasetPipelineTest extends App {

  implicit val spark = SparkSession.builder().appName("i2b2 logreg").master("local[4]")
        .config("spark.executor.memory", "2g").getOrCreate

  import spark.implicits._
  val trainPaths = Seq("/home/jose/Downloads/i2b2/concept_assertion_relation_training_data/partners"
  ,"/home/jose/Downloads/i2b2/concept_assertion_relation_training_data/beth")
  val testPaths = Seq("/home/jose/Downloads/i2b2/test_data")

  val embeddingsFile = s"/home/jose/Downloads/bio_nlp_vec/PubMed-shuffle-win-2.bin"

  def getAssertionStages(): Array[_ <: PipelineStage] = {

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val tokenizer = new RegexTokenizer()
      .setInputCols(Array("document"))
      .setOutputCol("token")

    val posTagger = new PerceptronApproach()
      .setCorpusPath("/anc-pos-corpus/")
      .setNIterations(10)
      .setInputCols("token", "document")
      .setOutputCol("pos")

    val assertionStatus = new AssertionLogRegApproach()
      .setInputCols("document", "pos")
      .setOutputCol("assertion")
      .setEmbeddingsSource(embeddingsFile, 200, WordEmbeddingsFormat.Binary)
      .setEmbeddingsFolder("/home/jose/Downloads/bio_nlp_vec")

    Array(documentAssembler,
      tokenizer,
      posTagger,
      assertionStatus)

  }

  val reader = new I2b2DatasetReader(embeddingsFile)

  def trainAssertionModel(paths: Seq[String]): PipelineModel = {
    System.out.println("Train Dataset Reading")
    val time = System.nanoTime()
    val dataset = reader.readDataFrame(paths)
    System.out.println(s"Done, ${(System.nanoTime() - time)/1e9}\n")
    System.out.println("Start fitting")

    // train Assertion Status
    val pipeline = new Pipeline()
      .setStages(getAssertionStages)

    pipeline.fit(dataset)
  }

  def testAssertionModel(path:Seq[String], model: PipelineModel) = {
    System.out.println("Test Dataset Reading")
    val dataset = reader.readDataFrame(path)
    model.transform(dataset)
  }



  val model = trainAssertionModel(trainPaths)
  val result = testAssertionModel(testPaths, model)

  /* TODO all this to common place */
  import spark.implicits._
  case class TpFnFp(tp: Int, fn: Int, fp: Int)
  val tpFnFp = result.map ({ r =>
    if (r.getAs[Double]("prediction") == r.getAs[Double]("label")) TpFnFp(1, 0, 0)
    else TpFnFp(0, 1, 1)
  }).collect().reduce((t1, t2) => TpFnFp(t1.tp + t2.tp, t1.fn + t2.fn, t1.fp + t2.fp))

  println(calcStat(tpFnFp.tp + tpFnFp.fn, tpFnFp.tp + tpFnFp.fp, tpFnFp.tp))


}
