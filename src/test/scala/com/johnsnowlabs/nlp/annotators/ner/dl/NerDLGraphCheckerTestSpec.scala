package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.nlp.annotator.{DistilBertEmbeddings, WordEmbeddings, WordEmbeddingsModel}
import com.johnsnowlabs.nlp.embeddings.HasEmbeddingsProperties
import com.johnsnowlabs.nlp.training.CoNLL
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorBuilder, AnnotatorModel}
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterEach
import org.scalatest.flatspec.AnyFlatSpec
import com.johnsnowlabs.util.TestUtils.captureOutput

import scala.util.{Failure, Success, Try}

class NerDLGraphCheckerTestSpec extends AnyFlatSpec with BeforeAndAfterEach {

  lazy private val conll = CoNLL(explodeSentences = false)
  lazy private val testingData =
    conll.readDataset(ResourceHelper.spark, "src/test/resources/ner-corpus/test_ner_dataset.txt")

  private def getGraphChecker(embeddings: AnnotatorModel[_] with HasEmbeddingsProperties) = {
    val nerDLGraphChecker = new NerDLGraphChecker()
      .setInputCols("sentence", "token")
      .setLabelColumn("label")
      .setEmbeddingsModel(embeddings)
    nerDLGraphChecker
  }

  class MockWordEmbeddingsModel(val embeddingsModel: WordEmbeddingsModel)
      extends WordEmbeddingsModel {

    setDimension(embeddingsModel.getDimension)
    setInputCols(embeddingsModel.getInputCols)
    setOutputCol(embeddingsModel.getOutputCol)
    setStorageRef(embeddingsModel.getStorageRef)

    override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
      fail("Embeddings should not be evaluated.")
    }
  }

  lazy private val originalEmbeddings = AnnotatorBuilder.getGLoveEmbeddings(testingData.toDF())
  lazy private val embeddings = new MockWordEmbeddingsModel(originalEmbeddings)

  lazy private val embeddingsInvalid = new MockWordEmbeddingsModel(
    new WordEmbeddings()
      .setStoragePath("src/test/resources/ner-corpus/embeddings.100d.test.txt", "TEXT")
      .setDimension(101)
      .setInputCols("sentence", "token")
      .setOutputCol("embeddings")
      .setStorageRef("embeddings_ner_100_invalid")
      .fit(testingData))

  class MockNerDLApproach extends NerDLApproach {
    override def beforeTraining(spark: SparkSession): Unit = {
      fail(
        "Should not be called during this test. The graph should've been checked and an exception thrown before.")
    }
  }

  lazy val mockNer: NerDLApproach = new MockNerDLApproach()
    .setInputCols("sentence", "token", "embeddings")
    .setOutputCol("ner")
    .setLabelColumn("label")
    .setOutputCol("ner")
    .setLr(1e-3f) // 0.001
    .setPo(5e-3f) // 0.005
    .setDropout(5e-1f) // 0.5
    .setMaxEpochs(1)
    .setRandomSeed(0)
    .setVerbose(0)
    .setBatchSize(8)
    .setEvaluationLogExtended(true)
    .setGraphFolder("src/test/resources/graph/")
    .setUseBestModel(true)

  behavior of "NerDLGraphChecker"

  it should "find the right graphs" taggedAs FastTest in {
    val nerDLGraphChecker = getGraphChecker(embeddings)

    val pipeline = new Pipeline().setStages(Array(embeddings, nerDLGraphChecker))

    val output = captureOutput {
      pipeline.fit(testingData)
    }

    println(output)
    assert(output.contains("found suitable graph"))
  }

  it should "throw an exception if the graph is not found" taggedAs FastTest in {
    val nerDLGraphChecker = getGraphChecker(embeddingsInvalid)

    val invalidPipeline = new Pipeline().setStages(Array(embeddingsInvalid, nerDLGraphChecker))

    Try {
      invalidPipeline.fit(testingData)
    } match {
      case Failure(exception) =>
        assert(
          exception.getMessage.contains("Could not find a suitable tensorflow graph"),
          "Should throw an exception if no suitable graph is found.")
      case Success(_) =>
        fail("Should not be able to fit the model with non-existing graph params.")
    }
  }

  it should "be serializable in a pipeline" taggedAs FastTest in {
    val nerDLGraphChecker = getGraphChecker(originalEmbeddings)

    val pipeline = new Pipeline().setStages(Array(nerDLGraphChecker, originalEmbeddings))

    pipeline.write.overwrite().save("tmp_nerdlgraphchecker_pipeline")
    val loadedPipeline = Pipeline.load("tmp_nerdlgraphchecker_pipeline")
    val pipelineModel = loadedPipeline.fit(testingData)

    pipelineModel.write.overwrite().save("tmp_nerdlgraphchecker_pipeline_model")
    val loadedPipelineModel = PipelineModel.load("tmp_nerdlgraphchecker_pipeline_model")
    loadedPipelineModel.transform(testingData).show()
  }

  it should "determine a suitable graph before training" taggedAs FastTest in {
    val nerDLGraphChecker: NerDLGraphChecker = getGraphChecker(embeddingsInvalid)

    val invalidPipeline =
      new Pipeline().setStages(Array(embeddingsInvalid, nerDLGraphChecker, mockNer))

    Try {
      invalidPipeline.fit(testingData)
    } match {
      case Failure(exception) =>
        assert(
          exception.getMessage.contains("Could not find a suitable tensorflow graph"),
          "Should throw an exception if no suitable graph is found.")
      case Success(_) =>
        fail("Should not be able to fit the model with non-existing graph params.")
    }
  }

  it should "determine graph size with batch process annotators" taggedAs SlowTest in {
    class MockDistilBert(val embeddingsModel: DistilBertEmbeddings) extends DistilBertEmbeddings {

      setDimension(
        embeddingsModel.getDimension + 3
      ) // Intentionally incorrect dimension so graph checker fails
      setInputCols(embeddingsModel.getInputCols)
      setOutputCol(embeddingsModel.getOutputCol)

      override def batchAnnotate(
          batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {
        fail("Embeddings should not be evaluated.")
      }
    }

    val mockDistilBert = new MockDistilBert(
      DistilBertEmbeddings
        .pretrained("distilbert_base_cased", "en")
        .setInputCols("sentence", "token")
        .setOutputCol("embeddings"))
    val nerDLGraphChecker = getGraphChecker(mockDistilBert)

    // IMPORTANT: the order of the stages matters here, graph checker MUST come before embeddings
    val pipeline = new Pipeline().setStages(Array(nerDLGraphChecker, mockDistilBert, mockNer))

    Try {
      pipeline.fit(testingData)
    } match {
      case Failure(exception) =>
        assert(
          exception.getMessage.contains("Could not find a suitable tensorflow graph"),
          "Should throw an exception if no suitable graph is found.")
      case Success(_) =>
        fail("Should not be able to fit the model with non-existing graph params.")
    }
  }
}
