package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.{DocumentAssembler, LightPipeline}
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class MPNetForSequenceClassificationTestSpec extends AnyFlatSpec {

  import spark.implicits._

  lazy val document = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  lazy val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

  lazy val sequenceClassifier = {
    MPNetForSequenceClassification
      .pretrained()
      .setInputCols(Array("document", "token"))
      .setOutputCol("label")
      .setBatchSize(2)
  }

  lazy val texts: Seq[String] = Seq(
    "I love driving my car.",
    "The next bus will arrive in 20 minutes.",
    "pineapple on pizza is the worst 🤮")
  lazy val data = texts.toDF("text")

  lazy val pipeline = new Pipeline().setStages(Array(document, tokenizer, sequenceClassifier))

  behavior of "MPNetForSequenceClassification"

  it should "correctly classify" taggedAs SlowTest in {
    val pipelineModel = pipeline.fit(data)
    val pipelineDF = pipelineModel.transform(data)

    val results = Annotation.collect(pipelineDF, "label").head.map(_.getResult)

    val expected = Seq("TRANSPORT/CAR", "TRANSPORT/MOVEMENT", "FOOD")

    expected.zip(results).map { case (expectedLabel, res) =>
      assert(expectedLabel == res, "Wrong label")
    }
  }

  it should "be serializable" taggedAs SlowTest in {

    val pipelineModel = pipeline.fit(data)
    pipelineModel.stages.last
      .asInstanceOf[MPNetForSequenceClassification]
      .write
      .overwrite()
      .save("./tmp_mpnet_seq_classification")

    val loadedModel = MPNetForSequenceClassification.load("./tmp_mpnet_seq_classification")
    val newPipeline: Pipeline =
      new Pipeline().setStages(Array(document, tokenizer, loadedModel))

    val pipelineDF = newPipeline.fit(data).transform(data)

    val results = Annotation.collect(pipelineDF, "label").head.map(_.getResult)

    val expected = Seq("TRANSPORT/CAR", "TRANSPORT/MOVEMENT", "FOOD")

    expected.zip(results).map { case (expectedLabel, res) =>
      assert(expectedLabel == res, "Wrong label")
    }
  }

  it should "be compatible with LightPipeline" taggedAs SlowTest in {
    val pipeline: Pipeline =
      new Pipeline().setStages(Array(document, tokenizer, sequenceClassifier))

    val pipelineModel = pipeline.fit(data)
    val lightPipeline = new LightPipeline(pipelineModel)
    val results = lightPipeline.fullAnnotate(texts.toArray)

    results.foreach { result =>
      println(result("label"))
      assert(result("document").nonEmpty)
      assert(result("token").nonEmpty)
      assert(result("label").nonEmpty)
    }
  }

}
