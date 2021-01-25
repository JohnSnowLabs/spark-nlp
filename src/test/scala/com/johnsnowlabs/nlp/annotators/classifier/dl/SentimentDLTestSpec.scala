package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest._

class SentimentDLTestSpec extends FlatSpec {
  val spark = ResourceHelper.spark

  "SentimentDL" should "correctly train on a test dataset" taggedAs SlowTest in {

    val smallCorpus = ResourceHelper.spark.read.option("header", "true").csv("src/test/resources/classifier/sentiment.csv")

    smallCorpus.show
    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val useEmbeddings = UniversalSentenceEncoder.pretrained()
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    val docClassifier = new SentimentDLApproach()
      .setInputCols("sentence_embeddings")
      .setOutputCol("sentiment")
      .setLabelColumn("label")
      .setBatchSize(32)
      .setMaxEpochs(1)
      .setLr(5e-3f)
      .setDropout(0.5f)

    val pipeline = new Pipeline()
      .setStages(
        Array(
          documentAssembler,
          useEmbeddings,
          docClassifier
        )
      )

    val pipelineModel = pipeline.fit(smallCorpus)
    pipelineModel.stages.last.asInstanceOf[SentimentDLModel].write.overwrite().save("./tmp_sentimentDL_model")

    val pipelineDF = pipelineModel.transform(smallCorpus)
    pipelineDF.select("document").show(1)
    pipelineDF.select("sentiment").show(1)
    pipelineDF.select("sentiment.result").show(1, false)
    pipelineDF.select("sentiment.metadata").show(1, false)

  }

}
