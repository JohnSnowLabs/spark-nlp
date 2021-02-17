package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.scalatest._

class ClassifierDLTestSpec extends FlatSpec {


  "ClassifierDL" should "correctly train IMDB train dataset" taggedAs SlowTest in {

    val smallCorpus = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/classifier/sentiment.csv")

    println("count of training dataset: ", smallCorpus.count)

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val useEmbeddings = UniversalSentenceEncoder.pretrained()
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    val docClassifier = new ClassifierDLApproach()
      .setInputCols("sentence_embeddings")
      .setOutputCol("category")
      .setLabelColumn("label")
      .setBatchSize(64)
      .setMaxEpochs(20)
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

    pipelineModel.transform(smallCorpus).select("document").show(1, false)

  }

  "ClassifierDL" should "not fail on empty inputs" taggedAs SlowTest in {

    val testData = ResourceHelper.spark.createDataFrame(Seq(
      (1, "This is my first sentence. This is my second."),
      (2, "This is my third sentence. . . . .... ..."),
      (3, "")
    )).toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val sentence = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")

    val useEmbeddings = UniversalSentenceEncoder.pretrained()
      .setInputCols("document")
      .setOutputCol("sentence_embeddings")

    val sarcasmDL = ClassifierDLModel.pretrained(name = "classifierdl_use_sarcasm")
      .setInputCols("sentence_embeddings")
      .setOutputCol("sarcasm")

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        useEmbeddings,
        sarcasmDL
      ))

    val pipelineDF = pipeline.fit(testData).transform(testData)
    pipelineDF.select("sentence.result").show(false)
    pipelineDF.select("sentence_embeddings.result").show(false)
    pipelineDF.select("sarcasm.result").show(false)

    pipelineDF.show()

  }

  "ClassifierDL" should "correctly download and load pre-trained model" taggedAs FastTest in {
    val classifierDL = ClassifierDLModel.pretrained("classifierdl_use_trec50")
    classifierDL.getClasses.foreach(x=>print(x+", "))
  }

}
