package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder
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

  "ClassifierDL" should "correctly download and load pre-trained model" taggedAs FastTest in {
    val classifierDL = ClassifierDLModel.pretrained("classifierdl_use_trec50")
    classifierDL.getClasses.foreach(x=>print(x+", "))
  }

}
