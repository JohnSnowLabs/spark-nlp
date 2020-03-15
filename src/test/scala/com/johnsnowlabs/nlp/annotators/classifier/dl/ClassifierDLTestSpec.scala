package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.scalatest._

class ClassifierDLTestSpec extends FlatSpec {


  "ClassifierDL" should "correctly train IMDB train dataset" in {

    val smallCorpus = ResourceHelper.spark.read.option("header","true").csv("src/test/resources/classifier/sentiment.csv")

    println("count of training dataset: ", smallCorpus.count)
    smallCorpus.show()
    smallCorpus.printSchema()

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val token = new Tokenizer()
      .setInputCols("document")
      .setOutputCol("tokens")

    val norm = new Normalizer()
      .setInputCols("tokens")
      .setOutputCol("cleaned")

    val newDocs = new TokenAssembler()
      .setInputCols("cleaned")
      .setOutputCol("newDocs")

    val useEmbeddings = UniversalSentenceEncoder.pretrained()
      .setInputCols("newDocs")
      .setOutputCol("sentence_embeddings")

    val docClassifier = new ClassifierDLApproach()
      .setInputCols("sentence_embeddings")
      .setOutputCol("category")
      .setLabelColumn("label")
      .setBatchSize(64)
      .setMaxEpochs(20)
      .setLr(5e-3f)
      .setDropout(0.5f)
    //      .setValidationSplit(0.2f)

    val pipeline = new Pipeline()
      .setStages(
        Array(
          documentAssembler,
          token,
          norm,
          newDocs,
          useEmbeddings,
          docClassifier
        )
      )

    val pipelineModel = pipeline.fit(smallCorpus)

    pipelineModel.transform(smallCorpus).select("document").show(1, false)
    pipelineModel.transform(smallCorpus).select("newDocs").show(1, false)

  }

}
