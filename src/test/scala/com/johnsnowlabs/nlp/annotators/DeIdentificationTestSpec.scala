package com.johnsnowlabs.nlp.annotators

import org.scalatest.FlatSpec
import com.johnsnowlabs.nlp.{DocumentAssembler, RecursivePipeline, SparkAccessor}
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.ner.NerConverter
import com.johnsnowlabs.nlp.annotators.ner.dl.{NerDLApproach, NerDLModel}
import com.johnsnowlabs.nlp.annotators.ner.crf.{NerCrfApproach, NerCrfModel}
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import com.johnsnowlabs.util.PipelineModels
import io.netty.handler.codec.spdy.DefaultSpdyDataFrame
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, SparkSession}

class DeIdentificationTestSpec extends FlatSpec with DeIdentificationBehaviors {

  private val emptyDataset = PipelineModels.dummyDataset
  private var nerDlModel = new NerDLModel()
  private var nerCrfModel = new NerCrfModel()

  private val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  private val sentenceDetector = new SentenceDetector()
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

  private val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

  private val nerTagger = NerDLModel.pretrained()

  private val deIdentification = new DeIdentification()
    .setInputCols("ner")
    .setOutputCol("dei")


  //Fixture creation methods
  def trainNerDlModel(trainDatasetPath: String): NerDLModel = {

    val nerTagger = new NerDLApproach()
      .setInputCols("sentence", "token")
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMaxEpochs(5)
      .setEmbeddingsSource("/Users/dburbano/Documents/JSL/Corpus/glove.6B/glove.6B.100d.txt",
        100, WordEmbeddingsFormat.TEXT)
      .setExternalDataset(trainDatasetPath)
      .setRandomSeed(0)
      .setVerbose(2)

    val pipeline = new RecursivePipeline().setStages(
      Array(documentAssembler,
        sentenceDetector,
        tokenizer,
        nerTagger
      )).fit(emptyDataset)

    pipeline.stages.last.asInstanceOf[NerDLModel]

  }

  def trainNerCRFModel(trainDatasetPath: String): NerCrfModel = {

    val posTagger = PerceptronModel.pretrained()

    val nerTagger = new NerCrfApproach()
      //.setInputCols("document", "token", "pos")
      .setInputCols("sentence", "token", "pos")
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMinEpochs(5)
      .setMaxEpochs(15)
      .setLossEps(1e-3)
      //.setEmbeddingsSource("/Users/dburbano/Documents/JSL/Corpus/glove.6B/glove.6B.100d.txt", 100, WordEmbeddingsFormat.TEXT)
      .setEmbeddingsSource("/Users/dburbano/Documents/JSL/Corpus/PubMed-shuffle-win-2.bin",
        200, WordEmbeddingsFormat.BINARY)
      .setExternalDataset(trainDatasetPath)
      .setL2(1)
      .setC0(1250000)
      .setRandomSeed(0)
      .setVerbose(2)

    val pipeline = new RecursivePipeline().setStages(
      Array(documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        nerTagger
      )).fit(emptyDataset)

    pipeline.stages.last.asInstanceOf[NerCrfModel]

  }


  "An NER with DL model" should "train de-identification entities" ignore  {
    nerDlModel = trainNerDlModel("/Users/dburbano/PycharmProjects/De-Identification/data/train_dataset_small.csv")
    assert(nerDlModel.isInstanceOf[NerDLModel])
  }

  it should behave like saveModel(nerDlModel.write, "/Users/dburbano/tmp/ner_dl_model")

  it should "load NER DL Model" ignore {
    val loadedNerDlModel = NerDLModel.read.load("/Users/dburbano/tmp/ner_dl_model")
    assert(loadedNerDlModel.isInstanceOf[NerDLModel])
  }

  "A NER with CRF model" should "train de-identification entities" ignore  {
    nerCrfModel = trainNerCRFModel("/Users/dburbano/PycharmProjects/De-Identification/data/train_dataset_medium.csv")
    assert(nerCrfModel.isInstanceOf[NerCrfModel])
  }

  it should behave like saveModel(nerCrfModel.write, "/Users/dburbano/tmp/ner_crf_model")

  it should "load model" in {
    nerCrfModel = NerCrfModel.read.load("/Users/dburbano/tmp/ner_crf_model")
    assert(nerCrfModel.isInstanceOf[NerCrfModel])
  }

  private var deIdentificationDataFrame = PipelineModels.dummyDataset

  "A de-identification annotator (using NER with DL)" should "return a spark dataframe" in {

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        nerTagger,
        deIdentification
      )).fit(emptyDataset)

    import SparkAccessor.spark.implicits._

    val testDataset = Seq(
      "Bob visited Switzerland a couple of years ago",
      "Rapunzel let down her long golden hair",
      "money market fund in Canada"
    ).toDS.toDF("text")

    deIdentificationDataFrame = pipeline.transform(testDataset)
    //deIdentificationDataFrame.show(false)
    assert(deIdentificationDataFrame.isInstanceOf[DataFrame])

  }

  it should behave like deIdentificationAnnotator(deIdentification)

}
