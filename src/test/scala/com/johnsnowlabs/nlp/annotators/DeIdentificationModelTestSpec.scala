package com.johnsnowlabs.nlp.annotators

import org.scalatest.FlatSpec
import com.johnsnowlabs.nlp.{Annotation, DocumentAssembler, RecursivePipeline, SparkAccessor}
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.ner.NerConverter
import com.johnsnowlabs.nlp.annotators.ner.dl.{NerDLApproach, NerDLModel}
import com.johnsnowlabs.nlp.annotators.ner.crf.{NerCrfApproach, NerCrfModel}
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import com.johnsnowlabs.util.PipelineModels
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import SparkAccessor.spark.implicits._
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT}


class DeIdentificationModelTestSpec extends FlatSpec with DeIdentificationBehaviors {

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

  private val nerConverter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_con")

  private val deIdentification = new DeIdentificationModel()
    .setInputCols(Array("ner_con", "document"))
    .setOutputCol("dei")

  private val testDataset = Seq(
    "Bob visited Switzerland a couple of years ago",
    "Rapunzel let down her long golden hair",
    "money market fund in Canada"
  ).toDS.toDF("text")

  private val testANDataset = Seq(
    "John was born on 05/10/1975",
    "Bob visited Switzerland a couple of years ago",
    "Rapunzel let down her long golden hair",
    "money market fund in Canada"
  ).toDS.toDF("text")

  //Fixture creation methods
  def trainNerDlModel(trainDatasetPath: String): NerDLModel = {

    val nerTagger = new NerDLApproach()
      .setInputCols("sentence", "token")
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMaxEpochs(20)
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


  "An NER with DL model" should "train de-identification entities" in  {
    nerDlModel = trainNerDlModel("src/test/resources/de-identification/train_dataset_small.csv")
    assert(nerDlModel.isInstanceOf[NerDLModel])
  }

  it should behave like saveModel(nerDlModel.write, "./tmp/ner_dl_model")

  it should "load NER DL Model" ignore {
    val loadedNerDlModel = NerDLModel.read.load("./tmp/ner_dl_model")
    assert(loadedNerDlModel.isInstanceOf[NerDLModel])
  }

  "A NER with CRF model" should "train de-identification entities" ignore {
    nerCrfModel = trainNerCRFModel("src/test/resources/de-identification/train_dataset_medium.csv")
    assert(nerCrfModel.isInstanceOf[NerCrfModel])
  }

  it should behave like saveModel(nerCrfModel.write, "./tmp/ner_crf_model")

  it should "load model" ignore {
    nerCrfModel = NerCrfModel.read.load("./tmp/ner_crf_model")
    assert(nerCrfModel.isInstanceOf[NerCrfModel])
  }

  private var deIdentificationDataFrame = PipelineModels.dummyDataset

  "A de-identification annotator (using NER trained with CRF)" should "return a spark dataframe" in {

    val posTagger = PerceptronModel.pretrained()
    val nerCRFTagger = NerCrfModel.pretrained()

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        nerCRFTagger,
        nerConverter,
        deIdentification
      )).fit(emptyDataset)

    deIdentificationDataFrame = pipeline.transform(testDataset)
    deIdentificationDataFrame.show(false)
    assert(deIdentificationDataFrame.isInstanceOf[DataFrame])

  }

  "A de-identification annotator (using NER trained with DL)" should "return a spark dataframe" in {

    val nerDlTagger = NerDLModel.pretrained()

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        nerDlTagger,
        nerConverter,
        deIdentification
      )).fit(emptyDataset)

    deIdentificationDataFrame = pipeline.transform(testDataset)
    //deIdentificationDataFrame.show(false)
    assert(deIdentificationDataFrame.isInstanceOf[DataFrame])

  }

  //Assert
  var testParams = TestParams(
    originalSentence = "Bob visited Switzerland a couple of years ago",
    tokenizedSentence = List(),
    annotations = Seq(
      Annotation(CHUNK, 0, 2, "Bob", Map("entity"->"PER")),
      Annotation(CHUNK, 12, 22, "Switzerland", Map("entity"->"LOC")),
      Annotation(DOCUMENT, 0, 44, "Bob visited Switzerland a couple of years ago", Map())
    )
  )

  var expectedParams = ExpectedParams(
    anonymizeSentence = "PER visited LOC a couple of years ago",
    unclassifiedEntities = List(),
    anonymizeAnnotation = Annotation(DOCUMENT, 0, 37, "PER visited LOC a couple of years ago",
      Map("sentence"->"protected")),
    protectedEntities = Seq(
      Annotation(CHUNK, 0, 2, "Bob", Map("entity"->"PER")),
      Annotation(CHUNK, 12, 22, "Switzerland", Map("entity"->"LOC"))
  ))

  it should behave like deIdentificationAnnotator(deIdentification, testParams, expectedParams)

  //Assert
  /*testParams = TestParams(
    originalSentence = "John was born on 05/10/1975 in Canada",
    tokenizedSentence = List("05/10/1975"),
    annotations = Seq(
      Annotation(CHUNK, 0, 2, "John", Map("entity"->"PER")),
      Annotation(CHUNK, 12, 22, "Canada", Map("entity"->"LOC")),
      Annotation(DOCUMENT, 0, 44, "John was born on 05/10/1975 in Canada", Map())
    )
  )

  expectedParams = ExpectedParams(
    anonymizeSentence = "PER was born on DATE in LOC",
    unclassifiedEntities = List("DATE"),
    anonymizeAnnotation = Annotation(DOCUMENT, 0, 37, "PER was born on DATE in LOC",
      Map("sentence"->"protected")),
    protectedEntities = Seq(
      Annotation(CHUNK, 0, 2, "John", Map("entity"->"PER")),
      Annotation(CHUNK, 12, 22, "Canada", Map("entity"->"LOC"))
    )
  )

  "when NER does not identify date on sentence" should behave like deIdentificationAnnotator(deIdentification,
    testParams, expectedParams)*/

}
