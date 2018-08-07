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
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.common.IndexedToken
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}


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

  def getDeIdentificationCRFPipeline: Pipeline = {
    val posTagger = PerceptronModel.pretrained()
    val nerCRFTagger = NerCrfModel.pretrained()
    val deIdentification = new DeIdentification()
      .setInputCols(Array("ner_con", "token", "document"))
      .setOutputCol("dei")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        nerCRFTagger,
        nerConverter,
        deIdentification
      ))
    pipeline
  }

  def getDeIdentificationDLPipeline: Pipeline = {
    val nerDlTagger = NerDLModel.pretrained()
    val deIdentification = new DeIdentification()
      .setInputCols(Array("ner_con", "token", "document"))
      .setOutputCol("dei")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        nerDlTagger,
        nerConverter,
        deIdentification
      ))
    pipeline
  }

  def getDeIdentificationDLPipelineWithDictionary: Pipeline = {
    val nerDlTagger = NerDLModel.pretrained()
    val deIdentification = new DeIdentification()
      .setInputCols(Array("ner_con", "token", "document"))
      .setOutputCol("dei")
      .setRegexPatternsDictionary(ExternalResource("src/test/resources/de-identification/DicRegexPatterns.txt",
        ReadAs.LINE_BY_LINE, Map()))

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        nerDlTagger,
        nerConverter,
        deIdentification
      ))
    pipeline
  }

  "A token with a match within a list of regex patterns " should "return true" in {
    //Arrange
    val deIdentificationModel = new DeIdentificationModel()
    val token = "05/10/1975"
    val regexPatterns = List("\\d{4}-\\d{2}-\\d{2}", "\\d{4}", "\\d{1,2}\\/\\d{1,2}\\/\\d{2,4}")

    //Act
    val isMatch = deIdentificationModel.isRegexMatch(token, regexPatterns)

    //Assert
    assert(isMatch)
  }

  "A token without a match within a list of regex patterns " should "return false" in {
    //Arrange
    val deIdentificationModel = new DeIdentificationModel()
    val token = "Bob"
    val regexPatterns = List("\\d{4}-\\d{2}-\\d{2}", "\\d{4}", "\\d{1,2}\\/\\d{1,2}\\/\\d{2,4}")

    //Act
    val isMatch = deIdentificationModel.isRegexMatch(token, regexPatterns)

    //Assert
    assert(!isMatch)
  }

  "An NER with DL model" should "train de-identification entities" ignore  {
    nerDlModel = trainNerDlModel("src/test/resources/de-identification/train_dataset_small.csv")
    assert(nerDlModel.isInstanceOf[NerDLModel])
  }

  "Our model" should "serialize for God's sake" in {
    saveModel(nerDlModel.write, "./tmp/ner_dl_model")
  }

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

    val pipeline = getDeIdentificationCRFPipeline.fit(emptyDataset)

    deIdentificationDataFrame = pipeline.transform(testDataset)
    //deIdentificationDataFrame.show(false)
    assert(deIdentificationDataFrame.isInstanceOf[DataFrame])

  }

  "A de-identification annotator (using NER trained with DL)" should "return a spark dataframe" in {

    val pipeline = getDeIdentificationDLPipeline.fit(emptyDataset)

    deIdentificationDataFrame = pipeline.transform(testDataset)
    //deIdentificationDataFrame.show(false)
    assert(deIdentificationDataFrame.isInstanceOf[DataFrame])

  }

  //Arrange
  var testParams = TestParams(
    originalSentence = "Bob visited Switzerland a couple of years ago",
    tokenizeSentence = Seq(
      IndexedToken("Bob", 0, 2),
      IndexedToken("visited", 4, 10),
      IndexedToken("Switzerland", 12, 22),
      IndexedToken("a", 24, 24),
      IndexedToken("couple", 26, 31),
      IndexedToken("of", 33, 34),
      IndexedToken("years", 36, 40),
      IndexedToken("ago", 42, 44)
    ),
    annotations = Seq(
      Annotation(CHUNK, 0, 2, "Bob", Map("entity"->"PER")),
      Annotation(CHUNK, 12, 22, "Switzerland", Map("entity"->"LOC")),
      Annotation(TOKEN, 0, 2, "Bob", Map("sentence"->"1")),
      Annotation(TOKEN, 4, 10, "visited", Map("sentence"->"1")),
      Annotation(TOKEN, 12, 22, "Switzerland", Map("sentence"->"1")),
      Annotation(TOKEN, 24, 24, "a", Map("sentence"->"1")),
      Annotation(TOKEN, 26, 31, "couple", Map("sentence"->"1")),
      Annotation(TOKEN, 33, 34, "of", Map("sentence"->"1")),
      Annotation(TOKEN, 36, 40, "years", Map("sentence"->"1")),
      Annotation(TOKEN, 42, 44, "ago", Map("sentence"->"1")),
      Annotation(DOCUMENT, 0, 44, "Bob visited Switzerland a couple of years ago", Map())
    )
  )

  var expectedParams = ExpectedParams(
    anonymizeSentence = "PER visited LOC a couple of years ago",
    regexEntities = Seq(),
    anonymizeAnnotation = Annotation(DOCUMENT, 0, 37, "PER visited LOC a couple of years ago",
      Map("sentence"->"protected")),
    nerEntities = Seq(
      Annotation(CHUNK, 0, 2, "Bob", Map("entity"->"PER")),
      Annotation(CHUNK, 12, 22, "Switzerland", Map("entity"->"LOC"))),
    mergedEntities = Seq(
      Annotation(CHUNK, 0, 2, "Bob", Map("entity"->"PER")),
      Annotation(CHUNK, 12, 22, "Switzerland", Map("entity"->"LOC")))
  )

  private val deIdentificationModel = new DeIdentification()
    .setInputCols(Array("ner_con", "token", "document"))
    .setOutputCol("dei")
    .setRegexPatternsDictionary(ExternalResource("src/test/resources/de-identification/DicRegexPatterns.txt",
      ReadAs.LINE_BY_LINE, Map("delimiter"->" "))).fit(emptyDataset)


  "A de-identification annotator with regex pattern dictionary" should "return a spark dataframe" in {

    val nerDlTagger = NerDLModel.pretrained()

    val deIdentification = new DeIdentification()
      .setInputCols(Array("ner_con", "token", "document"))
      .setOutputCol("dei")
      .setRegexPatternsDictionary(ExternalResource("src/test/resources/de-identification/DicRegexPatterns.txt",
        ReadAs.LINE_BY_LINE, Map("delimiter"->" ")))

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


 "A de-identification approach" should "transform regex dictionary" in {
   //Arrange
   val deIdentificationApproach = new DeIdentification()
     .setInputCols(Array("ner_con", "token ", "document"))
     .setOutputCol("dei")
     .setRegexPatternsDictionary(ExternalResource("src/test/resources/de-identification/DicRegexPatterns.txt",
       ReadAs.LINE_BY_LINE, Map()))

   val regexPatternsDictionary = List(
     ("DATE", "\\d{4}-\\d{2}-\\d{2}"),
     ("DATE", "\\d{4}"),
     ("DATE", "\\d{1,2}\\/\\d{1,2}\\/\\d{2,4}"),
     ("USERNAME", "[a-zA-Z]{2,3}\\d{1,3}"),
     ("AGE", "\\d{1,2}[a-zA-Z.\\/]+"),
     ("AGE", "\\d{1,2}(?:-|\\s|\\S)(?:year|yr)(?:-|\\s)old"))

   val expectedDictionary = Map(
     "DATE"->List("\\d{4}-\\d{2}-\\d{2}", "\\d{4}", "\\d{1,2}\\/\\d{1,2}\\/\\d{2,4}"),
     "USERNAME"->List("[a-zA-Z]{2,3}\\d{1,3}"),
     "AGE"->List("\\d{1,2}[a-zA-Z.\\/]+", "\\d{1,2}(?:-|\\s|\\S)(?:year|yr)(?:-|\\s)old")
   )

   //Act
   val dictionary = deIdentificationApproach.transformRegexPatternsDictionary(regexPatternsDictionary)

   //Assert
   assert(dictionary == expectedDictionary)

 }

  //Assert
  testParams = TestParams(
    originalSentence = "John was born on 05/10/1975 in Canada",
    tokenizeSentence = Seq(
      IndexedToken("John", 0, 2),
      IndexedToken("was", 4, 10),
      IndexedToken("born", 12, 22),
      IndexedToken("on", 24, 24),
      IndexedToken("05/10/1975", 26, 31),
      IndexedToken("in", 33, 34),
      IndexedToken("Canada", 36, 40)
    ),
    annotations = Seq(
      Annotation(CHUNK, 0, 2, "John", Map("entity"->"PER")),
      Annotation(CHUNK, 12, 22, "Canada", Map("entity"->"LOC")),
      Annotation(TOKEN, 0, 2, "John", Map("sentence"->"1")),
      Annotation(TOKEN, 4, 10, "was", Map("sentence"->"1")),
      Annotation(TOKEN, 12, 22, "born", Map("sentence"->"1")),
      Annotation(TOKEN, 24, 24, "on", Map("sentence"->"1")),
      Annotation(TOKEN, 26, 31, "05/10/1975", Map("sentence"->"1")),
      Annotation(TOKEN, 33, 34, "in", Map("sentence"->"1")),
      Annotation(TOKEN, 36, 40, "Canada", Map("sentence"->"1")),
      Annotation(DOCUMENT, 0, 44, "John was born on 05/10/1975 in Canada", Map())
    )
  )

  expectedParams = ExpectedParams(
    anonymizeSentence = "PER was born on DATE in LOC",
    regexEntities = Seq(
      Annotation(CHUNK, 26, 31, "05/10/1975", Map("regex_entity"->"DATE"))
    ),
    anonymizeAnnotation = Annotation(DOCUMENT, 0, 37, "PER was born on DATE in LOC",
      Map("sentence"->"protected")),
    nerEntities = Seq(
      Annotation(CHUNK, 0, 2, "John", Map("entity"->"PER")),
      Annotation(CHUNK, 12, 22, "Canada", Map("entity"->"LOC"))),
    mergedEntities = Seq(
      Annotation(CHUNK, 0, 2, "John", Map("entity"->"PER")),
      Annotation(CHUNK, 12, 22, "Canada", Map("entity"->"LOC")),
      Annotation(CHUNK, 26, 31, "05/10/1975", Map("entity"->"DATE")))
  )

  //Act
  "when NER does not identify date on sentence" should behave like deIdentificationAnnotator(deIdentificationModel,
    testParams, expectedParams)

}
