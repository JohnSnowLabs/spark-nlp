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
    .setUseAbbreviations(true)

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

  //Fixture creation methods
  def trainNerDlModel(trainDatasetPath: String): NerDLModel = {

    val nerTagger = new NerDLApproach()
      .setInputCols("sentence", "token")
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMaxEpochs(10)
      .setEmbeddingsSource("src/test/resources/ner-corpus/embeddings.100d.test.txt",
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
      .setInputCols("sentence", "token", "pos")
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMinEpochs(5)
      .setMaxEpochs(10)
      .setLossEps(1e-3)
      .setEmbeddingsSource("src/test/resources/ner-corpus/embeddings.100d.test.txt",
        100, WordEmbeddingsFormat.TEXT)
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
    nerDlModel = trainNerDlModel("src/test/resources/de-identification/train_dataset_main_small.csv")
    assert(nerDlModel.isInstanceOf[NerDLModel])
  }

  it should "be serializable" in  {
    saveModel(nerDlModel.write, "./tmp/ner_dl_model")
  }

  it should "be loaded from disk" in {
    nerDlModel = NerDLModel.read.load("./tmp/ner_dl_model")
    assert(nerDlModel.isInstanceOf[NerDLModel])
  }

  "A NER with CRF model" should "train de-identification entities" ignore {
    nerCrfModel = trainNerCRFModel("src/test/resources/de-identification/train_dataset_main_small.csv")
    assert(nerCrfModel.isInstanceOf[NerCrfModel])
  }

  it should "be serializable" ignore {
    saveModel(nerCrfModel.write, "./tmp/ner_crf_model")
  }

  it should "be loaded from disk" ignore {
    nerCrfModel = NerCrfModel.read.load("./tmp/ner_crf_model")
    assert(nerCrfModel.isInstanceOf[NerCrfModel])
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
      .setRegexPatternsDictionary("src/test/resources/de-identification/dic_regex_patterns_main_categories.txt")

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
    val regexPatterns = Array("\\d{4}-\\d{2}-\\d{2}", "\\d{4}", "\\d{1,2}\\/\\d{1,2}\\/\\d{2,4}")

    //Act
    val isMatch = deIdentificationModel.isRegexMatch(token, regexPatterns)

    //Assert
    assert(isMatch)
  }

  "A token without a match within a list of regex patterns " should "return false" in {
    //Arrange
    val deIdentificationModel = new DeIdentificationModel()
    val token = "Bob"
    val regexPatterns = Array("\\d{4}-\\d{2}-\\d{2}", "\\d{4}", "\\d{1,2}\\/\\d{1,2}\\/\\d{2,4}")

    //Act
    val isMatch = deIdentificationModel.isRegexMatch(token, regexPatterns)

    //Assert
    assert(!isMatch)
  }

 "A de-identification approach" should "transform a regex dictionary" in {
   //Arrange
   val deIdentificationApproach = new DeIdentification()
     .setInputCols(Array("ner_con", "token ", "document"))
     .setOutputCol("dei")
     .setRegexPatternsDictionary("src/test/resources/de-identification/dic_regex_patterns_sub_categories.txt")

   val regexPatternsDictionary = Array(
     ("DATE", "\\d{4}-\\d{2}-\\d{2}"),
     ("DATE", "\\d{4}"),
     ("DATE", "\\d{1,2}\\/\\d{1,2}\\/\\d{2,4}"),
     ("USERNAME", "[a-zA-Z]{2,3}\\d{1,3}"),
     ("AGE", "\\d{1,2}[a-zA-Z.\\/]+"),
     ("AGE", "\\d{1,2}(?:-|\\s|\\S)(?:year|yr)(?:-|\\s)old"))

   val expectedDictionary = Map(
     "USERNAME"->Array("[a-zA-Z]{2,3}\\d{1,3}"),
     "DATE"->Array("\\d{4}-\\d{2}-\\d{2}", "\\d{4}", "\\d{1,2}\\/\\d{1,2}\\/\\d{2,4}"),
     "AGE"->Array("\\d{1,2}[a-zA-Z.\\/]+", "\\d{1,2}(?:-|\\s|\\S)(?:year|yr)(?:-|\\s)old")
   )

   //Act
   val dictionary = deIdentificationApproach.transformRegexPatternsDictionary(regexPatternsDictionary)

   //Assert
   assert(dictionary.size==expectedDictionary.size)

 }

  "A de-identification model with ner and regex entities that differ in labeled entities" should
    "choose regex entities over ner entities" in {
    //Arrange
    val nerEntities = Seq(
      Annotation(CHUNK, 0, 2, "John", Map("entity"->"PER")),
      Annotation(CHUNK, 12, 22, "Canada", Map("entity"->"LOC")),
      Annotation(CHUNK, 26, 31, "05/10/1975", Map("entity"->"ID"))
    )

    val regexEntities = Seq(
      Annotation(CHUNK, 26, 31, "05/10/1975", Map("entity"->"DATE"))
    )

    val deIdentificationModel = new DeIdentificationModel()

    val expectedNerEntities = Seq(
      Annotation(CHUNK, 0, 2, "John", Map("entity"->"PER")),
      Annotation(CHUNK, 12, 22, "Canada", Map("entity"->"LOC"))
    )

    //Act
    val cleanNerEntities = deIdentificationModel.handleEntitiesDifferences(nerEntities, regexEntities)

    //Assert
    assert(cleanNerEntities.map(entity=>entity.result).toList==expectedNerEntities.map(entity=>entity.result).toList)
  }

  "A de-identification annotator when ner and regex entities overlaps with one regex entity" should
    "merge entities" in {
    //Arrange
    val nerEntities = Seq(
      Annotation(CHUNK, 0, 2, "John", Map("entity"->"PER")),
      Annotation(CHUNK, 12, 22, "Canada", Map("entity"->"LOC")),
      Annotation(CHUNK, 26, 31, "05/10/1975", Map("entity"->"DATE"))
      )

    val regexEntities = Seq(Annotation(CHUNK, 26, 31, "05/10/1975", Map("entity"->"DATE")))

    val deIdentificationModel = new DeIdentificationModel()

    val expectedProtectedEntities = Seq(
      Annotation(CHUNK, 0, 2, "John", Map("entity"->"PER")),
      Annotation(CHUNK, 12, 22, "Canada", Map("entity"->"LOC")),
      Annotation(CHUNK, 26, 31, "05/10/1975", Map("entity"->"DATE"))
    )

    //Act
    val protectedEntities = deIdentificationModel.mergeEntities(nerEntities, regexEntities)

    //Assert
    assert(protectedEntities.map(entity=>entity.result).toList==expectedProtectedEntities.map(entity=>entity.result).toList)
  }


  "A de-identification annotator when ner and regex entities overlaps with multiple regex entities" should
    "merge entities" in {
    //Arrange
    val nerEntities = Seq(
      Annotation(CHUNK, 0, 2, "John", Map("entity"->"PER")),
      Annotation(CHUNK, 12, 22, "Canada", Map("entity"->"LOC")),
      Annotation(CHUNK, 26, 31, "05/10/1975", Map("entity"->"DATE"))
    )

    val regexEntities = Seq(
      Annotation(CHUNK, 26, 31, "510-1975", Map("entity"->"ID")),
      Annotation(CHUNK, 26, 31, "05/10/1975", Map("entity"->"DATE"))
    )

    val deIdentificationModel = new DeIdentificationModel()

    val expectedProtectedEntities = Seq(
      Annotation(CHUNK, 0, 2, "John", Map("entity"->"PER")),
      Annotation(CHUNK, 12, 22, "Canada", Map("entity"->"LOC")),
      Annotation(CHUNK, 26, 31, "05/10/1975", Map("entity"->"DATE")),
      Annotation(CHUNK, 26, 31, "510-1975", Map("entity"->"ID"))
    )

    //Act
    val protectedEntities = deIdentificationModel.mergeEntities(nerEntities, regexEntities)

    //Assert
    assert(protectedEntities.map(entity=>entity.result).toList==expectedProtectedEntities.map(entity=>entity.result).toList)
  }

  "A de-identification annotator with empty regex entities" should
    "merge entities" in {
    //Arrange
    val nerEntities = Seq(
      Annotation(CHUNK, 0, 2, "John", Map("entity"->"PER")),
      Annotation(CHUNK, 12, 22, "Canada", Map("entity"->"LOC")),
      Annotation(CHUNK, 26, 31, "05/10/1975", Map("entity"->"DATE"))
    )

    val regexEntities = Seq()

    val deIdentificationModel = new DeIdentificationModel()

    val expectedProtectedEntities = Seq(
      Annotation(CHUNK, 0, 2, "John", Map("entity"->"PER")),
      Annotation(CHUNK, 12, 22, "Canada", Map("entity"->"LOC")),
      Annotation(CHUNK, 26, 31, "05/10/1975", Map("entity"->"DATE"))
    )

    //Act
    val protectedEntities = deIdentificationModel.mergeEntities(nerEntities, regexEntities)

    //Assert
    assert(protectedEntities.map(entity=>entity.result).toList==expectedProtectedEntities.map(entity=>entity.result).toList)
  }

  "A string that contains RegEx flavors (PCRE)" should  "replace those characters with escape" in {
    //Assert
    val word = "200[2"
    val expectedWord = "200\\[2"

    //Act
    val newWord = deIdentificationModel.replaceRegExFlavors(word)

    //Assert
    assert(newWord == expectedWord)

  }

  "A string that contains several RegEx flavors (PCRE)" should  "replace those characters with escape" in {
    //Assert
    val word = "(several.favors)"
    val expectedWord = "\\(several\\.favors\\)"

    //Act
    val newWord = deIdentificationModel.replaceRegExFlavors(word)

    //Assert
    assert(newWord == expectedWord)

  }

  //Assert
  var testParams = TestParams(
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

  var expectedParams = ExpectedParams(
    anonymizeSentence = "PER was born on DATE in LOC",
    regexEntities = Seq(
      Annotation(CHUNK, 26, 31, "05/10/1975", Map("entity"->"DATE"))
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

  private val deIdentificationModel = new DeIdentification()
      .setInputCols(Array("ner_con", "token", "document"))
      .setOutputCol("dei")
      .setRegexPatternsDictionary("src/test/resources/de-identification/dic_regex_patterns_main_categories.txt").fit(emptyDataset)

  //Act
  "A de-identification model when NER does not identify an entity on sentence" should behave like
    deIdentificationAnnotator(deIdentificationModel, testParams, expectedParams)

  //Arrange
  testParams = TestParams(
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

  expectedParams = ExpectedParams(
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

  "A de-identification model when NER identify all entities on sentence" should behave like
    deIdentificationAnnotator(deIdentificationModel, testParams, expectedParams)

  private var deIdentificationDataFrame = PipelineModels.dummyDataset

  "A de-identification annotator (using NER trained with CRF)" should "return a spark dataframe" ignore {

    val pipeline = getDeIdentificationCRFPipeline.fit(emptyDataset)

    deIdentificationDataFrame = pipeline.transform(testDataset)
    //deIdentificationDataFrame.show(false)
    assert(deIdentificationDataFrame.isInstanceOf[DataFrame])

  }

  "A de-identification annotator (using NER trained with DL)" should "return a spark dataframe" ignore {

    val pipeline = getDeIdentificationDLPipeline.fit(emptyDataset)

    deIdentificationDataFrame = pipeline.transform(testDataset)
    //deIdentificationDataFrame.show(false)
    assert(deIdentificationDataFrame.isInstanceOf[DataFrame])

  }

  "A de-identification annotator setting regex pattern dictionary" should "return a spark dataframe" ignore {

    val pipeline = getDeIdentificationDLPipelineWithDictionary.fit(emptyDataset)

    deIdentificationDataFrame = pipeline.transform(testDataset)
    //deIdentificationDataFrame.show(false)
    assert(deIdentificationDataFrame.isInstanceOf[DataFrame])

  }

  "A de-identification annotator with an NER DL trained with i2b2 dataset" should "transform data" in {
    val deIdentification = new DeIdentification()
      .setInputCols(Array("ner_con", "token", "document"))
      .setOutputCol("dei")
      .setRegexPatternsDictionary("src/test/resources/de-identification/dic_regex_patterns_main_categories.txt")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        nerDlModel,
        nerConverter,
        deIdentification
      )).fit(emptyDataset)


    val testDataset = Seq(
      "Record date: 2080-03-13",
      "Ms. Louise Iles is a 70yearold"
    ).toDS.toDF("text")

    deIdentificationDataFrame = pipeline.transform(testDataset)
    deIdentificationDataFrame.collect()
    //deIdentificationDataFrame.show(false)
    assert(deIdentificationDataFrame.isInstanceOf[DataFrame])

  }


  "A de-identification annotator with an NER CRF trained with i2b2 dataset" should "transform data" ignore {
    val posTagger = PerceptronModel.pretrained()
    val deIdentification = new DeIdentification()
      .setInputCols(Array("ner_con", "token", "document"))
      .setOutputCol("dei")
      .setRegexPatternsDictionary("src/test/resources/de-identification/dic_regex_patterns_main_categories.txt")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        nerCrfModel,
        nerConverter,
        deIdentification
      )).fit(emptyDataset)


    val testDataset = Seq(
      "Record date: 2080-03-13",
      "Ms. Louise Iles is a 70yearold",
      "200[2"
    ).toDS.toDF("text")

    deIdentificationDataFrame = pipeline.transform(testDataset)
    deIdentificationDataFrame.collect()
    //deIdentificationDataFrame.show(false)
    assert(deIdentificationDataFrame.isInstanceOf[DataFrame])

  }

}




