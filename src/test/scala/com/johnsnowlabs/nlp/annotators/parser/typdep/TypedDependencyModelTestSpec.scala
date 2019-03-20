package com.johnsnowlabs.nlp.annotators.parser.typdep

import java.nio.file.{Files, Paths}

import com.johnsnowlabs.nlp.{DataBuilder, DocumentAssembler, Finisher, SparkAccessor}
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.parser.dep.{DependencyParserApproach, DependencyParserModel}
import com.johnsnowlabs.nlp.annotators.pos.perceptron.{PerceptronApproach, PerceptronModel}
import com.johnsnowlabs.nlp.training.POS
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.util.PipelineModels
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.DataFrame
import org.scalatest.FlatSpec
import SparkAccessor.spark.implicits._
import org.apache.spark.ml.util.MLWriter

class TypedDependencyModelTestSpec extends FlatSpec {

  private val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  private val sentenceDetector = new SentenceDetector()
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

  private val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

  private val posTagger = getPerceptronModel

  private val dependencyParser = getDependencyParserModel

  private val typedDependencyParser = new TypedDependencyParserApproach()
    .setInputCols(Array("token", "pos", "dependency"))
    .setOutputCol("labdep")
    .setConll2009("src/test/resources/parser/labeled/example.train.conll2009")
    .setNumberOfIterations(10)

  private val typedDependencyParserConllU = new TypedDependencyParserApproach()
    .setInputCols(Array("token", "pos", "dependency"))
    .setOutputCol("labdep")
    .setConllU("src/test/resources/parser/labeled/train_small.conllu.txt")
    .setNumberOfIterations(10)

  private val emptyDataSet = PipelineModels.dummyDataset

  def getPerceptronModel: PerceptronModel = {
    val trainingPerceptronDF = POS().readDataset(ResourceHelper.spark, "src/test/resources/anc-pos-corpus-small/", "|", "tags")

    val perceptronTagger = new PerceptronApproach()
      .setInputCols(Array("token", "sentence"))
      .setOutputCol("pos")
      .setPosColumn("tags")
      .setNIterations(1)
      .fit(trainingPerceptronDF)

    val path = "./tmp_perceptrontagger"

    perceptronTagger.write.overwrite.save(path)
    val perceptronTaggerRead = PerceptronModel.read.load(path)
    perceptronTaggerRead
  }

  def getDependencyParserModel: DependencyParserModel = {
    val dependencyParser = new DependencyParserApproach()
      .setInputCols(Array("sentence", "pos", "token"))
      .setOutputCol("dependency")
      .setDependencyTreeBank("src/test/resources/parser/unlabeled/dependency_treebank")
      .setNumberOfIterations(50)
      .fit(DataBuilder.basicDataBuild("dummy"))

    val path = "./tmp_dp_model"
    dependencyParser.write.overwrite.save(path)
    DependencyParserModel.read.load(path)
  }

  def trainTypedDependencyParserModel(): TypedDependencyParserModel = {
    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser,
        typedDependencyParser
      ))

    val model = pipeline.fit(emptyDataSet)
    model.stages.last.asInstanceOf[TypedDependencyParserModel]
  }

  def saveModel(model: MLWriter, modelFilePath: String): Unit = {
    model.overwrite().save(modelFilePath)
    assertResult(true){
      Files.exists(Paths.get(modelFilePath))
    }
  }

  "A typed dependency parser model" should "save a trained model to local disk" in {
    val typedDependencyParser = new TypedDependencyParserApproach()
      .setInputCols(Array("token", "pos", "dependency"))
      .setOutputCol("labdep")
      .setConllU("src/test/resources/parser/labeled/train_small.conllu.txt")
      .setNumberOfIterations(10)

    val typedDependencyParserModel = typedDependencyParser.fit(emptyDataSet)
    saveModel(typedDependencyParserModel.write, "./tmp_tdp_model")
  }

  it should "load a pre-trained model from disk" in {
    val typedDependencyParserModel = TypedDependencyParserModel.read.load("./tmp_tdp_model")
    assert(typedDependencyParserModel.isInstanceOf[TypedDependencyParserModel])
  }

  "A typed dependency parser model with a sentence input" should
    "predict a labeled relationship between words in the sentence" in {
    import SparkAccessor.spark.implicits._

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser,
        typedDependencyParser
      ))

    val model = pipeline.fit(emptyDataSet)
    val typedDependencyParserModel = model.stages.last.asInstanceOf[TypedDependencyParserModel]
    val sentence = "I saw a girl with a telescope"
    val testDataSet = Seq(sentence).toDS.toDF("text")
    val typedDependencyParserDataFrame = model.transform(testDataSet)
    typedDependencyParserDataFrame.collect()
    //typedDependencyParserDataFrame.show(false)
    assert(typedDependencyParserModel.isInstanceOf[TypedDependencyParserModel])
    assert(typedDependencyParserDataFrame.isInstanceOf[DataFrame])

  }

  "A typed dependency parser (trained with CoNLL-U) model with a sentence input" should
    "predict a labeled relationship between words in the sentence" in {
    import SparkAccessor.spark.implicits._

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser,
        typedDependencyParserConllU
      ))
    val model = pipeline.fit(emptyDataSet)
    val typedDependencyParserModel = model.stages.last.asInstanceOf[TypedDependencyParserModel]
    val sentence = "I saw a girl with a telescope"
    val testDataSet = Seq(sentence).toDS.toDF("text")
    val typedDependencyParserDataFrame = model.transform(testDataSet)
    typedDependencyParserDataFrame.collect()
    //typedDependencyParserDataFrame.show(false)
    assert(typedDependencyParserModel.isInstanceOf[TypedDependencyParserModel])
    assert(typedDependencyParserDataFrame.isInstanceOf[DataFrame])

  }

  "A typed dependency parser model with a document input" should
    "predict a labeled relationship between words in each sentence" in {
    import SparkAccessor.spark.implicits._
    val typedDependencyParser = TypedDependencyParserModel.read.load("./tmp_tdp_model")
    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser,
        typedDependencyParser
      ))

    val model = pipeline.fit(emptyDataSet)

    val document = "The most troublesome report may be the August merchandise trade deficit due out tomorrow. " +
      "Meanwhile, September housing starts, due Wednesday, are thought to have inched upward."
    val testDataSet = Seq(document).toDS.toDF("text")
    val typedDependencyParserDataFrame = model.transform(testDataSet)
    typedDependencyParserDataFrame.collect()
    //typedDependencyParserDataFrame.show(false)
    assert(typedDependencyParserDataFrame.isInstanceOf[DataFrame])

  }

  "A typed dependency parser model with finisher in its pipeline" should
    "predict a labeled relationship between words in each sentence" in  {
    import SparkAccessor.spark.implicits._

    val finisher = new Finisher().setInputCols("labdep")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser,
        typedDependencyParser,
        finisher
      ))

    val model = pipeline.fit(emptyDataSet)

    val document = "The most troublesome report may be the August merchandise trade deficit due out tomorrow. " +
      "Meanwhile, September housing starts, due Wednesday, are thought to have inched upward."
    val testDataSet = Seq(document).toDS.toDF("text")
    val typedDependencyParserDataFrame = model.transform(testDataSet)
    typedDependencyParserDataFrame.collect()
    //typedDependencyParserDataFrame.show(false)
    assert(typedDependencyParserDataFrame.isInstanceOf[DataFrame])

  }

  "A typed dependency parser model with an input of more than one row" should
    "predict a labeled relationship between words in each sentence" in  {
    import SparkAccessor.spark.implicits._

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser,
        typedDependencyParser
      ))

    val model = pipeline.fit(emptyDataSet)

    val document = Seq(
      "The most troublesome report may be the August merchandise trade deficit due out tomorrow",
      "Meanwhile, September housing starts, due Wednesday, are thought to have inched upward",
      "I solved the problem with statistics")
    val testDataSet = document.toDS.toDF("text")
    val typedDependencyParserDataFrame = model.transform(testDataSet)
    typedDependencyParserDataFrame.collect()
    //typedDependencyParserDataFrame.show(false)
    assert(typedDependencyParserDataFrame.isInstanceOf[DataFrame])

  }

  "A typed dependency parser model whit few numberOfTrainingIterations" should
    "predict a labeled relationship between words in the sentence" in  {
    import SparkAccessor.spark.implicits._

    val typedDependencyParser = new TypedDependencyParserApproach()
      .setInputCols(Array("token", "pos", "dependency"))
      .setOutputCol("labdep")
      .setConll2009("src/test/resources/parser/labeled/example.train.conll2009")
      .setNumberOfIterations(5)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser,
        typedDependencyParser
      ))

    val model = pipeline.fit(emptyDataSet)

    val sentence =
      "The most troublesome report may be the August merchandise trade deficit due out tomorrow"
    val testDataSet = Seq(sentence).toDS.toDF("text")
    val typedDependencyParserDataFrame = model.transform(testDataSet)
    typedDependencyParserDataFrame.collect()
    //typedDependencyParserDataFrame.show(false)
    assert(typedDependencyParserDataFrame.isInstanceOf[DataFrame])

  }

  "A typed dependency parser (trained with CoNLLU) model whit few numberOfTrainingIterations" should
    "predict a labeled relationship between words in the sentence" in  {
    import SparkAccessor.spark.implicits._

    val typedDependencyParser = new TypedDependencyParserApproach()
      .setInputCols(Array("token", "pos", "dependency"))
      .setOutputCol("labdep")
      .setConllU("src/test/resources/parser/labeled/train_small.conllu.txt")
      .setNumberOfIterations(5)

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser,
        typedDependencyParser
      ))

    val model = pipeline.fit(emptyDataSet)

    val sentence =
      "The most troublesome report may be the August merchandise trade deficit due out tomorrow"
    val testDataSet = Seq(sentence).toDS.toDF("text")
    val typedDependencyParserDataFrame = model.transform(testDataSet)
    typedDependencyParserDataFrame.collect()
    //typedDependencyParserDataFrame.show(false)
    assert(typedDependencyParserDataFrame.isInstanceOf[DataFrame])

  }

  "A pre-trained typed dependency parser" should "find relationships between words" in {

    val document = "The most troublesome report may be the August merchandise trade deficit due out tomorrow. " +
      "Meanwhile, September housing starts, due Wednesday, are thought to have inched upward."
    val serializedModelPath = "./tmp_tdp_model"
    val typedDependencyParser = TypedDependencyParserModel.read.load(serializedModelPath)
      .setInputCols(Array("token", "pos", "dependency"))
      .setOutputCol("labdep")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser,
        typedDependencyParser
      ))
    val typedDependencyParserModel = pipeline.fit(emptyDataSet)
    val testDataSet = Seq(document).toDS.toDF("text")

    val typedDependencyParserDataFrame = typedDependencyParserModel.transform(testDataSet)
    typedDependencyParserDataFrame.collect()
    //typedDependencyParserDataFrame.show(false)

    assert(typedDependencyParserDataFrame.isInstanceOf[DataFrame])

  }

}
