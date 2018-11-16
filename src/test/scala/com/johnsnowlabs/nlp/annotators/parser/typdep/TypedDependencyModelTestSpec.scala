package com.johnsnowlabs.nlp.annotators.parser.typdep

import com.johnsnowlabs.nlp.{DataBuilder, DocumentAssembler, Finisher, SparkAccessor}
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.{PerceptronApproach, PerceptronModel}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.util.PipelineModels
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.DataFrame
import org.scalatest.FlatSpec

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

  private val posTagger =  getPerceptronModel //PerceptronModel.pretrained()

  private val dependencyParser = DependencyParserModel.read.load("./tmp_dp_model")

  private val typedDependencyParser = new TypedDependencyParserApproach()
    .setInputCols(Array("token", "pos", "dependency"))
    .setOutputCol("labdep")
    .setConll2009FilePath("src/test/resources/parser/train/example.train")

  private val emptyDataSet = PipelineModels.dummyDataset

  def getPerceptronModel: PerceptronModel = {
    val perceptronTagger = new PerceptronApproach()
      .setNIterations(1)
      .setCorpus(ExternalResource("src/test/resources/anc-pos-corpus-small/",
        ReadAs.LINE_BY_LINE, Map("delimiter" -> "|")))
      .setInputCols(Array("token", "sentence"))
      .setOutputCol("pos")
      .fit(DataBuilder.basicDataBuild("dummy"))
    val path = "./tmp_perceptrontagger"

    perceptronTagger.write.overwrite.save(path)
    val perceptronTaggerRead = PerceptronModel.read.load(path)
    perceptronTaggerRead
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


  "A typed dependency parser model with a document input" should
    "predict a labeled relationship between words in each sentence" in {
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
      .setConll2009FilePath("src/test/resources/parser/train/example.train")
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

}
