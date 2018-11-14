package com.johnsnowlabs.nlp.annotators.parser.typdep

import java.io.FileNotFoundException

import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.{PerceptronApproach, PerceptronModel}
import com.johnsnowlabs.nlp.{DataBuilder, DocumentAssembler, SparkAccessor}
import com.johnsnowlabs.util.PipelineModels
import org.apache.spark.ml.Pipeline
import org.scalatest.FlatSpec
import SparkAccessor.spark.implicits._
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}

class TypedDependencyParserApproachTestSpec extends FlatSpec{


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

  private val posTagger = getPerceptronModel //PerceptronModel.pretrained()

  private val dependencyParser = DependencyParserModel.read.load("./tmp/dp_model")

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
    val path = "./test-output-tmp/perceptrontagger"

    perceptronTagger.write.overwrite.save(path)
    val perceptronTaggerRead = PerceptronModel.read.load(path)
    perceptronTaggerRead
  }

  "A typed dependency parser approach that does not use Conll2009FilePath parameter" should
    "raise an error message" in {

    val typedDependencyParser = new TypedDependencyParserApproach()
      .setInputCols(Array("token", "pos", "dependency"))
      .setOutputCol("labdep")

    val expectedErrorMessage = "Training file with CoNLL 2009 format is required"

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser,
        typedDependencyParser
      ))

    val caught = intercept[IllegalArgumentException]{
      pipeline.fit(emptyDataSet)
    }
    assert(caught.getMessage == "requirement failed: " + expectedErrorMessage)

  }

  "A typed dependency parser approach with an empty CoNLL training file" should "raise an error message" in {

    val typedDependencyParser = new TypedDependencyParserApproach()
      .setInputCols(Array("token", "pos", "dependency"))
      .setOutputCol("labdep")
      .setConll2009FilePath("")

    val expectedErrorMessage = "Training file with CoNLL 2009 format is required"

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser,
        typedDependencyParser
      ))

    val caught = intercept[IllegalArgumentException]{
      pipeline.fit(emptyDataSet)
    }
    assert(caught.getMessage == "requirement failed: " + expectedErrorMessage)

  }

  "A typed dependency parser approach with an invalid file path or file name" should
    "raise FileNotFound exception" in {

    val typedDependencyParser = new TypedDependencyParserApproach()
      .setInputCols(Array("token", "pos", "dependency"))
      .setOutputCol("labdep")
      .setConll2009FilePath("wrong/path")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser,
        typedDependencyParser
      ))

    assertThrows[FileNotFoundException]{
      pipeline.fit(emptyDataSet)
    }

  }
}
