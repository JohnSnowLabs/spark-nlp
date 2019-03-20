package com.johnsnowlabs.nlp.annotators.parser.typdep

import java.io.FileNotFoundException

import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.parser.dep.{DependencyParserApproach, DependencyParserModel}
import com.johnsnowlabs.nlp.annotators.pos.perceptron.{PerceptronApproach, PerceptronModel}
import com.johnsnowlabs.nlp.training.POS
import com.johnsnowlabs.nlp.{DataBuilder, DocumentAssembler}
import com.johnsnowlabs.util.PipelineModels
import org.apache.spark.ml.Pipeline
import org.scalatest.FlatSpec
import com.johnsnowlabs.nlp.util.io.ResourceHelper

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

  private val posTagger = getPerceptronModel

  private val dependencyParser = getDependencyParserModel

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
    PerceptronModel.read.load(path)
  }

  def getDependencyParserModel: DependencyParserModel = {
    val dependencyParser = new DependencyParserApproach()
      .setInputCols(Array("sentence", "pos", "token"))
      .setOutputCol("dependency")
      .setDependencyTreeBank("src/test/resources/parser/unlabeled/dependency_treebank")
      .setNumberOfIterations(10)
      .fit(DataBuilder.basicDataBuild("dummy"))

    val path = "./tmp_dp_model"
    dependencyParser.write.overwrite.save(path)
    DependencyParserModel.read.load(path)
  }

  "A typed dependency parser that sets CoNLL-2009 and CoNLL-U format files " should "raise an error" in {

    val typedDependencyParserApproach = new TypedDependencyParserApproach()
      .setInputCols(Array("sentence", "pos", "token"))
      .setOutputCol("dependency")
      .setConll2009("src/test/resources/parser/labeled/conll-u/example.train.conll2009")
      .setConllU("src/test/resources/parser/labeled/conll-u/train_small.conllu.txt")
      .setNumberOfIterations(10)
    val expectedErrorMessage = "Use either CoNLL-2009 or CoNLL-U format file both are not allowed."

    val caught = intercept[IllegalArgumentException]{
      typedDependencyParserApproach.fit(emptyDataSet)
    }

    assert(caught.getMessage == expectedErrorMessage)
  }


  "A typed dependency parser that does not set TreeBank or CoNLL-U format files " should "raise an error" in {

    val pipeline = new TypedDependencyParserApproach()
    val expectedErrorMessage = "Either CoNLL-2009 or CoNLL-U format file is required."

    val caught = intercept[IllegalArgumentException]{
      pipeline.fit(emptyDataSet)
    }

    assert(caught.getMessage == expectedErrorMessage)
  }

  "A typed dependency parser approach with an invalid file path or file name" should
    "raise FileNotFound exception" in {

    val typedDependencyParser = new TypedDependencyParserApproach()
      .setInputCols(Array("token", "pos", "dependency"))
      .setOutputCol("labdep")
      .setConll2009("wrong/path")

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
