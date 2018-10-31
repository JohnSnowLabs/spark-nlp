package com.johnsnowlabs.nlp.annotators.parser.typdep

import java.io.FileNotFoundException

import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.{DocumentAssembler, SparkAccessor}
import com.johnsnowlabs.util.PipelineModels
import org.apache.spark.ml.Pipeline
import org.scalatest.FlatSpec
import SparkAccessor.spark.implicits._

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

  private val posTagger = PerceptronModel.pretrained()

  private val dependencyParser = DependencyParserModel.read.load("./tmp/dp_model")

  private val typedDependencyParser = new TypedDependencyParserApproach()
    .setInputCols(Array("dependency"))
    .setOutputCol("labdep")
    .setConll2009FilePath("src/test/resources/parser/train/example.train")

  private val emptyDataset = PipelineModels.dummyDataset

  "A typed dependency parser approach that does not use Conll2009FilePath parameter" should
    "raise an error message" ignore {

    val typedDependencyParser = new TypedDependencyParserApproach()
      .setInputCols(Array("dependency"))
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
      pipeline.fit(emptyDataset)
    }
    assert(caught.getMessage == "requirement failed: " + expectedErrorMessage)

  }

  "A typed dependency parser approach with an empty CoNLL training file" should "raise an error message" ignore {

    val typedDependencyParser = new TypedDependencyParserApproach()
      .setInputCols(Array("dependency"))
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
      pipeline.fit(emptyDataset)
    }
    assert(caught.getMessage == "requirement failed: " + expectedErrorMessage)

  }

  "A typed dependency parser approach with an invalid file path or file name" should
    "raise FileNotFound exception" ignore {

    val typedDependencyParser = new TypedDependencyParserApproach()
      .setInputCols(Array("dependency"))
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
      pipeline.fit(emptyDataset)
    }

  }

  "A typed dependency parser approach with a nonempty dataset" should "not raise an error message" ignore {

    val helloDataset = Seq("Hello World!").toDS.toDF("text")
    val typedDependencyParserApproach = new TypedDependencyParserApproach()
    val model = typedDependencyParserApproach.fit(helloDataset)

    assert(typedDependencyParserApproach.isInstanceOf[TypedDependencyParserApproach])
    assert(model.isInstanceOf[TypedDependencyParserModel])

  }

  "A typed dependency parser with the right pipeline" should "output a labeled dependency parser" in {

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        dependencyParser,
        typedDependencyParser
      ))

    val model = pipeline.fit(emptyDataset)

    val helloDataset = Seq("Hello World!").toDS.toDF("text")
    val result = model.transform(helloDataset)
    result.show()

  }

}
