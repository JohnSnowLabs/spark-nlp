package com.johnsnowlabs.nlp.annotators.parser.typdep

import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.parser.dep.{DependencyParserApproach, DependencyParserModel}
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

  private val typedDependencyParser = new TypedDependencyParserApproach()
    .setInputCols(Array("sentence", "pos", "token"))
    .setOutputCol("dependency")

  private val emptyDataset = PipelineModels.dummyDataset

  "A typed dependency parser approach with an empty CoNLL dataset" should "raise an error message" in {

    val expectedErrorMessage = "Training file with CoNLL 2009 format is required"
    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        posTagger,
        typedDependencyParser
      ))

    val caught = intercept[IllegalArgumentException]{
      pipeline.fit(emptyDataset)
    }
    assert(caught.getMessage == "requirement failed: " + expectedErrorMessage)

  }

  "A typed dependency parser approach with a nonempty dataset" should "not raise an error message" in {

    val helloDataset = Seq("Hello World!").toDS.toDF("text")
    val typedDependencyParserApproach = new TypedDependencyParserApproach()
    val model = typedDependencyParserApproach.fit(helloDataset)

    assert(typedDependencyParserApproach.isInstanceOf[TypedDependencyParserApproach])
    assert(model.isInstanceOf[TypedDependencyParserModel])

  }

  "A typed dependency parser " should " use a pre-trained dependency parser model" in {
    val dependencyParserModel = typedDependencyParser.loadPretrainedModel()

    assert(dependencyParserModel.isInstanceOf[DependencyParserModel])
  }

  "A typed dependency parser with the right pipeline" should "outputs a labeled dependency parser" ignore {

  }

}
