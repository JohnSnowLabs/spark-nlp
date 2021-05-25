package com.johnsnowlabs.nlp.annotators.parser.dl

import com.johnsnowlabs.nlp.{AnnotatorType, AssertAnnotations}
import com.johnsnowlabs.nlp.annotator.SentenceDetector
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.training.CoNLLU
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.scalatest.FlatSpec

class DependencyParserDLTestSpec extends FlatSpec {

  val conlluFile = "src/test/resources/conllu/en.test.conllu"

  "DependencyParserApproachDL" should "fit a dataset with CoNLL format to compute vocabulary" in {
    val conllDataSet = CoNLLU().readDataset(ResourceHelper.spark, conlluFile)

    val dependencyParser = new DependencyParserDLApproach()
      .setInputCols("document")
      .setOutputCol("dependency")

    val dependencyParserDLModel: DependencyParserDLModel = dependencyParser.fit(conllDataSet)

    assert(dependencyParserDLModel.uid.contains("DEPENDENCY_PARSER_DL"))
  }

  it should "fit a dataset without token column to compute vocabulary" in {
    import ResourceHelper.spark.implicits._

    val trainDataset = Seq("Text only with document and token columns").toDS.toDF()
      .withColumnRenamed("value", "text")
    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val sentence = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
    val tokenizer = new Tokenizer().setInputCols(Array("sentence")).setOutputCol("token")
    val dependencyParser = new DependencyParserDLApproach()
      .setInputCols("token")
      .setOutputCol("dependency")
      .setVocabularyColumn("token")
    val pipelineDependencyParserDL = new Pipeline()
      .setStages(Array(documentAssembler, sentence, tokenizer, dependencyParser))

    val dependencyParserDLPipelineModel: PipelineModel = pipelineDependencyParserDL.fit(trainDataset)

    assert(dependencyParserDLPipelineModel.uid.contains("pipeline"))
  }

  it should "raise an error when sampleFraction is out of range" in {
    val conllDataSet = CoNLLU().readDataset(ResourceHelper.spark, conlluFile)
    val expectedErrorMessage = "requirement failed: The sampleFraction must be between 0 and 1"

    val dependencyParser = new DependencyParserDLApproach()
      .setInputCols("lemma")
      .setOutputCol("dependency")
      .setSampleFraction(1.5)

    val errorMessage = intercept[IllegalArgumentException] {
      dependencyParser.fit(conllDataSet)
    }

    assert(errorMessage.getMessage == expectedErrorMessage)

  }

  "DependencyParserModelDL" should "transform a dataset" in {
    val conllDataSet = CoNLLU().readDataset(ResourceHelper.spark, conlluFile)
    val outputColumn = "dependencies"
    val dependencyParserDL = new DependencyParserDLApproach()
      .setInputCols("lemma")
      .setOutputCol(outputColumn)

    val dependencyParserDataSet = dependencyParserDL.fit(conllDataSet).transform(conllDataSet)

    AssertAnnotations.assertFieldsStruct(dependencyParserDataSet, outputColumn, AnnotatorType.LABELED_DEPENDENCY)
  }

  it should "work" in {
    import ResourceHelper.spark.implicits._

    val trainDataset = Seq("What if Google").toDS.toDF()
      .withColumnRenamed("value", "text")

    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val sentence = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
    val tokenizer = new Tokenizer().setInputCols(Array("sentence")).setOutputCol("token")
    val dependencyParser = new DependencyParserDLApproach()
      .setInputCols("token")
      .setOutputCol("dependency")
      .setVocabularyColumn("token")

    val pipelineDependencyParserDL = new Pipeline()
      .setStages(Array(documentAssembler, sentence, tokenizer, dependencyParser))

    val result = pipelineDependencyParserDL.fit(trainDataset).transform(trainDataset)
    result.show(false)
  }

}
