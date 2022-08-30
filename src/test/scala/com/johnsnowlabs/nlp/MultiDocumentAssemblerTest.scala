package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

class MultiDocumentAssemblerTest extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._

  val input1 = "This is the first input"
  val input2 = "This is the second input"

  private val twoInputDataset = Seq((input1, input2)).toDS.toDF("input1", "input2")
  private val dataset = Seq(input1).toDS.toDF("text")

  "MultiDocumentAssembler with two input cols" should "transform and output two cols" taggedAs FastTest in {

    val multiDocumentAssembler = new MultiDocumentAssembler()
      .setInputCols("input1", "input2")
      .setOutputCols("output1", "output2")
    tokenizer.setInputCols("output2").setOutputCol("token")

    val multiDocumentPipeline = pipeline.setStages(Array(multiDocumentAssembler, tokenizer))

    val result = multiDocumentPipeline.fit(twoInputDataset).transform(twoInputDataset)

    val actualOutput1 = AssertAnnotations.getActualResult(result, "output1")
    val actualOutput2 = AssertAnnotations.getActualResult(result, "output2")
    val actualResultToken = AssertAnnotations.getActualResult(result, "token")

    val expectedOutput1 =
      Array(Seq(Annotation(DOCUMENT, 0, input1.length - 1, input1, Map("sentence" -> "0"))))
    val expectedOutput2 =
      Array(Seq(Annotation(DOCUMENT, 0, input2.length - 1, input2, Map("sentence" -> "0"))))
    val expectedResultToken = Array(
      Seq(
        Annotation(TOKEN, 0, 3, "This", Map("sentence" -> "0")),
        Annotation(TOKEN, 5, 6, "is", Map("sentence" -> "0")),
        Annotation(TOKEN, 8, 10, "the", Map("sentence" -> "0")),
        Annotation(TOKEN, 12, 17, "second", Map("sentence" -> "0")),
        Annotation(TOKEN, 19, 23, "input", Map("sentence" -> "0"))))

    AssertAnnotations.assertFields(actualOutput1, expectedOutput1)
    AssertAnnotations.assertFields(actualOutput2, expectedOutput2)
    AssertAnnotations.assertFields(actualResultToken, expectedResultToken)
  }

  it should "fullAnnotate with LightPipeline and output two cols" taggedAs FastTest in {
    val multiDocumentAssembler = new MultiDocumentAssembler()
      .setInputCols("input1", "input2")
      .setOutputCols("output1", "output2")

    tokenizer.setInputCols("output2").setOutputCol("token")

    val pipelineModel =
      pipeline.setStages(Array(multiDocumentAssembler, tokenizer)).fit(twoInputDataset)

    val lightPipeline = new LightPipeline(pipelineModel)
    val actualResult = lightPipeline.fullAnnotate(input1, input2)

    val expectedResult =
      Map(
        "output1" -> List(Annotation(DOCUMENT, 0, input1.length - 1, input1, Map())),
        "output2" -> List(Annotation(DOCUMENT, 0, input2.length - 1, input2, Map())),
        "token" -> List(
          Annotation(TOKEN, 0, 3, "This", Map("sentence" -> "0")),
          Annotation(TOKEN, 5, 6, "is", Map("sentence" -> "0")),
          Annotation(TOKEN, 8, 10, "the", Map("sentence" -> "0")),
          Annotation(TOKEN, 12, 17, "second", Map("sentence" -> "0")),
          Annotation(TOKEN, 19, 23, "input", Map("sentence" -> "0"))))

    assert(actualResult.keySet == expectedResult.keySet)
    AssertAnnotations.assertFields(expectedResult.values.toArray, actualResult.values.toArray)

  }

  it should "fullAnnotate a list of inputs with LightPipeline and output of two cols" taggedAs FastTest in {
    val multiDocumentAssembler = new MultiDocumentAssembler()
      .setInputCols("input1", "input2")
      .setOutputCols("output1", "output2")

    tokenizer.setInputCols("output2").setOutputCol("token")

    val pipelineModel =
      pipeline.setStages(Array(multiDocumentAssembler, tokenizer)).fit(twoInputDataset)

    val lightPipeline = new LightPipeline(pipelineModel)
    val actualResults =
      lightPipeline.fullAnnotate(Array(input1, input1), Array(input2, input2))

    val expectedResults = Array(
      Map(
        "output1" -> List(Annotation(DOCUMENT, 0, input1.length - 1, input1, Map())),
        "output2" -> List(Annotation(DOCUMENT, 0, input2.length - 1, input2, Map())),
        "token" -> List(
          Annotation(TOKEN, 0, 3, "This", Map("sentence" -> "0")),
          Annotation(TOKEN, 5, 6, "is", Map("sentence" -> "0")),
          Annotation(TOKEN, 8, 10, "the", Map("sentence" -> "0")),
          Annotation(TOKEN, 12, 17, "second", Map("sentence" -> "0")),
          Annotation(TOKEN, 19, 23, "input", Map("sentence" -> "0")))),
      Map(
        "output1" -> List(Annotation(DOCUMENT, 0, input1.length - 1, input1, Map())),
        "output2" -> List(Annotation(DOCUMENT, 0, input2.length - 1, input2, Map())),
        "token" -> List(
          Annotation(TOKEN, 0, 3, "This", Map("sentence" -> "0")),
          Annotation(TOKEN, 5, 6, "is", Map("sentence" -> "0")),
          Annotation(TOKEN, 8, 10, "the", Map("sentence" -> "0")),
          Annotation(TOKEN, 12, 17, "second", Map("sentence" -> "0")),
          Annotation(TOKEN, 19, 23, "input", Map("sentence" -> "0")))))

    actualResults.zipWithIndex.foreach { case (actualResult, index) =>
      val expectedResult = expectedResults(index)
      assert(actualResult.keySet == expectedResult.keySet)
      AssertAnnotations.assertFields(expectedResult.values.toArray, actualResult.values.toArray)
    }
  }

  it should "annotate with LightPipeline and output two cols" taggedAs FastTest in {
    val multiDocumentAssembler = new MultiDocumentAssembler()
      .setInputCols("input1", "input2")
      .setOutputCols("output1", "output2")

    tokenizer.setInputCols("output2").setOutputCol("token")

    val pipelineModel =
      pipeline.setStages(Array(multiDocumentAssembler, tokenizer)).fit(twoInputDataset)

    val lightPipeline = new LightPipeline(pipelineModel)
    val actualResult = lightPipeline.annotate(input1, input2)

    val expectedResult =
      Map(
        "output1" -> List(input1),
        "output2" -> List(input2),
        "token" -> List("This", "is", "the", "second", "input"))

    assert(actualResult.keySet == expectedResult.keySet)
    assert(actualResult.values.toList == expectedResult.values.toList)
  }

  it should "annotate a list of inputs with LightPipeline and output two cols" taggedAs FastTest in {
    val multiDocumentAssembler = new MultiDocumentAssembler()
      .setInputCols("input1", "input2")
      .setOutputCols("output1", "output2")

    tokenizer.setInputCols("output2").setOutputCol("token")

    val pipelineModel =
      pipeline.setStages(Array(multiDocumentAssembler, tokenizer)).fit(twoInputDataset)

    val lightPipeline = new LightPipeline(pipelineModel)
    val actualResults =
      lightPipeline.annotate(Array(input1, input1), Array(input2, input2))

    val expectedResults = Array(
      Map(
        "output1" -> List(input1),
        "output2" -> List(input2),
        "token" -> List("This", "is", "the", "second", "input")),
      Map(
        "output1" -> List(input1),
        "output2" -> List(input2),
        "token" -> List("This", "is", "the", "second", "input")))

    actualResults.zipWithIndex.foreach { case (actualResult, index) =>
      val expectedResult = expectedResults(index)
      assert(actualResult.keySet == expectedResult.keySet)
      assert(actualResult.values.toList == expectedResult.values.toList)
    }

  }

  "MultiDocumentAssembler with one column" should "transform and output one col" taggedAs FastTest in {

    val multiDocumentAssembler = new MultiDocumentAssembler()
      .setInputCols("text")
      .setOutputCols("output")
    tokenizer.setInputCols("output").setOutputCol("token")

    val multiDocumentPipeline = pipeline.setStages(Array(multiDocumentAssembler, tokenizer))

    val result = multiDocumentPipeline.fit(dataset).transform(dataset)
    val actualOutput = AssertAnnotations.getActualResult(result, "output")
    val actualResultToken = AssertAnnotations.getActualResult(result, "token")

    val expectedOutput =
      Array(Seq(Annotation(DOCUMENT, 0, input1.length - 1, input1, Map("sentence" -> "0"))))
    val expectedResultToken = Array(
      Seq(
        Annotation(TOKEN, 0, 3, "This", Map("sentence" -> "0")),
        Annotation(TOKEN, 5, 6, "is", Map("sentence" -> "0")),
        Annotation(TOKEN, 8, 10, "the", Map("sentence" -> "0")),
        Annotation(TOKEN, 12, 16, "first", Map("sentence" -> "0")),
        Annotation(TOKEN, 18, 22, "input", Map("sentence" -> "0"))))

    AssertAnnotations.assertFields(actualOutput, expectedOutput)
    AssertAnnotations.assertFields(actualResultToken, expectedResultToken)
  }

  it should "fullAnnotate with LightPipeline and output one col" taggedAs FastTest in {
    val multiDocumentAssembler = new MultiDocumentAssembler()
      .setInputCols("text")
      .setOutputCols("output")
    tokenizer.setInputCols("output").setOutputCol("token")

    val pipelineModel = pipeline.setStages(Array(multiDocumentAssembler, tokenizer)).fit(dataset)

    val lightPipeline = new LightPipeline(pipelineModel)
    val actualResult = lightPipeline.fullAnnotate(input1)

    val expectedResult =
      Map(
        "output" -> List(Annotation(DOCUMENT, 0, input1.length - 1, input1, Map())),
        "token" -> List(
          Annotation(TOKEN, 0, 3, "This", Map("sentence" -> "0")),
          Annotation(TOKEN, 5, 6, "is", Map("sentence" -> "0")),
          Annotation(TOKEN, 8, 10, "the", Map("sentence" -> "0")),
          Annotation(TOKEN, 12, 16, "first", Map("sentence" -> "0")),
          Annotation(TOKEN, 18, 22, "input", Map("sentence" -> "0"))))

    assert(actualResult.keySet == expectedResult.keySet)
    AssertAnnotations.assertFields(expectedResult.values.toArray, actualResult.values.toArray)
  }

  it should "annotate with LightPipeline and output one col" taggedAs FastTest in {
    val multiDocumentAssembler = new MultiDocumentAssembler()
      .setInputCols("text")
      .setOutputCols("output")

    tokenizer.setInputCols("output").setOutputCol("token")

    val pipelineModel = pipeline.setStages(Array(multiDocumentAssembler, tokenizer)).fit(dataset)

    val lightPipeline = new LightPipeline(pipelineModel)
    val actualResult = lightPipeline.annotate(input1)

    val expectedResult =
      Map("output" -> List(input1), "token" -> List("This", "is", "the", "first", "input"))

    assert(actualResult.keySet == expectedResult.keySet)
    assert(actualResult.values.toList == expectedResult.values.toList)
  }

}
