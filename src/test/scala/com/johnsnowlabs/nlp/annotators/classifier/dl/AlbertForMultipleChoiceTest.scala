package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.MultiDocumentAssembler
import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class AlbertForMultipleChoiceTest extends AnyFlatSpec with SparkSessionTest {

  import spark.implicits._
  val onnxModelPath = "/media/danilo/Data/Danilo/JSL/models/transformers/onnx"
  val sparkNLPModelPath = "/media/danilo/Data/Danilo/JSL/models/transformers/spark-nlp"
  val openVinoModelPath = "/media/danilo/Data/Danilo/JSL/models/transformers/openvino"

  val testDataframe =
    Seq(("The Eiffel Tower is located in which country?", "Germany, France, Italy"))
      .toDF("question", "context")

  "AlbertForMultipleChoice" should "loadSavedModel ONNX model" in {
   val albertForMultipleChoice = AlbertForMultipleChoice.loadSavedModel(s"$onnxModelPath/albert_multiple_choice", spark)
    albertForMultipleChoice.write.overwrite.save(s"$sparkNLPModelPath/onnx/albert_multiple_choice_onnx")
  }

  it should "loadSavedModel OpenVINO model" in {
    val albertForMultipleChoice = AlbertForMultipleChoice.loadSavedModel(s"$openVinoModelPath/albert_multiple_choice_openvino", spark)
    albertForMultipleChoice.write.overwrite.save(s"$sparkNLPModelPath/openvino/albert_multiple_choice_openvino")
  }

  it should "work for ONNX" in {
    val pipelineModel = getAlbertForMultipleChoicePipelineModel(s"$sparkNLPModelPath/onnx/albert_multiple_choice_onnx")
    val resultDf = pipelineModel.transform(testDataframe)
    resultDf.show(truncate = false)
  }

  it should "work for OpenVINO" in {
    val pipelineModel = getAlbertForMultipleChoicePipelineModel(s"$sparkNLPModelPath/openvino/albert_multiple_choice_openvino")
    val resultDf = pipelineModel.transform(testDataframe)
    resultDf.show(truncate = false)
  }

  private def getAlbertForMultipleChoicePipelineModel(modelPath: String) = {
    val documentAssembler = new MultiDocumentAssembler()
      .setInputCols("question", "context")
      .setOutputCols("document_question", "document_context")

    val bertForMultipleChoice = AlbertForMultipleChoice
      .load(modelPath)
      .setInputCols("document_question", "document_context")
      .setOutputCol("answer")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, bertForMultipleChoice))

    pipeline.fit(emptyDataSet)
  }

}
