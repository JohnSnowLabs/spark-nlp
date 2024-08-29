package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class BlenderBotTestSpec extends AnyFlatSpec {


  //TODO: https://huggingface.co/docs/transformers/v4.44.2/en/model_doc/blenderbot#overview
  val mainModelPath = "/models/transformers"

  "BlenderBot" should "loadSavedModel" in {
    val tfModel = s"$mainModelPath/tf/blenderbot-400M-distill/saved_model/1"
    val blenderBotModel = BlenderBotTransformer.loadSavedModel(tfModel, ResourceHelper.spark)
      .setInputCols(Array("documents"))
      .setOutputCol("generation")

    blenderBotModel.write.overwrite().save(s"$mainModelPath/spark-nlp/tf/blenderbot_spark_nlp")
  }

  it should "load model" in {

    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq(
            (1, "My friends are cool but they eat too many carbs."))
        )
      .toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    val blenderBotModel = BlenderBotTransformer.load(s"$mainModelPath/spark-nlp/tf/blenderbot_spark_nlp")
      .setInputCols(Array("document"))
      .setOutputCol("generation")

    val pipelineModel = new Pipeline()
      .setStages(Array(documentAssembler, blenderBotModel))
      .fit(testData)


    val dataset = pipelineModel.transform(testData)

    dataset.show(truncate = false)
  }

}
