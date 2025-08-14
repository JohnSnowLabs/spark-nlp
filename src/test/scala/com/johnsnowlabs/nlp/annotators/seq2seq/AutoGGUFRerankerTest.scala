package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.{SlowTest, FastTest}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.scalatest.flatspec.AnyFlatSpec

class AutoGGUFRerankerTest extends AnyFlatSpec {

  import ResourceHelper.spark.implicits._

  behavior of "AutoGGUFModelTest"

  lazy val documentAssembler: DocumentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  lazy val query: String = "A man is eating pasta."
  lazy val modelPath = "/tmp/bge-reranker-v2-m3-Q4_K_M.gguf"
  lazy val model: AutoGGUFReranker = AutoGGUFReranker
    .loadSavedModel(modelPath, ResourceHelper.spark)
    .setInputCols("document")
    .setOutputCol("completions")
    .setBatchSize(4)
    .setQuery(query)

  lazy val data: Dataset[Row] = Seq(
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "一个中国男人在吃面条",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A young girl is playing violin.").toDF("text").repartition(1)
  lazy val pipeline: Pipeline = new Pipeline().setStages(Array(documentAssembler, model))

  def assertAnnotationsNonEmpty(resultDf: DataFrame): Unit = {
    Annotation
      .collect(resultDf, "completions")
      .foreach(annotations => {
        println(annotations.head)
        println(annotations.head.metadata)
        assert(annotations.head.result.nonEmpty)
      })
  }

  it should "create batch completions" taggedAs SlowTest in {
    val pipeline = new Pipeline().setStages(Array(documentAssembler, model))
    val result = pipeline.fit(data).transform(data)
    assertAnnotationsNonEmpty(result)
  }

  it should "be serializable" taggedAs SlowTest in {
    lazy val data: Dataset[Row] = Seq(
      "A man is eating food.",
      "A man is eating a piece of bread.",
      "一个中国男人在吃面条",
      "The girl is carrying a baby.",
      "A man is riding a horse.",
      "A young girl is playing violin.").toDF("text").repartition(1)
    lazy val pipeline = new Pipeline().setStages(Array(documentAssembler, model))
    model.setNPredict(5)

    val pipelineModel = pipeline.fit(data)
    val savePath = "/tmp/saved_gguf"
    pipelineModel.stages.last
      .asInstanceOf[AutoGGUFReranker]
      .write
      .overwrite()
      .save(savePath)

    val loadedModel = AutoGGUFReranker.load(savePath)
    val newPipeline: Pipeline = new Pipeline().setStages(Array(documentAssembler, loadedModel))

    newPipeline
      .fit(data)
      .transform(data)
      .select("completions")
      .show(truncate = false)
  }

  it should "contain metadata when loadSavedModel" taggedAs SlowTest in {
    lazy val modelPath = "/tmp/bge-reranker-v2-m3-Q4_K_M.gguf"
    val model = AutoGGUFReranker.loadSavedModel(modelPath, ResourceHelper.spark)
    val metadata = model.getMetadata
    assert(metadata.nonEmpty)

    val metadataMap = model.getMetadataMap

    assert(metadataMap.nonEmpty)
  }

  it should "be able to also load pretrained AutoGGUFReranker" taggedAs SlowTest in {
    val model = AutoGGUFReranker
      .pretrained("Qwen2.5_VL_3B_Instruct_Q4_K_M_gguf")
      .setInputCols("document")
      .setOutputCol("completions")
      .setBatchSize(2)

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, model))
    val result = pipeline.fit(data).transform(data)

    result.show()
  }
}
