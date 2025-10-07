package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.finisher.GGUFRankingFinisher
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.scalatest.flatspec.AnyFlatSpec

class AutoGGUFRerankerTest extends AnyFlatSpec {

  import ResourceHelper.spark.implicits._

  behavior of "AutoGGUFRerankerTest"

  lazy val documentAssembler: DocumentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  lazy val query: String = "A man is eating pasta."
  lazy val modelPath = "/tmp/bge_reranker_v2_m3_Q4_K_M.gguf"
  lazy val model: AutoGGUFReranker = AutoGGUFReranker
    .loadSavedModel(modelPath, ResourceHelper.spark)
    .setInputCols("document")
    .setOutputCol("reranked_documents")
    .setBatchSize(4)
    .setQuery(query)

  lazy val data: Dataset[Row] = Seq(
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "一个中国男人在吃面条",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A young girl is playing violin.").toDF("text").repartition(1)

  lazy val finisher: GGUFRankingFinisher = new GGUFRankingFinisher()
    .setInputCols("reranked_documents")
    .setOutputCol("ranked_documents")
    .setTopK(-1)
    .setMinRelevanceScore(0.1)
    .setMinMaxScaling(true)
  lazy val pipeline: Pipeline =
    new Pipeline().setStages(Array(documentAssembler, model))

  def assertAnnotationsNonEmpty(resultDf: DataFrame): Unit = {
    Annotation
      .collect(resultDf, "reranked_documents")
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
      .select("reranked_documents")
      .show(truncate = false)
  }

  it should "contain metadata when loadSavedModel" taggedAs SlowTest in {
    lazy val modelPath = "/tmp/bge_reranker_v2_m3_Q4_K_M.gguf"
    val model = AutoGGUFReranker.loadSavedModel(modelPath, ResourceHelper.spark)
    val metadata = model.getMetadata
    assert(metadata.nonEmpty)

    val metadataMap = model.getMetadataMap

    assert(metadataMap.nonEmpty)
  }

  it should "be able to also load pretrained AutoGGUFReranker" taggedAs SlowTest in {
    val model = AutoGGUFReranker
      .pretrained()
      .setInputCols("document")
      .setOutputCol("reranked_documents")
      .setBatchSize(2)

    val pipeline =
      new Pipeline().setStages(Array(documentAssembler, model))
    val result = pipeline.fit(data).transform(data)

    result.show()
  }
  it should "throw an error if the query is not set" taggedAs SlowTest in {
    val model: AutoGGUFReranker = AutoGGUFReranker
      .loadSavedModel(modelPath, ResourceHelper.spark)
      .setInputCols("document")
      .setOutputCol("reranked_documents")
      .setBatchSize(4)
    val pipeline = new Pipeline().setStages(Array(documentAssembler, model))
    assertThrows[org.apache.spark.SparkException] {
      val result = pipeline.fit(data).transform(data)
      result.show()
    }
  }

  it should "be able to finisher the reranked documents" taggedAs SlowTest in {
    model.setQuery(query)
    val pipeline = new Pipeline().setStages(Array(documentAssembler, model, finisher))
    val result = pipeline.fit(data).transform(data)

//    assertAnnotationsNonEmpty(result)
    result.select("ranked_documents").show(truncate = false)
  }

  it should "load models with deprecated parameters" taggedAs SlowTest in {
    // testing only, should be able to load
    AutoGGUFReranker.pretrained("Nomic_Embed_Text_v1.5.Q8_0.gguf")
  }
}
