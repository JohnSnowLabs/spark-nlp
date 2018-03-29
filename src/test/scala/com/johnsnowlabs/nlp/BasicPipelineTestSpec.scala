package com.johnsnowlabs.nlp

import com.johnsnowlabs.pretrained.pipelines.en.BasicPipeline
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.FlatSpec

class BasicPipelineTestSpec extends FlatSpec {

  import SparkAccessor.spark.implicits._

  val data: Dataset[Row] = ContentProvider.parquetData.limit(1000)
  val text = "This is a single sentence to annotate"
  val texts = data.select("text").as[String].collect

  "A ReadyModel basic pipeline" should "annotate datasets, strings and arrays" in {
    val transformed = BasicPipeline().annotate(data, "text")
    transformed.show(5)
    assert(transformed.columns.length == 8)
    println(BasicPipeline().annotate(text))
    BasicPipeline().annotate(texts).take(5).foreach(println(_))
  }

}
