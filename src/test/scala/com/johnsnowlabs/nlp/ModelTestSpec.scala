package com.johnsnowlabs.nlp

import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.FlatSpec

class ModelTestSpec extends FlatSpec {

  import SparkAccessor.spark.implicits._

  val data: Dataset[Row] = ContentProvider.parquetData.limit(1000)
  val text = "This is a single sentence to annotate"
  val texts = data.select("text").as[String].collect

  "A ReadyModel basic pipeline" should "annotate datasets, strings and arrays" in {
    Model.en.Basic.annotate(data, "text").show(5)
    println(Model.en.Basic.annotate(text))
    Model.en.Basic.annotate(texts).take(5).foreach(println(_))
  }

}
