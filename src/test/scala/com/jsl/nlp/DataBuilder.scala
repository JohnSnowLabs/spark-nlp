package com.jsl.nlp

import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.{FlatSpec, Suite}

/**
  * Created by saif on 02/05/17.
  */
object DataBuilder extends FlatSpec with SparkBasedTest { this: Suite =>
  import spark.implicits._
  case class StructContainer(document: Document)
  def basicDataBuild(content: String): Dataset[Row] = {
    val docs = Seq(
      StructContainer(
        Document(
          "id",
          content
        )
      )
    )
    docs.toDS().toDF("document")
  }

}
