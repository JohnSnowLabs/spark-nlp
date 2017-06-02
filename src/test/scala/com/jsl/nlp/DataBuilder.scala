package com.jsl.nlp

import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by saif on 02/05/17.
  */
object DataBuilder extends FlatSpec with BeforeAndAfterAll { this: Suite =>

  import SparkAccessor.spark.implicits._
  case class StructContainer(document: Document)
  def basicDataBuild(content: String): Dataset[Row] = {
    val docs = Seq(
      StructContainer(
        Document(
          "id",
          content,
          Map[String, String]()
        )
      )
    )
    docs.toDS().toDF("document")
  }

}
