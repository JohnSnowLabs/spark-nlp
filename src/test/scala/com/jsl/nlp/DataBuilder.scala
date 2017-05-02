package com.jsl.nlp

import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.{FlatSpec, Suite}

/**
  * Created by saif on 02/05/17.
  */
object DataBuilder extends FlatSpec with SparkBasedTest { this: Suite =>

  def basicDataBuild(content: String): Dataset[Row] = {
    val docs = Seq(
      TestRow(Document(
        "id",
        content
      ))
    )
    spark.createDataFrame(docs)
  }

}
