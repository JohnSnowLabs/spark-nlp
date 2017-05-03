package com.jsl.nlp

import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, Suite}

trait SparkBasedTest extends BeforeAndAfterAll { this: Suite =>

  var spark: SparkSession = _

  override def beforeAll: Unit = {
    spark = SparkSession
      .builder()
      .appName("test")
      .getOrCreate()
  }

  override def afterAll: Unit = {
    spark.stop()
    spark = null
  }
}

case class TestRow(document: Document)