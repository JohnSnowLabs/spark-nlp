package com.jsl.nlp

import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, Suite}

trait SparkBasedTest extends BeforeAndAfterAll { this: Suite =>

  val spark: SparkSession = SparkSession
    .builder()
    .appName("test")
    .master("local[4]")
    .getOrCreate()

}