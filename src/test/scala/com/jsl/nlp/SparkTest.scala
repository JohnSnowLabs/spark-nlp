package com.jsl.nlp

import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, FunSuite}

trait SparkTest extends FunSuite with BeforeAndAfterAll {

  var spark: SparkSession = _

  val testContent: String = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et " +
    "dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut " +
    "aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum " +
    "dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui " +
    "officia deserunt mollit anim id est laborum."

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