package com.johnsnowlabs.nlp.util.io

import com.johnsnowlabs.nlp.SparkAccessor
import com.johnsnowlabs.tags.FastTest
import org.scalatest.FlatSpec

class SparkSqlHelperTestSpec extends FlatSpec {

  import SparkAccessor.spark.implicits._

  "Spark SQL Helper" should "get unique elements from an array column" taggedAs FastTest in {
    val row1 = Seq("Google", "is", "a", "nice", "search", "engine")
    val row2 = Seq("Let", "it", "snow", "let", "it", "snow", "let", "it", "snow")
    val document = Seq(row1, row2)
    val documentDataSet = document.toDS.toDF("text")
    val expectedUniqueElements = Seq(row1, Seq("Let", "it", "snow", "let"))
    val expectedColumns = Seq("text", "unique_text_elements")

    val actualUniqueElementsDF = SparkSqlHelper.uniqueArrayElements(documentDataSet, "text")
    val actualUniqueElements = actualUniqueElementsDF
      .select("unique_text_elements").rdd.map(row => row.getSeq(0)).collect.toSeq

    assert(expectedUniqueElements == actualUniqueElements)
    assert(expectedColumns == actualUniqueElementsDF.columns.toSeq)
  }

  it should "work when having nested arrays in one column" taggedAs FastTest in {
    val row1 = Seq(Seq("Google", "is", "a", "nice", "search", "engine"),
                   Seq("Let", "it", "snow", "let", "it", "snow", "let", "it", "snow"))
    val document = Seq(row1)
    val documentDataSet = document.toDS.toDF("text")
    val expectedUniqueElements = Seq(Seq("Google", "is", "a", "nice", "search", "engine", "Let", "it", "snow", "let"))
    val expectedColumns = Seq("text", "unique_text_elements")

    val actualUniqueElementsDF = SparkSqlHelper.uniqueArrayElements(documentDataSet, "text")
    val actualUniqueElements = actualUniqueElementsDF
      .select("unique_text_elements").rdd.map(row => row.getSeq(0)).collect.toSeq

    assert(expectedUniqueElements == actualUniqueElements)
    assert(expectedColumns == actualUniqueElementsDF.columns.toSeq)
  }

  it should "get unique elements from an array of integers column" taggedAs FastTest in {
    val document = Seq(Seq(300, 150, 300))
    val documentDataSet = document.toDS.toDF("numbers")
    val expectedUniqueElements = Seq(Seq(300, 150))
    val expectedColumns = Seq("numbers", "unique_numbers_elements")

    val actualUniqueElementsDF = SparkSqlHelper.uniqueArrayElements(documentDataSet, "numbers")
    val actualUniqueElements = actualUniqueElementsDF
      .select("unique_numbers_elements").rdd.map(row => row.getSeq(0)).collect.toSeq

    assert(expectedUniqueElements == actualUniqueElements)
    assert(expectedColumns == actualUniqueElementsDF.columns.toSeq)
  }

  it should "return the same dataset when a column is not an array type" taggedAs FastTest in {
    val document = Seq(300)
    val documentDataSet = document.toDS.toDF("text")
    val expectedColumns = Seq("text")

    val result = SparkSqlHelper.uniqueArrayElements(documentDataSet, "text")

    assert(expectedColumns == result.columns.toSeq)
  }

}
