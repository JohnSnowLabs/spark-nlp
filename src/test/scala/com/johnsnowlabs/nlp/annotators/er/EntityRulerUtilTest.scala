package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.nlp.annotators.SparkSessionTest
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark
import org.apache.spark.sql.functions.{collect_list, collect_set}
import org.apache.spark.sql.types.{BooleanType, StringType, StructField, StructType}
import org.scalatest.flatspec.AnyFlatSpec

class EntityRulerUtilTest extends AnyFlatSpec with SparkSessionTest {

  //TODO: Remove unused tests

  "EntityRulerUtil" should "merge intervals" in {

    var intervals =
      List(List(57, 59), List(7, 10), List(7, 15), List(57, 64), List(12, 15), List(0, 15))
    var expectedMerged = List(List(0, 15), List(57, 64))

    var actualMerged = EntityRulerUtil.mergeIntervals(intervals)

    assert(expectedMerged == actualMerged)

    intervals = List(List(2, 3), List(4, 5), List(6, 7), List(8, 9), List(1, 10))
    expectedMerged = List(List(1, 10))

    actualMerged = EntityRulerUtil.mergeIntervals(intervals)
    assert(expectedMerged == actualMerged)

    intervals = List(List(5, 10), List(15, 20))
    expectedMerged = List(List(5, 10), List(15, 20))

    actualMerged = EntityRulerUtil.mergeIntervals(intervals)
    assert(expectedMerged == actualMerged)
  }

  "Reading JSON" should "work" in {

    val path = "src/test/resources/entity-ruler"

    println("keywords_regex_with_id.json")
    val KeywordsRegexWithId = spark.read.option("multiline", "true")
      .json(s"$path/keywords_regex_with_id.json")
    KeywordsRegexWithId.show(false)

    println("keywords_with_id.jsonl")
    val keywords_with_id = spark.read
      .json(s"$path/keywords_with_id.jsonl")
    keywords_with_id.show(false)

    println("keywords_regex.json")
    val keywords_regex = spark.read.option("multiline", "true")
      .json(s"$path/keywords_regex.json")
    keywords_regex.show(false)

    println("keywords_only.json")
    val keywordsOnly = spark.read.option("multiline", "true")
      .json(s"$path/keywords_only.json")
    keywordsOnly.show(false)

    println("keywords_regex_without_id")
    val KeywordsRegexWithoutRegexField = spark.read
      .json(s"$path/keywords_regex_without_id.jsonl")
    KeywordsRegexWithoutRegexField.show(false)

  }

  "Reading CSV" should "work" in {
    //keywords_without_regex_field.csv
    val path = "src/test/resources/entity-ruler"
    val patternOptions = Map("format" -> "csv", "delimiter" -> "|")
    val patternsSchema = StructType(Array(
      StructField("label", StringType, nullable = false),
      StructField("pattern", StringType, nullable = false),
      StructField("regex", BooleanType, nullable = true)
    ))

    val patternsDataFrame = spark.read
      .format(patternOptions("format"))
      .options(patternOptions)
      .option("delimiter", patternOptions("delimiter"))
      .schema(patternsSchema)
      .load(s"$path/keywords_without_regex_field.csv")
      .na.fill(value = false, Array("regex"))

    patternsDataFrame.show(false)

    val groupedByPatternsDataFrame = patternsDataFrame
      .groupBy("label", "regex")
      .agg(collect_set("pattern").alias("patterns"))

    groupedByPatternsDataFrame.show(false)
  }

}
