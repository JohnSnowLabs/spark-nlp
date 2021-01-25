package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{ContentProvider, DataBuilder}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

class RegexMatcherTestSpec extends FlatSpec with RegexMatcherBehaviors {
  val df: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.englishPhrase)
  val strategy = "MATCH_ALL"
  val rules = Array(
    ("the\\s\\w+", "followed by 'the'"),
    ("ceremonies", "ceremony")
  )
  "A full RegexMatcher pipeline with content" should behave like customizedRulesRegexMatcher(df, rules, strategy)
}
