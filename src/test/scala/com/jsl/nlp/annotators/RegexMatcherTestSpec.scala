package com.jsl.nlp.annotators

import com.jsl.nlp.{ContentProvider, DataBuilder}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest._

/**
  * Created by Saif Addin on 5/10/2017.
  */
class RegexMatcherTestSpec extends FlatSpec with RegexMatcherBehaviors {

  val regexMatcher = new RegexMatcher
  "a RegexMatcher" should s"be of type ${RegexMatcher.aType}" in {
    assert(regexMatcher.aType == RegexMatcher.aType)
  }

  val latinBodyData: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.latinBody)

  val rules = Seq(
    ("the\\s\\w+", "followed by 'the'"),
    ("ceremonies", "ceremony")
  )

  val strategy = "MATCH_ALL"

  "A full RegexMatcher pipeline with latin content" should behave like predefinedRulesRegexMatcher(latinBodyData, rules, strategy)

}
