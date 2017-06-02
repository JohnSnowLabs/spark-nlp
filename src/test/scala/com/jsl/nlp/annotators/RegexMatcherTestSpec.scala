package com.jsl.nlp.annotators

import com.jsl.nlp.util.{RegexRule, RegexStrategy}
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
    RegexRule("the\\s\\w+".r, "followed by the", RegexStrategy.MatchAll),
    RegexRule("ceremonies".r, "ceremony", RegexStrategy.MatchFirst)
  )

  "A full RegexMatcher pipeline with latin content" should behave like predefinedRulesRegexMatcher(latinBodyData, rules)

}
