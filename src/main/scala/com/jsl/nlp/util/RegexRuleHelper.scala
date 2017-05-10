package com.jsl.nlp.util

import com.jsl.nlp.util.RegexStrategy.Strategy

import scala.util.matching.Regex

/**
  * Created by Saif Addin on 5/7/2017.
  */

case class RegexRule(regex: Regex, value: String, strategy: Strategy)

object RegexStrategy extends Enumeration {
  type Strategy = Value
  val MatchAll, MatchFirst, MatchComplete = Value
}

object RegexRuleHelper {

  def loadRules: Array[RegexRule] = {
    ???
  }

}