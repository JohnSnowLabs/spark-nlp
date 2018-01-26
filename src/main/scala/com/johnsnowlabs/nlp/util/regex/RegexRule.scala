package com.johnsnowlabs.nlp.util.regex

import scala.util.matching.Regex

/**
  * General structure for an identified regular expression
  * @param rx a java.matching.Regex object
  * @param identifier some description that might help link the regex to its meaning
  */
class RegexRule(rx: Regex, val identifier: String) extends Serializable {
  def this(rx: String, identifier: String) {
    this(rx.r, identifier)
  }
  val regex: Regex = rx
  val rule: String = rx.pattern.pattern()
}