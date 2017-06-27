package com.jsl.nlp.util.regex

import scala.util.matching.Regex

/**
  * Created by Saif Addin on 5/28/2017.
  */

/**
  * General structure for an identified regular expression
  * @param regex a java.matching.Regex object
  * @param identifier some description that might help link the regex to its meaning
  */
case class RegexRule(regex: Regex, identifier: String)
