/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.tokenizer.normalizer

import org.apache.commons.lang.StringUtils

import scala.util.matching.Regex

private[johnsnowlabs] class MosesPunctNormalizer() {

  private val COMBINED_REPLACEMENT = List(
    (raw"""\r|\u00A0«\u00A0|«\u00A0|«|\u00A0»\u00A0|\u00A0»|»|“|《|》|」|「""".r, raw""""""),

    (raw""" +""".r, raw""" """),

    (raw"""\( ?""".r, raw"""("""),

    (raw""" ?\)""".r, raw""")"""),

    (raw""" :|\u00A0:|∶|：""".r, raw""":"""),

    (raw""" ;""|\u00A0;|；""".r, raw""";"""),

    (raw"""`|´|‘|‚|’|’""".r, raw"""'"""),

    (raw"""''|„|“|”|''|´´""".r, raw""" " """)
  )
  private val EXTRA_WHITESPACE = List(
    (raw"""(""", raw""" ("""),
    (raw""")""", raw""") """),
    (raw"""\) ([.!:?;.r,])""".r, raw""")$$1"""),
    (raw"""(\d) %""".r, raw"""$$1%""")
  )

  private val NORMALIZE_UNICODE = List(
    (raw"""–|━""".r, raw"""-"""),
    (raw"""—""", raw""" - """),
    (raw"""([a-zA-Z])‘([a-zA-Z])""".r, raw"""$$1'$$2"""),
    (raw"""([a-zA-Z])’([a-zA-Z])""".r, raw"""$$1'$$2"""),
    (raw"""…"""", raw"""...""")
  )

  private val HANDLE_PSEUDO_SPACES = List(
    (raw"""\u00A0%""".r, raw"""%"""),
    (raw"""nº\u00A0""".r, raw"""nº """),
    (raw"""\u00A0ºC""".r, raw""" ºC"""),
    (raw"""\u00A0cm""".r, raw""" cm"""),
    (raw"""\u00A0\\?""".r, raw"""?"""),
    (raw"""\u00A0\\!""".r, raw"""!"""),
    (raw""""".,\u00A0""".r, raw""""", """)
  )

  private val EN_QUOTATION_FOLLOWED_BY_COMMA = List(
    (raw""""([,.]+)""".r, raw"""$$1""")
  )

  private val DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA = List(
    (raw""","""", raw"""","""),
    (raw"""(\.+)"(\s*?[^<])""".r, raw""""$$1$$2""")
  )

  private val DE_ES_CZ_CS_FR = List(
    (raw"""(\\d)\u00A0(\\d)""".r, raw"""$$1,$$2""")
  )

  private val OTHER = List(
    (raw"""(\\d)\u00A0(\\d)""".r, raw"""$$1.$$2""")
  )

  private val REPLACE_UNICODE_PUNCTUATION = List(
    (raw"""，""", raw""","""),
    (raw"""。\s*?""".r, raw""". """),
    (raw"""、""", raw""","""),
    (raw"""？""", raw"""?"""),
    (raw"""！""", raw"""!"""),
    (raw"""０""", raw"""0"""),
    (raw"""１""", raw"""1"""),
    (raw"""２""", raw"""2"""),
    (raw"""３""", raw"""3"""),
    (raw"""４""", raw"""4"""),
    (raw"""５""", raw"""5"""),
    (raw"""６""", raw"""6"""),
    (raw"""７""", raw"""7"""),
    (raw"""８""", raw"""8"""),
    (raw"""９""", raw"""9"""),
    (raw"""．\s*?""".r, raw""". """),
    (raw"""～""", raw"""~"""),
    (raw"""…""", raw"""..."""),
    (raw"""━""", raw"""-"""),
    (raw"""〈""", raw"""<"""),
    (raw"""〉""", raw""">"""),
    (raw"""【""", raw"""["""),
    (raw"""】""", raw"""]"""),
    (raw"""％""", raw"""%""")
  )

  private val substitutions = List.concat(
    COMBINED_REPLACEMENT,
    EXTRA_WHITESPACE,
    NORMALIZE_UNICODE,
    HANDLE_PSEUDO_SPACES,
    EN_QUOTATION_FOLLOWED_BY_COMMA,
    DE_ES_FR_QUOTATION_FOLLOWED_BY_COMMA,
    DE_ES_CZ_CS_FR,
    OTHER,
    REPLACE_UNICODE_PUNCTUATION)

  def normalize(text: String): String = {
    var acc: String = text

    substitutions
      .foreach {
        case (pattern, replacement) => pattern match {
          case pattern: Regex => acc = pattern.replaceAllIn(acc, replacement)
          case pattern: String => acc = StringUtils.replace(acc, pattern, replacement)
        }
      }
    acc
  }


  private val printingCharTypes = Set(
    Character.CONTROL,
    Character.DIRECTIONALITY_COMMON_NUMBER_SEPARATOR,
    Character.FORMAT,
    Character.PRIVATE_USE,
    Character.SURROGATE,
    Character.UNASSIGNED
  )

  //  Port of https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/remove-non-printing-char.perl
  def removeNonPrintingChar(t: String): String = {
    def isNonPrintingChar(c: Char): Boolean = !printingCharTypes.contains(Character.getType(c).toByte)

    t.toCharArray.filter(isNonPrintingChar).mkString
  }
}
