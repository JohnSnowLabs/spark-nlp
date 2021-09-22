/*
 * Copyright 2017-2021 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.sbd.pragmatic

/**
 * Extends RuleSymbols with specific symbols used for the pragmatic approach. Right now, the only one.
 */
object PragmaticSymbols extends RuleSymbols {

  /**
   * ====================
   * BREAKING SYMBOLS
   * ====================
   */

  /**
   * Punctuation line breaker
   * alt 200
   */
  val PUNCT_INDICATOR = "╚"

  /**
   * Ellipsis breaker
   * looks up -> "..."
   * alt 203
   */
  val ELLIPSIS_INDICATOR = "╦"

  /**
   * =====================
   * NON BREAKING SYMBOLS
   * =====================
   */

  /**
   * Non separation dots for abbreviations
   * looks up -> .
   * alt 198
   */
  val ABBREVIATOR = "╞"

  /**
   * Non separation dots for numbers
   * looks up -> .
   * alt 199
   */
  val NUM_INDICATOR = "╟"

  /**
   * Period container non breaker
   * looks up -> .
   * alt 201
   */
  val MULT_PERIOD = "╔"

  /**
   * Special non breaking symbol
   * alt 202
   */
  val SPECIAL_PERIOD = "╩"

  /**
   * Question in quotes
   * alt 209
   */
  val QUESTION_IN_QUOTE = "╤" // ?

  /**
   * Exclamation mark in rules
   * alt 210
   */
  val EXCLAMATION_INDICATOR = "╥" // !


  override def symbolRecovery: Map[String, String] = super.symbolRecovery ++ Map(
    ABBREVIATOR -> ".",
    NUM_INDICATOR -> ".",
    MULT_PERIOD -> ".",
    QUESTION_IN_QUOTE -> "?",
    EXCLAMATION_INDICATOR -> "!",
    ELLIPSIS_INDICATOR -> "..."
  )

}
