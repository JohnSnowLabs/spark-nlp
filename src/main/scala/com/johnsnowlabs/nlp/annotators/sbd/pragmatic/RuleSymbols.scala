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

package com.johnsnowlabs.nlp.annotators.sbd.pragmatic

/**
 * Base Symbols that may be extended later on. For now kept in the pragmatic scope.
 */
trait RuleSymbols {

  /**
   * Separation symbols for list items and numbers
   * 
   */
  val BREAK_INDICATOR = "\uF050"

  /**
   * looks up .
   */
  val DOT = "\uF051"

  /**
   * looks up ,
   */
  val COMMA = "\uF052"

  /**
   * looks up ;
   */
  val SEMICOLON = "\uF053"

  /**
   * looks up ?
   */
  val QUESTION = "\uF054"

  /**
   * looks up !
   */
  val EXCLAMATION = "\uF055"

  /**
   * ====================
   * PROTECTION SYMBOL
   * ====================
   */

  /**
   * Between punctuations marker
   */
  val PROTECTION_MARKER_OPEN = "\uF056"
  val PROTECTION_MARKER_CLOSE = "\uF057"

  /**
   * Magic regex ensures no breaking within protection
   */
  // http://rubular.com/r/Tq7lWxGkQl
  val UNPROTECTED_BREAK_INDICATOR = s"$BREAK_INDICATOR(?![^$PROTECTION_MARKER_OPEN]*$PROTECTION_MARKER_CLOSE)"

  def symbolRecovery: Map[String, String] = Map(
    DOT -> ".",
    SEMICOLON -> ";",
    QUESTION -> "?",
    EXCLAMATION -> "!",
    PROTECTION_MARKER_OPEN -> "",
    PROTECTION_MARKER_CLOSE -> ""
  )

}
