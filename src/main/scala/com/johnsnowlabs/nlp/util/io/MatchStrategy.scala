/*
 * Copyright 2017-2023 John Snow Labs
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

package com.johnsnowlabs.nlp.util.io

import com.johnsnowlabs.nlp.util.regex.RuleFactory

/** Allowed strategies for [[RuleFactory]] applications regarding replacement */
object MatchStrategy extends Enumeration {

  implicit def str2frmt(v: String): Format = {
    v.toUpperCase match {
      case "MATCH_ALL" => MATCH_ALL
      case "MATCH_FIRST" => MATCH_FIRST
      case "MATCH_COMPLETE" => MATCH_COMPLETE
      case _ =>
        throw new MatchError(
          s"Invalid MatchStrategy. Must be either of ${this.values.mkString("|")}")
    }
  }

  type Format = Value
  val MATCH_ALL, MATCH_FIRST, MATCH_COMPLETE = Value
}
