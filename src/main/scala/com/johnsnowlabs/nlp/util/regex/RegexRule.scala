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