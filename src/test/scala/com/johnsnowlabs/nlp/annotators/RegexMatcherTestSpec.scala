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

package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{ContentProvider, DataBuilder}
import org.apache.spark.sql.{Dataset, Row}
import org.scalatest.flatspec.AnyFlatSpec

class RegexMatcherTestSpec extends AnyFlatSpec with RegexMatcherBehaviors {
  val df: Dataset[Row] = DataBuilder.basicDataBuild(ContentProvider.englishPhrase)
  val strategy = "MATCH_ALL"
  val rules = Array(
    ("the\\s\\w+", "followed by 'the'"),
    ("ceremonies", "ceremony")
  )
  "A full RegexMatcher pipeline with content" should behave like customizedRulesRegexMatcher(df, rules, strategy)
}
