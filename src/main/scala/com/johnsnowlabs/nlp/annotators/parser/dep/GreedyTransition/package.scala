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

package com.johnsnowlabs.nlp.annotators.parser.dep

package object GreedyTransition {
  type ClassNum  = Int
  type ClassName = String

  type DependencyIndex = Int
  type Move = Int

  type FeatureName = String
  type FeatureData = String
  type Score = Float

  type Word = String
  type Sentence = List[WordData]

  case class Feature(name: FeatureName, data: FeatureData)

  case class WordData(raw: Word, pos: ClassName = "", dep: DependencyIndex = -1) {
    lazy val norm: Word = {
      if (raw.length == 1) {
        if  (raw(0).isDigit) "#NUM#"
        else raw
      }
      else if (raw.forall(c => c.isDigit || c == '-' || c == '.')) {
        if (raw.forall(_.isDigit) && raw.length == 4) "#YEAR#" else "#NUM#"
      }
      else raw
    }
  }
}
