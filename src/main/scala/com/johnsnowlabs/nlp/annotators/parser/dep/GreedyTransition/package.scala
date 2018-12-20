package com.johnsnowlabs.nlp.annotators.parser.dep

package object GreedyTransition {
  type ClassNum  = Int
  type ClassName = String

  type DependencyIndex = Int
  type Move = Int

  type FeatureName = String
  type FeatureData = String
  type Score = Double

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
