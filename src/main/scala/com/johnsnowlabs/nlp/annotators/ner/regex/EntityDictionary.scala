package com.johnsnowlabs.nlp.annotators.ner.regex

import com.johnsnowlabs.nlp.util.regex.{MatchStrategy, RegexRule, RuleFactory}

import scala.util.matching.Regex

class EntityDictionary(dict: Map[String, String]) extends Serializable {
  def predict(sentence: String): Seq[Map[String, String]] = {
    val entityRuleFactory = new RuleFactory(MatchStrategy.MATCH_ALL)
    dict.flatMap {
      case (keyWord, entity) =>
        val entityRegex = new Regex("(?is)(?:\\s+|^)(" + keyWord.toLowerCase.trim.replace("\\(", "").
          replace("\\)", "").replace(".", "\\.") + ")(?:\\s+|$|\\.)")
        entityRuleFactory.setRules(Seq(new RegexRule(entityRegex, "entity as regex")))
        entityRuleFactory.findMatch(sentence.toLowerCase).map {
          m =>
            Map(
              "word" -> m.content.matched.trim,
              "entity" -> entity,
              "start" -> m.content.start.toString,
              "end" -> m.content.end.toString
            )
        }
    }.to[Seq]
  }

  def getDict: Map[String, String] = dict
}
