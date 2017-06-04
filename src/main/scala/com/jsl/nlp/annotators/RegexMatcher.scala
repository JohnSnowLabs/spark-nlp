package com.jsl.nlp.annotators

import com.jsl.nlp.util.regex.RuleStrategy.MatchStrategy
import com.jsl.nlp.util.regex.{RegexRule, RuleFactory, RuleStrategy}
import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.param.Param

/**
  * Created by Saif Addin on 5/7/2017.
  */
class RegexMatcher extends Annotator {

  override val aType: String = RegexMatcher.aType

  override val requiredAnnotationTypes: Array[String] = Array()

  private val rules: Param[Seq[RegexRule]] = new Param(this, "Regex Rules", "regex patterns to match")
  private val strategy: Param[MatchStrategy] = new Param(this, "Matching Strategy", "MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")

  private val matchFactory = new RuleFactory(_)(RuleStrategy.NO_TRANSFORM)

  // ToDo: Decide whether we want to let create his own RegexRule or keep it private?
  def setPatterns(value: Seq[(String, String)]): RegexMatcher = {
    val newRules = value.map(v => RegexRule(v._1.r, v._2))
    matchFactory(_).setRules(newRules)
    set(
      rules,
      newRules
    )
  }

  def addPattern(value: (String, String)): RegexMatcher = {
    val newRule = RegexRule(value._1.r, value._2)
    set(
      rules,
      get(rules).getOrElse(Seq[RegexRule]() :+ newRule)
    )
    matchFactory(_).addRule(newRule)
    this
  }

  def getPatterns: Seq[RegexRule] = $(rules)

  def setStrategy(value: String): RegexMatcher = set(strategy, value match {
    case "MATCH_ALL" => RuleStrategy.MATCH_ALL
    case "MATCH_FIRST" => RuleStrategy.MATCH_FIRST
    case "MATCH_COMPLETE" => RuleStrategy.MATCH_COMPLETE
    case _ => throw new IllegalArgumentException("Invalid strategy. must be MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")
  })

  def getStrategy: MatchStrategy = $(strategy)

  override def annotate(
                       document: Document, annotations: Seq[Annotation]
                       ): Seq[Annotation] = {
    getStrategy match {
      case RuleStrategy.MATCH_ALL => matchFactory(RuleStrategy.MATCH_ALL).find(document.text).map(m => {
        Annotation(
          RegexMatcher.aType,
          m.start,
          m.end,
          Map(m.description -> m.content)
        )
      })
    }
  }
}
object RegexMatcher {
  val aType = "regex"
}