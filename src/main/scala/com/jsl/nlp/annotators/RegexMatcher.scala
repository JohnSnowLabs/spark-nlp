package com.jsl.nlp.annotators

import com.jsl.nlp.util.regex.MatchStrategy.MatchStrategy
import com.jsl.nlp.util.regex.{MatchStrategy, RegexRule, RuleFactory, TransformStrategy}
import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.param.Param

/**
  * Created by Saif Addin on 5/7/2017.
  */
class RegexMatcher extends Annotator {

  override val aType: String = RegexMatcher.aType

  override val requiredAnnotationTypes: Array[String] = Array()

  protected val rules: Param[Seq[RegexRule]] = new Param(this, "Regex Rules", "regex patterns to match")
  protected val strategy: Param[MatchStrategy] = new Param(this, "Matching Strategy", "MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")

  private val matchFactory = RuleFactory.lateMatching(TransformStrategy.NO_TRANSFORM)(_)

  // ToDo: Decide whether we want to let create his own RegexRule or keep it private?
  def setPatterns(value: Seq[(String, String)]): RegexMatcher = {
    val newRules = value.map(v => RegexRule(v._1.r, v._2))
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
    this
  }

  def getPatterns: Seq[RegexRule] = $(rules)

  def setStrategy(value: String): RegexMatcher = set(strategy, value match {
    case "MATCH_ALL" => MatchStrategy.MATCH_ALL
    case "MATCH_FIRST" => MatchStrategy.MATCH_FIRST
    case "MATCH_COMPLETE" => MatchStrategy.MATCH_COMPLETE
    case _ => throw new IllegalArgumentException("Invalid strategy. must be MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")
  })

  def getStrategy: String = $(strategy).toString

  private def getFactoryStrategy: MatchStrategy = $(strategy)

  override def annotate(
                       document: Document, annotations: Seq[Annotation]
                       ): Seq[Annotation] = {
    matchFactory(getFactoryStrategy)
      .setRules(getPatterns)
      .find(document.text).map(m => {
      Annotation(
        RegexMatcher.aType,
        m.start,
        m.end,
        Map(m.description -> m.content)
      )
    })
  }
}
object RegexMatcher {
  val aType = "regex"
}