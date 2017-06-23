package com.jsl.nlp.annotators

import com.jsl.nlp.util.regex.MatchStrategy.MatchStrategy
import com.jsl.nlp.util.regex.{MatchStrategy, RegexRule, RuleFactory, TransformStrategy}
import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Created by Saif Addin on 5/7/2017.
  */
class RegexMatcher(override val uid: String) extends Annotator {


  protected val rules: Param[Seq[RegexRule]] = new Param(this, "Regex Rules", "regex patterns to match")
  protected val strategy: Param[MatchStrategy] = new Param(this, "Matching Strategy", "MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")

  private val matchFactory = RuleFactory.lateMatching(TransformStrategy.NO_TRANSFORM)(_)

  override val aType: String = RegexMatcher.aType

  override var requiredAnnotationTypes: Array[String] = Array()

  def this() = this(Identifiable.randomUID(RegexMatcher.aType))

  // ToDo: Decide whether we want to let create his own RegexRule or keep it private?
  def setPatterns(value: Seq[(String, String)]): this.type = {
    val newRules = value.map(v => RegexRule(v._1.r, v._2))
    set(
      rules,
      newRules
    )
  }

  def addPattern(value: (String, String)): this.type = {
    val newRule = RegexRule(value._1.r, value._2)
    set(
      rules,
      get(rules).getOrElse(Seq[RegexRule]() :+ newRule)
    )
  }

  def getPatterns: Seq[RegexRule] = $(rules)

  def setStrategy(value: String): this.type = set(strategy, value match {
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
      .findMatch(document.text).map(m => {
      Annotation(
        RegexMatcher.aType,
        m.content.start,
        m.content.end,
        Map(m.description -> m.content.matched)
      )
    })
  }
}
object RegexMatcher extends DefaultParamsReadable[RegexMatcher] {
  val aType = "regex"
}