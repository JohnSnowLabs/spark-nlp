package com.jsl.nlp.annotators

import com.jsl.nlp.util.regex.MatchStrategy.MatchStrategy
import com.jsl.nlp.util.regex.{MatchStrategy, RegexRule, RuleFactory, TransformStrategy}
import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Created by Saif Addin on 5/7/2017.
  */

/**
  * Matches regular expressions and maps them to specified values optionally provided
  * Rules are provided from external source file
  * @param uid internal element required for storing annotator to disk
  * @@ rules: Set of rules to be mattched
  * @@ strategy:
  *   -- MATCH_ALL brings one-to-many results
  *   -- MATCH_FIRST catches only first match
  *   -- MATCH_COMPLETE returns only if match is entire target.
  */
class RegexMatcher(override val uid: String) extends Annotator {

  // ToDo: Check wether this annotator can be stored to disk as is. otherwise turn regex into string
  protected val rules: Param[Seq[RegexRule]] = new Param(this, "Regex Rules", "regex patterns to match")
  protected val strategy: Param[MatchStrategy] = new Param(this, "Matching Strategy", "MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")

  private val matchFactory = RuleFactory.lateMatching(TransformStrategy.NO_TRANSFORM)(_)

  override val annotatorType: String = RegexMatcher.annotationType

  override var requiredAnnotatorTypes: Array[String] = Array()

  def this() = this(Identifiable.randomUID(RegexMatcher.annotationType))

  // ToDo: Add load regex from source

  /** sets and overrides set of rules */
  def setPatterns(value: Seq[(String, String)]): this.type = {
    val newRules = value.map(v => RegexRule(v._1.r, v._2))
    set(
      rules,
      newRules
    )
  }

  /** adds pattern to current set of existing rules */
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

  /** one-to-many annotation that returns matches as annotations*/
  override def annotate(
                       document: Document, annotations: Seq[Annotation]
                       ): Seq[Annotation] = {
    matchFactory(getFactoryStrategy)
      .setRules(getPatterns)
      .findMatch(document.text).map(m => {
      Annotation(
        RegexMatcher.annotationType,
        m.content.start,
        m.content.end,
        Map(m.identifier -> m.content.matched)
      )
    })
  }
}
object RegexMatcher extends DefaultParamsReadable[RegexMatcher] {
  val annotationType = "regex"
}