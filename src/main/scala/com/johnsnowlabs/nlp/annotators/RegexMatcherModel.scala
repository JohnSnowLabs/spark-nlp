package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.serialization.ArrayFeature
import com.johnsnowlabs.nlp.util.regex.MatchStrategy.MatchStrategy
import com.johnsnowlabs.nlp.util.regex.{MatchStrategy, RegexRule, RuleFactory, TransformStrategy}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.Identifiable

/**
  * Matches regular expressions and maps them to specified values optionally provided
  * Rules are provided from external source file
  *
  * @param uid internal element required for storing annotator to disk
  * @@ rules: Set of rules to be mattched
  * @@ strategy:
  *    -- MATCH_ALL brings one-to-many results
  *    -- MATCH_FIRST catches only first match
  *    -- MATCH_COMPLETE returns only if match is entire target.
  *
  *    See [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/RegexMatcherTestSpec.scala]] for example on how to use this API.
  * @groupname anno Annotator types
  * @groupdesc anno Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc Parameters A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  *
  */
class RegexMatcherModel(override val uid: String) extends AnnotatorModel[RegexMatcherModel] with WithAnnotate[RegexMatcherModel] {

  /** Input annotator type: CHUNK
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = CHUNK
  /** Input annotator type: DOCUMENT
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  /** rules
    *
    * @group param
    **/
  val rules: ArrayFeature[(String, String)] = new ArrayFeature[(String, String)](this, "rules")

  /** MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE
    *
    * @group param
    **/
  val strategy: Param[String] = new Param(this, "strategy", "MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")

  def this() = this(Identifiable.randomUID("REGEX_MATCHER"))


  /** Can be any of MATCH_FIRST|MATCH_ALL|MATCH_COMPLETE
    *
    * @group setParam
    **/
  def setStrategy(value: String): this.type = set(strategy, value)

  /** Can be any of MATCH_FIRST|MATCH_ALL|MATCH_COMPLETE
    *
    * @group getParams
    **/
  def getStrategy: String = $(strategy).toString

  /** Path to file containing a set of regex,key pair. readAs can be LINE_BY_LINE or SPARK_DATASET. options contain option passed to spark reader if readAs is SPARK_DATASET.
    *
    * @group setParam
    **/
  def setRules(value: Array[(String, String)]): this.type = set(rules, value)

  /** Rules represented as Array of Tuples
    *
    * @group getParams
    **/
  def getRules: Array[(String, String)] = $$(rules)

  /** MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE */
  private def getFactoryStrategy: MatchStrategy = $(strategy) match {
    case "MATCH_ALL" => MatchStrategy.MATCH_ALL
    case "MATCH_FIRST" => MatchStrategy.MATCH_FIRST
    case "MATCH_COMPLETE" => MatchStrategy.MATCH_COMPLETE
    case _ => throw new IllegalArgumentException("Invalid strategy. must be MATCH_ALL|MATCH_FIRST|MATCH_COMPLETE")
  }

  lazy private val matchFactory = RuleFactory
    .lateMatching(TransformStrategy.NO_TRANSFORM)(getFactoryStrategy)
    .setRules($$(rules).map(r => new RegexRule(r._1, r._2)))

  /** one-to-many annotation that returns matches as annotations*/
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.zipWithIndex.flatMap { case (annotation, annotationIndex) =>
      matchFactory
        .findMatch(annotation.result).zipWithIndex.map { case (matched, idx) =>
          Annotation(
            outputAnnotatorType,
            matched.content.start,
            matched.content.end - 1,
            matched.content.matched,
            Map("identifier" -> matched.identifier, "sentence" -> annotationIndex.toString, "chunk" -> idx.toString)
          )
        }
    }
  }
}

object RegexMatcherModel extends ParamsAndFeaturesReadable[RegexMatcherModel]