package com.jsl.nlp.annotators.sda.vivekn

import com.jsl.nlp.AnnotatorApproach
import com.jsl.nlp.util.io.ResourceHelper
import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

import scala.collection.mutable.{ListBuffer, Map => MMap}

/** Inspired on vivekn sentiment analysis algorithm
  * https://github.com/vivekn/sentiment/
  */
class ViveknSentimentApproach(override val uid: String)
  extends AnnotatorApproach[ViveknSentimentModel] {

  import com.jsl.nlp.AnnotatorType._

  override val description: String = "Vivekn inspired sentiment analysis model"


  /** Requires sentence boundaries to give score in context
    * Tokenization to make sure tokens are within bounds
    * Transitivity requirements are also required
    */
  val positiveSourcePath = new Param[String](this, "positiveSource", "source file for positive sentences")
  val negativeSourcePath = new Param[String](this, "negativeSource", "source file for negative sentences")
  val pruneCorpus = new BooleanParam(this, "pruneCorpus", "set to false if training corpus is small")
  setDefault(pruneCorpus, true)

  def this() = this(Identifiable.randomUID("VIVEKN"))

  override val annotatorType: AnnotatorType = SENTIMENT

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  def setPositiveSourcePath(value: String): this.type = set(positiveSourcePath, value)

  def setNegativeSourcePath(value: String): this.type = set(negativeSourcePath, value)

  def setCorpusPrune(value: Boolean): this.type = set(pruneCorpus, value)

  override def train(dataset: Dataset[_]): ViveknSentimentModel = {

    var positive: MMap[String, Int] = ResourceHelper.wordCount(
      $(positiveSourcePath),
      "txt",
      clean=false,
      f=Some(w => ViveknSentimentApproach.negateSequence(w))
    )
    var negative: MMap[String, Int] = ResourceHelper.wordCount(
      $(negativeSourcePath),
      "txt",
      clean=false,
      f=Some(w => ViveknSentimentApproach.negateSequence(w))
    )

    /** add negated words */
    negative = ResourceHelper.wordCount(
      $(positiveSourcePath),
      "txt",
      m=negative,
      clean=false,
      prefix=Some("not_"),
      f=Some(w => ViveknSentimentApproach.negateSequence(w))
    )
    positive = ResourceHelper.wordCount(
      $(negativeSourcePath),
      "txt",
      m=positive,
      clean=false,
      prefix=Some("not_"),
      f=Some(w => ViveknSentimentApproach.negateSequence(w))
    )

    /** remove features that appear only once */
    if ($(pruneCorpus)) {
      positive = positive.filter { case (_, count) => count > 1 }
      negative = negative.filter { case (_, count) => count > 1 }
    }

    val positiveTotals = positive.values.sum
    val negativeTotals = negative.values.sum

    def mutualInformation(word: String): Double = {
      val T = positiveTotals + negativeTotals
      val W = positive(word) + negative(word)
      var I: Double = 0.0
      if (W == 0) {
        return 0
      }
      if (negative(word) > 0) {
        val negativeDeltaScore: Double = (negativeTotals - negative(word)) * T / (T - W) / negativeTotals
        I += (negativeTotals - negative(word)) / T * scala.math.log(negativeDeltaScore)
        val negativeScore: Double = negative(word) * T / W / negativeTotals
        I += negative(word) / T * scala.math.log(negativeScore)
      }
      if (positive(word) > 0) {
        val positiveDeltaScore: Double = (positiveTotals - positive(word)) * T / (T - W) / positiveTotals
        I += (positiveTotals - positive(word)) / T * scala.math.log(positiveDeltaScore)
        val positiveScore: Double = positive(word) * T / W / positiveTotals
        I += positive(word) / T * scala.math.log(positiveScore)
      }
      I
    }

    val words = (positive.keys ++ negative.keys).toArray.distinct.sortBy(- mutualInformation(_))

    new ViveknSentimentModel()
      .setPositive(positive.toMap)
      .setNegative(negative.toMap)
      .setPositiveTotals(positiveTotals)
      .setNegativeTotals(negativeTotals)
      .setWords(words)
  }


}
private object ViveknSentimentApproach extends DefaultParamsReadable[ViveknSentimentApproach] {
  /** Detects negations and transforms them into not_ form */
  private[vivekn] def negateSequence(words: List[String]): List[String] = {
    val negations = Seq("not", "cannot", "no")
    val delims = Seq("?.,!:;")
    val result = ListBuffer.empty[String]
    var negation = false
    var prev: Option[String] = None
    var pprev: Option[String] = None
    words.foreach( word => {
      val processed = word.toLowerCase
      val negated = if (negation) "not_" + processed else processed
      result.append(negated)
      if (prev.isDefined) {
        val bigram = prev.get + " " + negated
        result.append(bigram)
        if (pprev.isDefined) {
          result.append(pprev.get + " " + bigram)
        }
        pprev = prev
      }
      prev = Some(negated)
      if (negations.contains(processed) || processed.endsWith("n't")) negation = !negation
      if (delims.exists(word.contains)) negation = false
    })
    result.toList
  }
}
