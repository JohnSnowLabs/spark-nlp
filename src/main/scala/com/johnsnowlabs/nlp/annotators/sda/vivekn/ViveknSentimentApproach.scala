package com.johnsnowlabs.nlp.annotators.sda.vivekn

import java.io.FileNotFoundException

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.ExternalResource
import com.johnsnowlabs.nlp.util.io.ResourceHelper.{SourceStream, listResourceDirectory}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

import scala.collection.mutable.{ListBuffer, Map => MMap}

/** Inspired on vivekn sentiment analysis algorithm
  * https://github.com/vivekn/sentiment/
  */
class ViveknSentimentApproach(override val uid: String)
  extends AnnotatorApproach[ViveknSentimentModel] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val description: String = "Vivekn inspired sentiment analysis model"


  /** Requires sentence boundaries to give score in context
    * Tokenization to make sure tokens are within bounds
    * Transitivity requirements are also required
    */
  val positiveSource = new ExternalResourceParam(this, "positiveSource", "positive sentiment file or folder")
  val negativeSource = new ExternalResourceParam(this, "negativeSource", "negative sentiment file or folder")
  val pruneCorpus = new IntParam(this, "pruneCorpus", "Removes unfrequent scenarios from scope. The higher the better performance. Defaults 1")
  setDefault(pruneCorpus, 1)

  def this() = this(Identifiable.randomUID("VIVEKN"))

  override val annotatorType: AnnotatorType = SENTIMENT

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  def setPositiveCorpus(value: ExternalResource): this.type = {
    require(value.options.contains("tokenPattern"), "vivekn corpus needs 'tokenPattern' regex for tagging words. e.g. \\S+")
    set(positiveSource, value)
  }

  def setNegativeCorpus(value: ExternalResource): this.type = {
    require(value.options.contains("tokenPattern"), "vivekn corpus needs 'tokenPattern' regex for tagging words. e.g. \\S+")
    set(negativeSource, value)
  }

  def setCorpusPrune(value: Int): this.type = set(pruneCorpus, value)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): ViveknSentimentModel = {

    val fromPositive: (MMap[String, Int], MMap[String, Int]) = ViveknSentimentApproach.ViveknWordCount(
      er=$(positiveSource),
      prune=$(pruneCorpus),
      f=w => ViveknSentimentApproach.negateSequence(w)
    )

    val (negative, positive) = ViveknSentimentApproach.ViveknWordCount(
      er=$(negativeSource),
      prune=$(pruneCorpus),
      f=w => ViveknSentimentApproach.negateSequence(w),
      fromPositive._2,
      fromPositive._1
    )

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

  private[vivekn] def ViveknWordCount(
                       er: ExternalResource,
                       prune: Int,
                       f: (List[String] => List[String]),
                       left: MMap[String, Int] = MMap.empty[String, Int].withDefaultValue(0),
                       right: MMap[String, Int] = MMap.empty[String, Int].withDefaultValue(0)
                     ): (MMap[String, Int], MMap[String, Int]) = {
    val regex = er.options("tokenPattern").r
    val prefix = "not_"
    val sourceStream = SourceStream(er.path)
    if (sourceStream.isResourceFolder) {
      try {
        listResourceDirectory(er.path)
            .map(filename => ViveknWordCount(ExternalResource(filename.toString, er.readAs, er.options), prune, f, left, right))
      } catch {
        case _: NullPointerException =>
          sourceStream
            .content
            .getLines()
            .map(fileName => ViveknWordCount(ExternalResource(er.path + "/" + fileName, er.readAs, er.options), prune, f, left, right))
            .toArray
          sourceStream.close()
      }
    } else {
      sourceStream.content.getLines.foreach(line => {
        val words = regex.findAllMatchIn(line).map(_.matched).toList
        f.apply(words).foreach(w => {
          left(w) += 1
          right(prefix + w) += 1
        })
      })
      sourceStream.close()
    }
    if (left.isEmpty || right.isEmpty) throw new FileNotFoundException("Word count dictionary for spell checker does not exist or is empty")
    if (prune > 0)
      (left.filter{case (_, v) => v > 1}, right.filter{case (_, v) => v > 1})
    else
      (left, right)
  }

}
