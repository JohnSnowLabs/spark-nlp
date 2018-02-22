package com.johnsnowlabs.nlp.annotators.sda.vivekn

import java.io.FileNotFoundException

import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, AnnotatorType}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ResourceHelper}
import com.johnsnowlabs.nlp.util.io.ResourceHelper.{SourceStream, listResourceDirectory}
import com.johnsnowlabs.util.spark.MapAccumulator
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
  val sentimentCol = new Param[String](this, "sentimentCol", "column with the sentiment result of every row. Must be 'positive' or 'negative'")
  val positiveSource = new ExternalResourceParam(this, "positiveSource", "positive sentiment file or folder")
  val negativeSource = new ExternalResourceParam(this, "negativeSource", "negative sentiment file or folder")
  val pruneCorpus = new IntParam(this, "pruneCorpus", "Removes unfrequent scenarios from scope. The higher the better performance. Defaults 1")
  setDefault(pruneCorpus, 1)

  def this() = this(Identifiable.randomUID("VIVEKN"))

  override val annotatorType: AnnotatorType = SENTIMENT

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  def setSentimentCol(value: String): this.type = set(sentimentCol, value)

  def setPositiveSource(value: ExternalResource): this.type = {
    require(value.options.contains("tokenPattern"), "vivekn corpus needs 'tokenPattern' regex for tagging words. e.g. \\S+")
    set(positiveSource, value)
  }

  def setPositiveSource(path: String, tokenPattern: String, readAs: String = "LINE_BY_LINE"): this.type = {
    require(Seq("LINE_BY_LINE", "SPARK_DATASET").contains(readAs.toUpperCase), "readAs needs to be 'LINE_BY_LINE' or 'SPARK_DATASET'")
    set(positiveSource, ExternalResource(path, readAs, Map("tokenPattern" -> tokenPattern)))
  }

  def setNegativeSource(value: ExternalResource): this.type = {
    require(value.options.contains("tokenPattern"), "vivekn corpus needs 'tokenPattern' regex for tagging words. e.g. \\S+")
    set(negativeSource, value)
  }

  def setNegativeSource(path: String, tokenPattern: String, readAs: String = "LINE_BY_LINE"): this.type = {
    require(Seq("LINE_BY_LINE", "SPARK_DATASET").contains(readAs.toUpperCase), "readAs needs to be 'LINE_BY_LINE' or 'SPARK_DATASET'")
    set(negativeSource, ExternalResource(path, readAs, Map("tokenPattern" -> tokenPattern)))
  }

  def setCorpusPrune(value: Int): this.type = set(pruneCorpus, value)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): ViveknSentimentModel = {

    val (positive, negative): (Map[String, Long], Map[String, Long]) = {
      if (get(sentimentCol).isDefined) {
        import ResourceHelper.spark.implicits._
        val positiveDS = new MapAccumulator()
        val negativeDS = new MapAccumulator()
        val prefix = "not_"
        val tokenColumn = dataset.schema.fields
          .find(f => f.metadata.contains("annotatorType") && f.metadata.getString("annotatorType") == AnnotatorType.TOKEN)
          .map(_.name).get

        dataset.select(tokenColumn, $(sentimentCol)).as[(Array[Annotation], String)].foreach(tokenSentiment => {
          ViveknSentimentApproach.negateSequence(tokenSentiment._1.map(_.result).toList).foreach(w => {
            if (tokenSentiment._2 == "positive") {
              positiveDS.add(w, 1)
              negativeDS.add(prefix + w, 1)
            } else if (tokenSentiment._2 == "negative") {
              negativeDS.add(w, 1)
              positiveDS.add(prefix + w, 1)
            }
          })
        })
        (positiveDS.value.withDefaultValue(0), negativeDS.value.withDefaultValue(0))
      } else {
        val fromNegative: (MMap[String, Long], MMap[String, Long]) = ViveknSentimentApproach.ViveknWordCount(
          er=$(negativeSource),
          prune=$(pruneCorpus),
          f=w => ViveknSentimentApproach.negateSequence(w)
        )

        val (mpos, mneg) = ViveknSentimentApproach.ViveknWordCount(
          er=$(positiveSource),
          prune=$(pruneCorpus),
          f=w => ViveknSentimentApproach.negateSequence(w),
          fromNegative._2,
          fromNegative._1
        )
        (mpos.toMap.withDefaultValue(0), mneg.toMap.withDefaultValue(0))
      }
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
      .setPositive(positive)
      .setNegative(negative)
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
                                       left: MMap[String, Long] = MMap.empty[String, Long].withDefaultValue(0),
                                       right: MMap[String, Long] = MMap.empty[String, Long].withDefaultValue(0)
                                     ): (MMap[String, Long], MMap[String, Long]) = {
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
