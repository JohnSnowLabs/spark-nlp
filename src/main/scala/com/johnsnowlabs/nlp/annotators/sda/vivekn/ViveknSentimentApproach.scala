package com.johnsnowlabs.nlp.annotators.sda.vivekn

import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, AnnotatorType}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.util.spark.MapAccumulator
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

import scala.collection.mutable.{Map => MMap}

/** Inspired on vivekn sentiment analysis algorithm
  * https://github.com/vivekn/sentiment/
  */
class ViveknSentimentApproach(override val uid: String)
  extends AnnotatorApproach[ViveknSentimentModel] with ViveknSentimentUtils {

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

  protected val importantFeatureRatio = new DoubleParam(this, "importantFeatureRatio", "proportion of feature content to be considered relevant. Defaults to 0.5")
  protected val unimportantFeatureStep = new DoubleParam(this, "unimportantFeatureStep", "proportion to lookahead in unimportant features. Defaults to 0.025")
  protected val featureLimit = new IntParam(this, "featureLimit", "content feature limit, to boost performance in very dirt text. Default disabled with -1")

  def setImportantFeatureRatio(v: Double): this.type = set(importantFeatureRatio, v)
  def setUnimportantFeatureStep(v: Double): this.type = set(unimportantFeatureStep, v)
  def setFeatureLimit(v: Int): this.type = set(featureLimit, v)

  def getImportantFeatureRatio(v: Double): Double = $(importantFeatureRatio)
  def getUnimportantFeatureStep(v: Double): Double = $(unimportantFeatureStep)
  def getFeatureLimit(v: Int): Int = $(featureLimit)

  setDefault(
    importantFeatureRatio -> 0.5,
    unimportantFeatureStep -> 0.025,
    featureLimit -> -1,
    pruneCorpus -> 1
  )

  def this() = this(Identifiable.randomUID("VIVEKN"))

  override val annotatorType: AnnotatorType = SENTIMENT

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  def setSentimentCol(value: String): this.type = set(sentimentCol, value)

  def setPositiveSource(value: ExternalResource): this.type = {
    require(value.options.contains("tokenPattern"), "vivekn corpus needs 'tokenPattern' regex for tagging words. e.g. \\S+")
    set(positiveSource, value)
  }

  def setPositiveSource(path: String,
                        tokenPattern: String,
                        readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                        options: Map[String, String] = Map("format" -> "text")): this.type =
    set(positiveSource, ExternalResource(path, readAs, options ++ Map("tokenPattern" -> tokenPattern)))

  def setNegativeSource(value: ExternalResource): this.type = {
    require(value.options.contains("tokenPattern"), "vivekn corpus needs 'tokenPattern' regex for tagging words. e.g. \\S+")
    set(negativeSource, value)
  }

  def setNegativeSource(path: String,
                        tokenPattern: String,
                        readAs: ReadAs.Format = ReadAs.LINE_BY_LINE,
                        options: Map[String, String] = Map("format" -> "text")): this.type =
    set(negativeSource, ExternalResource(path, readAs, options ++ Map("tokenPattern" -> tokenPattern)))

  def setCorpusPrune(value: Int): this.type = set(pruneCorpus, value)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): ViveknSentimentModel = {

    val (positive, negative): (Map[String, Long], Map[String, Long]) = {
      if (get(sentimentCol).isDefined) {
        import ResourceHelper.spark.implicits._
        val positiveDS = new MapAccumulator()
        val negativeDS = new MapAccumulator()
        dataset.sparkSession.sparkContext.register(positiveDS)
        dataset.sparkSession.sparkContext.register(negativeDS)
        val prefix = "not_"
        val tokenColumn = dataset.schema.fields
          .find(f => f.metadata.contains("annotatorType") && f.metadata.getString("annotatorType") == AnnotatorType.TOKEN)
          .map(_.name).get

        dataset.select(tokenColumn, $(sentimentCol)).as[(Array[Annotation], String)].foreach(tokenSentiment => {
          negateSequence(tokenSentiment._1.map(_.result)).foreach(w => {
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
        val fromNegative: (MMap[String, Long], MMap[String, Long]) = ViveknWordCount(
          er=$(negativeSource),
          prune=$(pruneCorpus),
          f=w => negateSequence(w.toArray)
        )

        val (mpos, mneg) = ViveknWordCount(
          er=$(positiveSource),
          prune=$(pruneCorpus),
          f=w => negateSequence(w.toArray),
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
      .setImportantFeatureRatio($(importantFeatureRatio))
      .setUnimportantFeatureStep($(unimportantFeatureStep))
      .setFeatureLimit($(featureLimit))
      .setPositive(positive)
      .setNegative(negative)
      .setPositiveTotals(positiveTotals)
      .setNegativeTotals(negativeTotals)
      .setWords(words)
  }


}

private object ViveknSentimentApproach extends DefaultParamsReadable[ViveknSentimentApproach]
