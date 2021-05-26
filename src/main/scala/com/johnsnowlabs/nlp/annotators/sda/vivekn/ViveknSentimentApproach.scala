package com.johnsnowlabs.nlp.annotators.sda.vivekn

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, AnnotatorType}
import com.johnsnowlabs.util.spark.MapAccumulator
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

/** Inspired on vivekn sentiment analysis algorithm [[https://github.com/vivekn/sentiment/]].
  *
  * requires sentence boundaries to give score in context. Tokenization to make sure tokens are within bounds. Transitivity requirements are also required.
  *
  *
  * See [[https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/test/scala/com/johnsnowlabs/nlp/annotators/sda/vivekn]] for further reference on how to use this API.
  *
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
  * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
  */
class ViveknSentimentApproach(override val uid: String)
  extends AnnotatorApproach[ViveknSentimentModel] with ViveknSentimentUtils {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Vivekn inspired sentiment analysis model */
  override val description: String = "Vivekn inspired sentiment analysis model"


  /** column with the sentiment result of every row. Must be 'positive' or 'negative'
    *
    * @group param
    **/
  val sentimentCol = new Param[String](this, "sentimentCol", "column with the sentiment result of every row. Must be 'positive' or 'negative'")
  /** Removes unfrequent scenarios from scope. The higher the better performance. Defaults 1
    *
    * @group param
    **/
  val pruneCorpus = new IntParam(this, "pruneCorpus", "Removes unfrequent scenarios from scope. The higher the better performance. Defaults 1")
  /** proportion of feature content to be considered relevant. Defaults to 0.5
    *
    * @group param
    **/
  protected val importantFeatureRatio = new DoubleParam(this, "importantFeatureRatio", "proportion of feature content to be considered relevant. Defaults to 0.5")
  /** proportion to lookahead in unimportant features. Defaults to 0.025
    *
    * @group param
    **/
  protected val unimportantFeatureStep = new DoubleParam(this, "unimportantFeatureStep", "proportion to lookahead in unimportant features. Defaults to 0.025")
  /** content feature limit, to boost performance in very dirt text. Default disabled with -1
    *
    * @group param
    **/
  protected val featureLimit = new IntParam(this, "featureLimit", "content feature limit, to boost performance in very dirt text. Default disabled with -1")


  /** Set Proportion of feature content to be considered relevant. Defaults to 0.5
    *
    * @group setParam
    **/
  def setImportantFeatureRatio(v: Double): this.type = set(importantFeatureRatio, v)

  /** Set Proportion to lookahead in unimportant features. Defaults to 0.025
    *
    * @group setParam
    **/
  def setUnimportantFeatureStep(v: Double): this.type = set(unimportantFeatureStep, v)

  /** Set content feature limit, to boost performance in very dirt text. Default disabled with -1
    *
    * @group setParam
    **/
  def setFeatureLimit(v: Int): this.type = set(featureLimit, v)


  /** Get Proportion of feature content to be considered relevant. Defaults to 0.5
    *
    * @group getParam
    **/
  def getImportantFeatureRatio(v: Double): Double = $(importantFeatureRatio)

  /** Get Proportion to lookahead in unimportant features. Defaults to 0.025
    *
    * @group getParam
    **/
  def getUnimportantFeatureStep(v: Double): Double = $(unimportantFeatureStep)

  /** Get content feature limit, to boost performance in very dirt text. Default disabled with -1
    *
    * @group getParam
    **/
  def getFeatureLimit(v: Int): Int = $(featureLimit)

  setDefault(
    importantFeatureRatio -> 0.5,
    unimportantFeatureStep -> 0.025,
    featureLimit -> -1,
    pruneCorpus -> 1
  )

  def this() = this(Identifiable.randomUID("VIVEKN"))

  /** Output annotator type : SENTIMENT
    *
    * @group anno
    **/
  override val outputAnnotatorType: AnnotatorType = SENTIMENT
  /** Input annotator type : TOKEN, DOCUMENT
    *
    * @group anno
    **/
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN, DOCUMENT)

  /** Column with sentiment analysis row’s result for training. If not set, external sources need to be set instead. Column with the sentiment result of every row. Must be 'positive' or 'negative'
    *
    * @group setParam
    **/
  def setSentimentCol(value: String): this.type = set(sentimentCol, value)

  /** when training on small data you may want to disable this to not cut off infrequent words
    *
    * @group setParam
    **/
  def setCorpusPrune(value: Int): this.type = set(pruneCorpus, value)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): ViveknSentimentModel = {

    require(get(sentimentCol).isDefined, "ViveknSentimentApproach needs 'sentimentCol' to be set for training")


    val (positive, negative): (Map[String, Long], Map[String, Long]) = {
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
