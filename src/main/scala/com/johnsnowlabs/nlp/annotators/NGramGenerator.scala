package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable}
import org.apache.spark.ml.param.{BooleanParam, IntParam, Param, ParamValidators}
import org.apache.spark.ml.util.Identifiable

/**
  * A feature transformer that converts the input array of strings (annotatorType TOKEN) into an
  * array of n-grams (annotatorType CHUNK).
  * Null values in the input array are ignored.
  * It returns an array of n-grams where each n-gram is represented by a space-separated string of
  * words.
  *
  * When the input is empty, an empty array is returned.
  * When the input array length is less than n (number of elements per n-gram), no n-grams are
  * returned.
  */
class NGramGenerator (override val uid: String) extends AnnotatorModel[NGramGenerator] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  override val outputAnnotatorType: AnnotatorType = CHUNK

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  def this() = this(Identifiable.randomUID("NGRAM_GENERATOR"))

  /**
    * Minimum n-gram length, greater than or equal to 1.
    * Default: 2, bigram features
    *
    * @group param
    */
  val n: IntParam = new IntParam(this, "n", "number elements per n-gram (>=1)",
    ParamValidators.gtEq(1))

  val enableCumulative: BooleanParam = new BooleanParam(this, "enableCumulative",
    "whether to calculate just the actual n-grams or all n-grams from 1 through n")

  val delimiter: Param[String] = new Param[String](this, "delimiter",
    "Glue character used to join the tokens")

  def setN(value: Int): this.type = set(n, value)
  def setEnableCumulative(value: Boolean): this.type = set(enableCumulative, value)
  def setDelimiter(value: String): this.type = {
    require(value.length==1, "Delimiter should have length == 1")
    set(delimiter, value)
  }

  /** @group getParam */
  def getN: Int = $(n)
  def getEnableCumulative: Boolean = $(enableCumulative)
  def getDelimiter: String = $(delimiter)

  setDefault(
    n -> 2,
    enableCumulative -> false,
    delimiter -> " "
  )

  private def generateNGrams(documents: Seq[(Int, Seq[Annotation])]): Seq[Annotation] = {

    val docAnnotation = documents.flatMap { case (idx: Int, annotation: Seq[Annotation]) =>

      val range = if($(enableCumulative)) 1 to $(n) else $(n) to $(n)
      val ngramsAnnotation = range.flatMap(k =>{
        annotation.iterator.sliding(k).withPartial(false).map { tokens =>

          Annotation(
            outputAnnotatorType,
            tokens.head.begin,
            tokens.last.end,
            tokens.map(_.result).mkString($(delimiter)),
            tokens.head.metadata
          )
        }
      }).toArray
      ngramsAnnotation
    }

    docAnnotation
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val documentsWithTokens = annotations
      .filter(token => token.annotatorType == TOKEN)
      .groupBy(_.metadata.head._2.toInt)
      .toSeq
      .sortBy(_._1)

    generateNGrams(documentsWithTokens)

  }
}

object NGramGenerator extends ParamsAndFeaturesReadable[NGramGenerator]