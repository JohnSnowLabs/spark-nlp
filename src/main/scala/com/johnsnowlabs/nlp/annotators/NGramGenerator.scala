package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable}
import org.apache.spark.ml.param.{IntParam, ParamValidators}
import org.apache.spark.ml.util.Identifiable

/**
  * A feature transformer that converts the input array of strings into an array of n-grams. Null
  * values in the input array are ignored.
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

  def setN(value: Int): this.type = set(n, value)

  /** @group getParam */
  def getN: Int = $(n)

  setDefault(
    n -> 2
  )

  private def generateNGrams(documents: Seq[(Int, Seq[Annotation])]): Seq[Annotation] = {

    val docAnnotation = documents.flatMap { case (idx: Int, annotation: Seq[Annotation]) =>

      val ngramsAnnotation = annotation.iterator.sliding($(n)).withPartial(false).map { tokens =>

        Annotation(
          outputAnnotatorType,
          tokens.head.begin,
          tokens.last.end,
          tokens.map(_.result).mkString(" "),
          tokens.head.metadata
        )
      }.toArray

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