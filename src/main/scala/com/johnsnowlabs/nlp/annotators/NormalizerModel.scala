package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable}
import org.apache.spark.ml.param.{BooleanParam, StringArrayParam}
import org.apache.spark.ml.util.Identifiable

class NormalizerModel(override val uid: String) extends AnnotatorModel[NormalizerModel] {

  override val outputAnnotatorType: AnnotatorType = TOKEN

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  case class TokenizerAndNormalizerMap(beginTokenizer: Int, endTokenizer: Int, token: String,
                                       beginNormalizer: Int, endNormalizer: Int, normalizer: String)

  val cleanupPatterns = new StringArrayParam(this, "cleanupPatterns",
    "normalization regex patterns which match will be removed from token")

  val lowercase = new BooleanParam(this, "lowercase", "whether to convert strings to lowercase")

  protected val slangDict: MapFeature[String, String] = new MapFeature(this, "slangDict")

  val slangMatchCase = new BooleanParam(this, "slangMatchCase", "whether or not to be case sensitive to match slangs. Defaults to false.")

  def this() = this(Identifiable.randomUID("NORMALIZER"))

  def setCleanupPatterns(value: Array[String]): this.type = set(cleanupPatterns, value)

  def getCleanupPatterns: Array[String] = $(cleanupPatterns)

  def setLowercase(value: Boolean): this.type = set(lowercase, value)

  def getLowercase: Boolean = $(lowercase)

  def setSlangDict(value: Map[String, String]): this.type = set(slangDict, value)

  def setSlangMatchCase(value: Boolean): this.type = set(slangMatchCase, value)

  def getSlangMatchCase: Boolean = $(slangMatchCase)

  def applyRegexPatterns(word: String): String = {

    val nToken = {
      get(cleanupPatterns).map(_.foldLeft(word)((currentText, compositeToken) => {
        currentText.replaceAll(compositeToken, "")
      })).getOrElse(word)
    }
    nToken
  }

  protected def getSlangDict: Map[String, String] = $$(slangDict)

  /** ToDo: Review implementation, Current implementation generates spaces between non-words, potentially breaking tokens */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val normalizedAnnotations = annotations.flatMap { originalToken =>

      /** slang dictionary keys should have been lowercased if slangMatchCase is false */
      val unslanged = $$(slangDict).get(
        if ($(slangMatchCase)) originalToken.result
        else originalToken.result.toLowerCase
      )

      /** simple-tokenize the unslanged slag phrase */
      val tokenizedUnslang = {
        unslanged.map(unslang => {
          unslang.split(" ")
        }).getOrElse(Array(originalToken.result))
      }

      val cleaned = tokenizedUnslang.map(word => applyRegexPatterns(word))

      val cased = if ($(lowercase)) cleaned.map(_.toLowerCase) else cleaned

      cased.filter(_.nonEmpty).map { finalToken => {
        Annotation(
          outputAnnotatorType,
          originalToken.begin,
          originalToken.begin + finalToken.length - 1,
          finalToken,
          originalToken.metadata
        )
      }}

    }

    if (normalizedAnnotations.size > annotations.size) {
      normalizedAnnotations
    } else {
      resetIndexAnnotations(annotations, normalizedAnnotations)
    }

  }

  private def resetIndexAnnotations(tokenizerAnnotations: Seq[Annotation], normalizerAnnotations: Seq[Annotation]):
  Seq[Annotation] = {
    val wrongIndex = getFirstAnnotationIndexWithWrongIndexValues(tokenizerAnnotations, normalizerAnnotations)
    if (wrongIndex == -1) {
      normalizerAnnotations
    } else {
      val offset = getOffset(tokenizerAnnotations, normalizerAnnotations, wrongIndex)
      val rightAnnotations = normalizerAnnotations.slice(0, wrongIndex)
      val wrongAnnotations = normalizerAnnotations.slice(wrongIndex, normalizerAnnotations.length)
      val resetAnnotations = wrongAnnotations.zipWithIndex.map{ case (normalizedToken, index) =>
        var begin = normalizedToken.begin - offset
        if (begin < 0) {
          begin =  normalizedToken.begin - rightAnnotations.last.end
        }
        val end = begin + normalizedToken.result.length - 1
        Annotation(normalizedToken.annotatorType, begin, end, normalizedToken.result, normalizedToken.metadata)
      }
      val fullResetAnnotations = rightAnnotations ++ resetAnnotations
      fullResetAnnotations
    }
  }

  private def getFirstAnnotationIndexWithWrongIndexValues(tokenizerAnnotations: Seq[Annotation],
                                                          normalizerAnnotations: Seq[Annotation]): Int = {
   val wrongIndex = normalizerAnnotations.zipWithIndex.flatMap { case (normalizer, index) =>
      if (normalizer.begin != tokenizerAnnotations(index).begin) Some(index) else None
    }
   if (wrongIndex.isEmpty) -1 else wrongIndex.head
  }

  private def getOffset(tokenizerAnnotations: Seq[Annotation], normalizerAnnotations: Seq[Annotation], wrongIndex: Int):
  Int = {
    if (wrongIndex > 0) {
      val resultOffset = tokenizerAnnotations(wrongIndex - 1).result.length - normalizerAnnotations(wrongIndex - 1).result.length
      var removedNewLinesOffset = 0
      if (tokenizerAnnotations(wrongIndex).result.toLowerCase != normalizerAnnotations(wrongIndex).result.toLowerCase) {
        removedNewLinesOffset = tokenizerAnnotations.size - normalizerAnnotations.size
      }
      resultOffset + removedNewLinesOffset
    } else {
      normalizerAnnotations.head.begin
    }
  }

}

object NormalizerModel extends ParamsAndFeaturesReadable[NormalizerModel]