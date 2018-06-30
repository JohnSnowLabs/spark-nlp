package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, ParamsAndFeaturesReadable}
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.ml.param.{BooleanParam, StringArrayParam}
import org.apache.spark.ml.util.Identifiable

class NormalizerModel(override val uid: String) extends AnnotatorModel[NormalizerModel] {

  override val annotatorType: AnnotatorType = TOKEN

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)

  val patterns = new StringArrayParam(this, "patterns",
    "normalization regex patterns which match will be replaced with a space")

  val lowercase = new BooleanParam(this, "lowercase", "whether to convert strings to lowercase")

  protected val slangDict: MapFeature[String, String] = new MapFeature(this, "slangDict")

  def this() = this(Identifiable.randomUID("NORMALIZER"))

  def setPatterns(value: Array[String]): this.type = set(patterns, value)

  def setLowercase(value: Boolean): this.type = set(lowercase, value)

  def setSlangDict(value: Map[String, String]): this.type = set(slangDict, value)

  def applyRegexPatterns(word: String): String = {

    val nToken = {
      get(patterns).map(_.foldLeft(word)((currentText, compositeToken) => {
        currentText.replaceAll(compositeToken, "")
      })).getOrElse(word)
    }
    nToken
  }

  def getAnnotation(word: String, token: Annotation, finalWords: Array[String], index: Int): Annotation = {
    if (finalWords.length > 1) {
      Annotation(annotatorType,0,word.length-1,word,Map("sentence"->index.toString))
    } else {
      Annotation(
        annotatorType,
        token.begin,
        token.end,
        word,
        token.metadata)
    }

  }

  protected def getSlangDict: Map[String, String] = $$(slangDict)

  /** ToDo: Review implementation, Current implementation generates spaces between non-words, potentially breaking tokens */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] =

    annotations.flatMap { token =>

      val cased =
        if ($(lowercase)) token.result.toLowerCase
        else token.result

      val correctedWords = $$(slangDict).getOrElse(cased, cased)

      val finalWords = correctedWords.split(" ").map(word => applyRegexPatterns(word))

      val annotations = finalWords.zipWithIndex.map{case (word, index) =>
        getAnnotation(word, token, finalWords, index+1)}

      annotations
    }.filter(_.result.nonEmpty)

}

object NormalizerModel extends ParamsAndFeaturesReadable[NormalizerModel]