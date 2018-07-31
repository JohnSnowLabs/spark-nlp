package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT}
import com.johnsnowlabs.nlp.annotators.common.{SentenceSplit, TokenizedWithSentence}
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

class DeIdentificationModel(override val uid: String) extends  AnnotatorModel[DeIdentificationModel]{

  override val annotatorType: AnnotatorType = DOCUMENT
  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT, CHUNK)

  val regexPatternsDictionary: MapFeature[String, String] = new MapFeature(this, "regexPatternsDictionary")

  def this() = this(Identifiable.randomUID("DE-IDENTIFICATION"))

  def setRegexPatternsDictionary(value: Map[String, String]): this.type = set(regexPatternsDictionary, value)

  def getSentence(annotations: Seq[Annotation]): String = {
    val sentences = SentenceSplit.unpack(annotations)
    val sentence = sentences.map(sentence => sentence.content)
    sentence.mkString(" ")
  }

  def getProtectedEntities(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.filter(annotation => annotation.annotatorType == CHUNK)
  }
  //http://www.baeldung.com/java-date-regular-expressions
  def getAnonymizeSentence(sentence: String, protectedEntities: Seq[Annotation]): String = {
    var anonymizeSentence = sentence
    protectedEntities.foreach(annotation => {
      val wordToReplace = annotation.result
      val replacement = annotation.metadata("entity")
      anonymizeSentence = anonymizeSentence.replaceFirst(wordToReplace, replacement)
    })
    anonymizeSentence
  }

  def createAnonymizeAnnotation(anonymizeSentence: String): Annotation = {
    Annotation(annotatorType, 0, anonymizeSentence.length, anonymizeSentence, Map("sentence" -> "protected"))
  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentence = getSentence(annotations)
    val protectedEntities = getProtectedEntities(annotations)
    val anonymizeSentence = getAnonymizeSentence(sentence, protectedEntities)
    Seq(createAnonymizeAnnotation(anonymizeSentence))
  }

}

object DeIdentificationModel extends DefaultParamsReadable[DeIdentificationModel]
