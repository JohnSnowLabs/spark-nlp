package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, NAMED_ENTITY}
import com.johnsnowlabs.nlp.annotators.common.{SentenceSplit, TokenizedWithSentence}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

class DeIdentification(override val uid: String) extends  AnnotatorModel[DeIdentification]{

  override val annotatorType: AnnotatorType = DOCUMENT
  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT, DOCUMENT)

  def this() = this(Identifiable.randomUID("DE-IDENTIFICATION"))

  def getProtectedEntities(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.filter(annotation => annotation.result != "O")
  }

  def getAnonymizeSentence(sentence: String, protectedEntities: Seq[Annotation]): String = {
    var anonymizeSentence = sentence
    protectedEntities.foreach(annotation => {
      val wordToReplace = annotation.metadata("word")
      val replacement = annotation.result.substring(2, annotation.result.length)
      anonymizeSentence = anonymizeSentence.replaceFirst(wordToReplace, replacement)
    })
    anonymizeSentence
  }

  def createAnonymizeAnnotation(annotations: Seq[Annotation], anonymizeSentence: String): Annotation = {
    val protectedEntities = getProtectedEntities(annotations)
    val sentence = annotations.map(annotation => annotation.metadata("word")).toList.mkString(" ")
    val anonymizeSentence = getAnonymizeSentence(sentence, protectedEntities)
    Annotation(annotatorType, 0, anonymizeSentence.length, anonymizeSentence, Map("sentence" -> "protected"))
  }


  def getSentence(annotations: Seq[Annotation]): String = {
    //annotations(2).result
    val sentenceAnnotations = annotations.filter(annotation => annotation.metadata.isEmpty).map(_.)
    //  sentenceAnnotation.map(sentence => sentence.result)

  }

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = SentenceSplit.unpack(Seq(annotations.head))

    //val anonymizeSentences = sentences.map(sentence => Seq(createAnonymizeAnnotation(annotations, sentence.content)))
    //anonymizeSentences
    // val protectedEntities = getProtectedEntities(annotations)
    //val sentence = annotations.map(annotation => annotation.metadata("word")).mkString(" ")
    //val anonymizeSentence = getAnonymizeSentence(sentence, protectedEntities)
    //Seq(createAnonymizeAnnotation(annotations, anonymizeSentence))
    annotations
  }

}

object DeIdentification extends DefaultParamsReadable[DeIdentification]
