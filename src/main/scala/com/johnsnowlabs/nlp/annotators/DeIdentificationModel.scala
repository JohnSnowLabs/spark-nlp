package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, SentenceSplit, TokenizedWithSentence}
import com.johnsnowlabs.nlp.serialization.MapFeature
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

import scala.collection.mutable.ListBuffer

class DeIdentificationModel(override val uid: String) extends  AnnotatorModel[DeIdentificationModel]{

  override val annotatorType: AnnotatorType = DOCUMENT
  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT, TOKEN, CHUNK)

  val regexPatternsDictionary: MapFeature[String, List[String]] = new MapFeature(this, "regexPatternsDictionary")

  def this() = this(Identifiable.randomUID("DE-IDENTIFICATION"))

  def setRegexPatternsDictionary(value: Map[String, List[String]]): this.type = set(regexPatternsDictionary, value)

  def getRegexPatternsDictionary: Map[String, List[String]] = $$(regexPatternsDictionary)

  private lazy val dictionary = getRegexPatternsDictionary

  def getSentence(annotations: Seq[Annotation]): String = {
    val sentences = SentenceSplit.unpack(annotations)
    val sentence = sentences.map(sentence => sentence.content)
    sentence.mkString(" ")
  }

  def getTokens(annotations: Seq[Annotation]): Seq[IndexedToken] = {
    TokenizedWithSentence.unpack(annotations).flatMap(tokens => tokens.indexedTokens)
  }

  def getNerEntities(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.filter(annotation => annotation.annotatorType == CHUNK)
  }

  def getRegexEntities(tokensSentence: Seq[IndexedToken]): Seq[Annotation] = {

    var regexEntities = new ListBuffer[Annotation]()

    if (dictionary.isEmpty){
      return Seq()
    }

    for ((entity, regexPatterns) <- dictionary){
      tokensSentence.foreach{tokenSentence =>
         if (isRegexMatch(tokenSentence.token, regexPatterns)){

           val regexEntity = Annotation(CHUNK, tokenSentence.begin, tokenSentence.end,
             tokenSentence.token, Map("regex_entity"->entity))
           regexEntities += regexEntity
         }

      }
    }
    regexEntities.toList
  }

  def isRegexMatch(token: String, regexPatterns: List[String]): Boolean ={
    val matches = regexPatterns.flatMap(regexPattern => regexPattern.r.findFirstMatchIn(token))
    if (matches.nonEmpty){
      val realMatch = matches.filter(fullMatch => fullMatch.group(0).length == token.length)
      realMatch.nonEmpty
    } else {
      false
    }
  }

  def mergeEntities(nerEntities: Seq[Annotation], regexEntities: Seq[Annotation]): Seq[Annotation] = {
    nerEntities ++ regexEntities
  }

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
    val tokens = getTokens(annotations)
    val nerEntities = getNerEntities(annotations)
    val regexEntities = getRegexEntities(tokens)
    val protectedEntities = mergeEntities(nerEntities, regexEntities)
    val anonymizeSentence = getAnonymizeSentence(sentence, protectedEntities)
    Seq(createAnonymizeAnnotation(anonymizeSentence))
  }

}

object DeIdentificationModel extends DefaultParamsReadable[DeIdentificationModel]
