package com.johnsnowlabs.nlp.annotators.anonymizer

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.common.{IndexedToken, SentenceSplit, TokenizedWithSentence}
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

import scala.collection.mutable.ListBuffer

class DeIdentificationModel(override val uid: String) extends AnnotatorModel[DeIdentificationModel]{

  def this() = this(Identifiable.randomUID("DE-IDENTIFICATION"))

  override val annotatorType: AnnotatorType = DOCUMENT
  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT, TOKEN, CHUNK)

  // val regexPatternsDictionary: MapFeature[String, Array[String]] = new MapFeature(this, "regexPatternsDictionary")

  val regexPatternsDictionary: StructFeature[RegexPatternsDictionary] =
    new StructFeature[RegexPatternsDictionary](this, "regexPatternsDictionary")

  //def setRegexPatternsDictionary(value: Map[String, Array[String]]): this.type = set(regexPatternsDictionary, value)

  def setRegexPatternsDictionary(value: RegexPatternsDictionary): this.type = set(regexPatternsDictionary, value)

  //def getRegexPatternsDictionary: Map[String, Array[String]] = $$(regexPatternsDictionary)
  def getRegexPatternsDictionary: RegexPatternsDictionary = $$(regexPatternsDictionary)

  private lazy val regexDictionary = getRegexPatternsDictionary

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

    //if (regexDictionary.isEmpty){
    if (regexDictionary.dictionary.isEmpty){
      return Seq()
    }

    for ((entity, regexPatterns) <- regexDictionary.dictionary){
      tokensSentence.foreach{tokenSentence =>
         if (isRegexMatch(tokenSentence.token, regexPatterns)){

           val regexEntity = Annotation(CHUNK, tokenSentence.begin, tokenSentence.end,
             tokenSentence.token, Map("entity"->entity))
           regexEntities += regexEntity
         }

      }
    }
    regexEntities.toList
  }

  def isRegexMatch(token: String, regexPatterns: Array[String]): Boolean ={

    val matches = regexPatterns.flatMap(regexPattern => regexPattern.r.findFirstMatchIn(token))
    if (matches.nonEmpty){
      val realMatch = matches.filter(fullMatch => fullMatch.group(0).length == token.length)
      realMatch.nonEmpty
    } else {
      false
    }
  }

  def mergeEntities(nerEntities: Seq[Annotation], regexEntities: Seq[Annotation]): Seq[Annotation] = {

    val cleanEntities = handleEntitiesDifferences(nerEntities, regexEntities)

    val duplicatedEntities = regexEntities.flatMap(regexEntity => cleanEntities.
      filter(nerEntity => nerEntity.equals(regexEntity)))

    val cleanRegexEntities = regexEntities diff duplicatedEntities

    nerEntities ++ cleanRegexEntities
  }

  def handleEntitiesDifferences(nerEntities: Seq[Annotation], regexEntities: Seq[Annotation]): Seq[Annotation] = {

    val differedEntities = regexEntities.flatMap(regexEntity => nerEntities.
      filter(nerEntity => (nerEntity.result == regexEntity.result) &&
                          (nerEntity.metadata("entity") != regexEntity.metadata("entity"))))

    val cleanNerEntities = nerEntities diff differedEntities

    cleanNerEntities
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
