package com.jsl.nlp.annotators

import java.io.FileNotFoundException

import com.jsl.nlp.annotators.Lemmatizer.TargetWord
import com.jsl.nlp.util.ResourceHelper
import com.jsl.nlp.{Annotation, Annotator, Document}
import com.typesafe.config.{Config, ConfigFactory}

import scala.io.Source
import scala.util.matching.Regex

/**
  * Created by saif on 28/04/17.
  */
class Lemmatizer extends Annotator {

  override val aType: String = Lemmatizer.aType

  override val requiredAnnotationTypes: Seq[String] = Seq(Normalizer.aType)

  /**
    * Would need to verify this implementation, as I am flattening multiple to one annotations
    * @param document
    * @param annotations
    * @return
    */
  override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.collect {
      case token: Annotation if token.aType == Normalizer.aType =>
        val subtext = token.metadata.getOrElse(
          Normalizer.token,
          throw new IllegalArgumentException(
            s"Annotation of type ${Normalizer.aType} does not provide proper token in metadata"
          )
        )
        val targetWords: Array[TargetWord] = Lemmatizer.getWords(subtext)
        targetWords.map(target => Annotation(
          aType,
          target.begin,
          target.end,
          Map(aType -> Lemmatizer.lemmatize(target.text))
        ))
    }.flatten
  }

}

object Lemmatizer {

  private case class TargetWord(text: String, begin: Int, end: Int)

  /**
    * Lemma Dictionary Structure
    * No really need for a wrapper class yet
    */
  private type LemmaDictionary = Map[String, String]

  /**
    * Lemma dictionary in memory
    * Execution pushed to the first time it is needed
    * POTENTIAL candidate for sc.broadcast
    */
  private lazy val lemmaDict: LemmaDictionary = loadLemmaDict

  val aType = "lemma"

  /**
    * Probably could use a ConfigHelper object
    */
  private val config: Config = ConfigFactory.load

  private def getWords(text: String): Array[TargetWord] = {
    val regex: Regex = "\\w+".r
    regex.findAllMatchIn(text).map(word => {
      TargetWord(word.matched, word.start, word.end)
    }).toArray
  }

  private def loadLemmaDict: Map[String, String] = {
    val lemmaFilePath = config.getString("nlp.lemmaDict.file")
    val lemmaFormat = config.getString("nlp.lemmaDict.format")
    val lemmaKeySep = config.getString("nlp.lemmaDict.kvSeparator")
    val lemmaValSep = config.getString("nlp.lemmaDict.vSeparator")
    val lemmaDict = ResourceHelper.flattenValuesAsKeys(lemmaFilePath, lemmaFormat, lemmaKeySep, lemmaValSep)
    lemmaDict
  }

  private def lemmatize(target: String): String = {
    lemmaDict.getOrElse(target, target)
  }

}
