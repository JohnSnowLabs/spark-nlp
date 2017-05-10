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
        val targetToken = token.metadata.getOrElse(
          Normalizer.token,
          throw new IllegalArgumentException(
            s"Annotation of type ${Normalizer.aType} does not provide proper token in metadata"
          )
        )
        Annotation(
          aType,
          token.begin,
          token.end,
          Map(aType -> Lemmatizer.lemmatize(targetToken))
        )
    }
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
