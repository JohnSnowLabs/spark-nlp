package com.jsl.nlp.annotators

import com.jsl.nlp.util.io.ResourceHelper
import com.jsl.nlp._
import com.jsl.nlp.util.ConfigHelper
import com.typesafe.config.Config
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Created by alext on 10/23/16.
  */

/**
  * Extracts entities out of provided phrases
  * @param uid internally required UID to make it writable
  * @@ entities: Unique set of phrases
  * @@ requireSentences: May use sentence boundaries provided by a previous SBD annotator
  * @@ maxLen: Auto limit for phrase lenght
  */
class EntityExtractor(override val uid: String) extends AnnotatorModel[EntityExtractor] {

  import com.jsl.nlp.AnnotatorType._

  val maxLen: IntParam = new IntParam(this, "maxLen", "maximum phrase length")

  val entities: Param[String] = new Param(this, "entities", "set of entities (phrases)")
  private var loadedEntities: Array[Array[String]] = loadEntities

  override val annotatorType: AnnotatorType = ENTITY

  override val requiredAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  setDefault(inputCols, Array(DOCUMENT))

  /** internal constructor for writabale annotator */
  def this() = this(Identifiable.randomUID("ENTITY_EXTRACTOR"))

  def setEntities(value: String): this.type = {
    set(entities, value)
    loadedEntities = loadEntities
    this
  }

  def getEntities: Array[Array[String]] = loadedEntities

  def setMaxLen(value: Int): this.type = set(maxLen, value)

  def getMaxLen: Int = $(maxLen)

  /**
    * Loads entities from a provided source.
    */
  private def loadEntities: Array[Array[String]] = {
    val src = get(entities).map(path => EntityExtractor.retrieveEntityExtractorPhrases(path))
      .getOrElse(EntityExtractor.retrieveEntityExtractorPhrases())
    val tokenizer = new RegexTokenizer().setPattern("\\w+")
    val stemmer = new Stemmer()
    val normalizer = new Normalizer()
    val phrases: Array[Array[String]] = src.map {
      line =>
        val annotation = Seq(Annotation(line))
        val tokens = tokenizer.annotate(annotation)
        val stems = stemmer.annotate(tokens)
        val nTokens = normalizer.annotate(stems)
        nTokens.map(_.metadata(TOKEN)).toArray
    }
    phrases
  }

  /**
    * matches entities depending on utilized annotators and stores them in the annotation
    * @param nTokens pads annotation content to phrase limits
    * @param maxLen applies limit not to exceed results
    * @param entities entities to find within annotators results
    * @return
    */
  private def phraseMatch(nTokens: Seq[Annotation], maxLen: Int, entities: Array[Array[String]]): Seq[Annotation] = {
    nTokens.padTo(nTokens.length + maxLen - (nTokens.length % maxLen), null).sliding(maxLen).flatMap {
      window =>
        window.filter(_ != null).inits.filter {
          phraseCandidate =>
            entities.contains(phraseCandidate.map(_.metadata(TOKEN)).toArray)
        }.map {
          phrase =>
            Annotation(
              "entity",
              phrase.head.begin,
              phrase.last.end,
              Map(annotatorType -> phrase.map(_.metadata(TOKEN)).mkString(" "))
            )
        }
    }.toSeq
  }

  /** Defines annotator phrase matching depending on whether we are using SBD or not */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.filter {
      token: Annotation => token.annotatorType == DOCUMENT
    }.flatMap {
      sentence =>
        val ntokens = annotations.filter {
          token: Annotation =>
            token.annotatorType == TOKEN &&
              token.begin >= sentence.begin &&
              token.end <= sentence.end
        }
        phraseMatch(ntokens, $(maxLen), loadedEntities)
    }
  }

}

object EntityExtractor extends DefaultParamsReadable[EntityExtractor] {

  private val config: Config = ConfigHelper.retrieve

  protected def retrieveEntityExtractorPhrases(
                                      entitiesPath: String = "__default",
                                      fileFormat: String = config.getString("nlp.entityExtractor.format")
                                    ): Array[String] = {
    val filePath = if (entitiesPath == "__default") config.getString("nlp.entityExtractor.file") else entitiesPath
    ResourceHelper.parseLinesText(filePath, fileFormat)
  }


}