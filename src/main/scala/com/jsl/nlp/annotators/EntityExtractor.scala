package com.jsl.nlp.annotators

import java.io.{FileInputStream, InputStream}

import com.jsl.nlp.annotators.sbd.SentenceDetector
import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.param.Param
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
class EntityExtractor(override val uid: String) extends Annotator {

  val maxLen: Param[Int] = new Param(this, "maxLen", "maximum phrase length")

  val requireSentences: Param[Boolean] = new Param(this, "require sentences", "whether to require sentence boundaries or simple tokens")

  val entities: Param[Set[Seq[String]]] = new Param(this, "entities", "set of entities (phrases)")

  override val annotatorType: String = EntityExtractor.annotatorType

  override var requiredAnnotatorTypes: Array[String] = Array()

  /** internal constructor for writabale annotator */
  def this() = this(Identifiable.randomUID(EntityExtractor.annotatorType))

  def getRequireSentences: Boolean = get(requireSentences).getOrElse(false)

  def setRequireSentences(value: Boolean): this.type = {
    if (value) requiredAnnotatorTypes = Array(SentenceDetector.aType)
    set(requireSentences, value)
  }

  def setEntities(value: Set[Seq[String]]): this.type = set(entities, value)

  def getEntities: Set[Seq[String]] = $(entities)

  def setMaxLen(value: Int): this.type = set(maxLen, value)

  def getMaxLen: Int = $(maxLen)

  /** Defines annotator phrase matching depending on whether we are using SBD or not */
  override def annotate(
                         document: Document, annotations: Seq[Annotation]
                       ): Seq[Annotation] =
    if (getRequireSentences) {
      annotations.filter {
        token: Annotation => token.annotatorType == SentenceDetector.aType
      }.flatMap {
        sentence =>
          val ntokens = annotations.filter {
            token: Annotation =>
              token.annotatorType == Normalizer.annotatorType &&
                token.begin >= sentence.begin &&
                token.end <= sentence.end
          }
          EntityExtractor.phraseMatch(ntokens, $(maxLen), $(entities))
      }
    } else {
      val nTokens = annotations.filter {
        token: Annotation => token.annotatorType == Normalizer.annotatorType
      }
      EntityExtractor.phraseMatch(nTokens, $(maxLen), $(entities))
    }

}

object EntityExtractor extends DefaultParamsReadable[EntityExtractor] {

  val annotatorType = "entity"

  /**
    * Loads entities from a provided source.
    * ToDo: Should use [[com.jsl.nlp.util.io.ResourceHelper]] and get tokenPattern from RegexTokenizer
    * @param tokenPattern overrides tokenizer separator
    */
  def loadEntities(inputStream: InputStream, tokenPattern: String): Set[Seq[String]] = {
    val src = scala.io.Source.fromInputStream(inputStream)
    val tokenizer = new RegexTokenizer().setPattern(tokenPattern)
    val stemmer = new Stemmer()
    val normalizer = new Normalizer()
    val phrases: Set[Seq[String]] = src.getLines.map {
      line =>
        val doc = Document(line)
        val tokens = tokenizer.annotate(doc, Seq())
        val stems = stemmer.annotate(doc, tokens)
        val nTokens = normalizer.annotate(doc, stems)
        nTokens.map(_.metadata(Normalizer.annotatorType)).toList
    }.toSet
    src.close()
    phrases
  }

  def loadEntities(path: String, tokenPattern: String): Set[Seq[String]] = {
    loadEntities(new FileInputStream(path), tokenPattern)
  }

  /**
    * matches entities depending on utilized annotators and stores them in the annotation
    * @param nTokens pads annotation content to phrase limits
    * @param maxLen applies limit not to exceed results
    * @param entities entities to find within annotators results
    * @return
    */
  def phraseMatch(nTokens: Seq[Annotation], maxLen: Int, entities: Set[Seq[String]]): Seq[Annotation] = {
    nTokens.padTo(nTokens.length + maxLen - (nTokens.length % maxLen), null).sliding(maxLen).flatMap {
      window =>
        window.filter(_ != null).inits.filter {
          phraseCandidate =>
            entities.contains(phraseCandidate.map(_.metadata(Normalizer.annotatorType)))
        }.map {
          phrase =>
            Annotation(
              "entity",
              phrase.head.begin,
              phrase.last.end,
              Map("entity" -> phrase.map(_.metadata(Normalizer.annotatorType)).mkString(" "))
            )
        }
    }.toSeq
  }
}