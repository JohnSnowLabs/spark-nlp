package com.jsl.nlp.annotators

import java.io.{FileInputStream, InputStream}

import com.jsl.nlp.annotators.sbd.SentenceDetector
import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.param.Param
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}

/**
  * Created by alext on 10/23/16.
  */
class EntityExtractor(override val uid: String) extends Annotator {

  val maxLen: Param[Int] = new Param(this, "maxLen", "maximum phrase length")

  val requireSentences: Param[Boolean] = new Param(this, "require sentences", "whether to require sentence boundaries or simple tokens")

  val entities: Param[Set[Seq[String]]] = new Param(this, "entities", "set of entities (phrases)")

  override val aType: String = EntityExtractor.aType

  override var requiredAnnotationTypes: Array[String] = Array()

  def this() = this(Identifiable.randomUID(EntityExtractor.aType))

  def getRequireSentences: Boolean = get(requireSentences).getOrElse(false)

  def setRequireSentences(value: Boolean): this.type = {
    if (value) requiredAnnotationTypes = Array(SentenceDetector.aType)
    set(requireSentences, value)
  }

  def setEntities(value: Set[Seq[String]]): this.type = set(entities, value)

  def getEntities: Set[Seq[String]] = $(entities)

  def setMaxLen(value: Int): this.type = set(maxLen, value)

  def getMaxLen: Int = $(maxLen)

  override def annotate(
                         document: Document, annotations: Seq[Annotation]
                       ): Seq[Annotation] =
    if (getRequireSentences) {
      annotations.filter {
        token: Annotation => token.aType == SentenceDetector.aType
      }.flatMap {
        sentence =>
          val ntokens = annotations.filter {
            token: Annotation =>
              token.aType == Normalizer.aType &&
                token.begin >= sentence.begin &&
                token.end <= sentence.end
          }
          EntityExtractor.phraseMatch(ntokens, $(maxLen), $(entities))
      }
    } else {
      val nTokens = annotations.filter {
        token: Annotation => token.aType == Normalizer.aType
      }
      EntityExtractor.phraseMatch(nTokens, $(maxLen), $(entities))
    }

}

object EntityExtractor extends DefaultParamsReadable[EntityExtractor] {

  val aType = "entity"

  def loadEntities(inputStream: InputStream, tokenPattern: String): Set[Seq[String]] = {
    val src = scala.io.Source.fromInputStream(inputStream)
    val tokenizer = new RegexTokenizer().setPattern(tokenPattern)
    val stemmer = new Stemmer()
    val normalizer = new Normalizer()
    val lemmatizer = new Lemmatizer()
    val phrases: Set[Seq[String]] = src.getLines.map {
      line =>
        val doc = Document("", line)
        val tokens = tokenizer.annotate(doc, Seq())
        val stems = stemmer.annotate(doc, tokens)
        val nTokens = normalizer.annotate(doc, stems)
        val lemmas = lemmatizer.annotate(doc, nTokens)
        nTokens.map(_.metadata(Normalizer.aType)).toList
    }.toSet
    src.close()
    phrases
  }

  def loadEntities(path: String, tokenPattern: String): Set[Seq[String]] = {
    loadEntities(new FileInputStream(path), tokenPattern)
  }

  def phraseMatch(nTokens: Seq[Annotation], maxLen: Int, entities: Set[Seq[String]]): Seq[Annotation] = {
    nTokens.padTo(nTokens.length + maxLen - (nTokens.length % maxLen), null).sliding(maxLen).flatMap {
      window =>
        window.filter(_ != null).inits.filter {
          phraseCandidate =>
            entities.contains(phraseCandidate.map(_.metadata(Normalizer.aType)))
        }.map {
          phrase =>
            Annotation(
              "entity",
              phrase.head.begin,
              phrase.last.end,
              Map("entity" -> phrase.map(_.metadata(Normalizer.aType)).mkString(" "))
            )
        }
    }.toSeq
  }
}