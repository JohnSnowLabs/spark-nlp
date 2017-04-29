package com.jsl.nlp.annotators

import java.io.{FileInputStream, InputStream}

import com.jsl.nlp.{Document, Annotation, Annotator}
import org.apache.spark.ml.param.Param

/**
  * Created by alext on 10/23/16.
  */
class EntityExtractor(fromSentences: Boolean = false) extends Annotator {

  val maxLen: Param[Int] = new Param(this, "maxLen", "maximum phrase length")

  override val aType: String = "entity"

  override def annotate(
                         document: Document, annotations: Seq[Annotation]
  ): Seq[Annotation] =
    if (fromSentences) {
      annotations.filter {
        token: Annotation => token.aType == "sentence"
      }.flatMap {
        sentence =>
          val ntokens = annotations.filter {
            token: Annotation =>
              token.aType == "ntoken" &&
                token.begin >= sentence.begin &&
                token.end <= sentence.end
          }
          EntityExtractor.phraseMatch(ntokens, $(maxLen), $(entities))
      }
    } else {
      val nTokens = annotations.filter {
        token: Annotation => token.aType == "ntoken"
      }
      EntityExtractor.phraseMatch(nTokens, $(maxLen), $(entities))
    }

  override val requiredAnnotationTypes: Seq[String] =
    if (fromSentences) {
      Seq("sentence")
    } else {
      Seq()
    }

  val entities: Param[Set[Seq[String]]] = new Param(this, "entities", "set of entities (phrases)")

  def setEntities(value: Set[Seq[String]]): this.type = set(entities, value)

  def getEntities: Set[Seq[String]] = $(entities)

  def setMaxLen(value: Int): this.type = set(maxLen, value)

  def getMaxLen: Int = $(maxLen)

}

object EntityExtractor {
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
        nTokens.map(_.metadata("ntoken")).toList
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
            entities.contains(phraseCandidate.map(_.metadata("ntoken")))
        }.map {
          phrase =>
            Annotation(
              "entity",
              phrase.head.begin,
              phrase.last.end,
              Map("entity" -> phrase.map(_.metadata("ntoken")).mkString(" "))
            )
        }
    }.toSeq
  }
}