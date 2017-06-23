package com.jsl.nlp.annotators.ner

import com.jsl.nlp.annotators.RegexTokenizer
import com.jsl.nlp.annotators.sbd.SentenceDetector
import com.jsl.nlp.{Annotation, Annotator, Document}
import opennlp.tools.util
import org.apache.spark.ml.param.Param

/**
  * Created by alext on 6/14/17.
  */
class NERTagger() extends Annotator {
  /**
    * This is the annotation type
    */
  override protected val aType: String = NERTagger.aType
  /**
    * This is the annotation types that this annotator expects to be present
    */
  override protected val requiredAnnotationTypes: Array[String] = Array(SentenceDetector.aType, RegexTokenizer.aType)

  /**
    * This takes a document and annotations and produces new annotations of this annotator's annotation type
    *
    * @return
    */
  override protected def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = annotations.filter(anno => anno.aType == SentenceDetector.aType)
    sentences.flatMap {
      sentence =>
        val tokens = annotations.filter {
          token: Annotation =>
            token.aType == RegexTokenizer.aType &&
              token.begin >= sentence.begin &&
              token.end <= sentence.end
        }.toIndexedSeq
        val tokenText = tokens.map(t => document.text.substring(t.begin, t.end))
        val tags: IndexedSeq[(String, util.Span)] = OpenNLPNERWrapper.ner(tokenText, $(language))
        tags.map {
          case (tag, span) =>
            Annotation(aType, tokens(span.getStart).begin, tokens(span.getEnd - 1).end, Map(aType -> tag)  )
        }
    }
  }

  val language: Param[String] = new Param(this, "language", "this is the language of the text")

  def setLanguage(value: String): NERTagger = set(language, value)

  def getLanguage: String = $(language)

  setDefault(language, "person")
}

object NERTagger {
  val aType = "named_entity"
}