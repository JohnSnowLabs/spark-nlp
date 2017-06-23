package com.jsl.nlp.clinical.negex

import com.jsl.nlp.annotators.clinical.negex.GenNegEx
import com.jsl.nlp.annotators.sbd.SentenceDetector
import com.jsl.nlp.annotators.sbd.pragmatic.PragmaticApproach
import com.jsl.nlp.{Annotation, Annotator, Document}

import scala.collection.mutable.ArrayBuffer
import scala.util.matching.Regex

/**
  * Created by alext on 6/21/17.
  */
class NegexTagger() extends Annotator {
  /**
    * This is the annotation type
    */
  override val aType: String = NegexTagger.aType
  /**
    * This is the annotation types that this annotator expects to be present
    */
  override val requiredAnnotationTypes: Array[String] = Array("sbd")

  /**
    * This takes a document and annotations and produces new annotations of this annotator's annotation type
    *
    * @return
    */
  override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.flatMap {
      sentence =>
        val sentText = sentence.metadata(SentenceDetector.aType)
        val negChecked = GenNegEx.negCheck(sentText)
        var prevEnd = 0
        var offset = 0
        var prevState = ""
        val buffer = ArrayBuffer[Annotation]()
        NegexTagger.negTagTypes.findAllMatchIn(negChecked).foreach {
          m =>
            offset += NegexTagger.tagLen
            val currState = m.group(0)
            if (currState != prevState) {
              prevState match {
                case "[pren]" | "[prep]" =>
                  currState match {
                    case "[pren]" =>
                      buffer.append(Annotation(NegexTagger.aType, prevEnd, m.start - offset + NegexTagger.tagLen))
                    case "[prep]" =>
                      buffer.append(Annotation(NegexTagger.aType, prevEnd, m.start - offset + NegexTagger.tagLen))
                    case "[post]" =>
                    case "[posp]" =>
                    case "[conj]" =>
                      buffer.append(Annotation(NegexTagger.aType, prevEnd, m.start - offset + NegexTagger.tagLen))
                    case "[pseu]" =>
                      buffer.append(Annotation(NegexTagger.aType, prevEnd, m.start - offset + NegexTagger.tagLen))
                    case "" =>
                  }
                case "[conj]" | "[pseu]" | "" =>
                  currState match {
                    case "[pren]" =>
                    case "[prep]" =>
                    case "[post]" =>
                      buffer.append(Annotation(NegexTagger.aType, prevEnd, m.start - offset + NegexTagger.tagLen))
                    case "[posp]" =>
                      buffer.append(Annotation(NegexTagger.aType, prevEnd, m.start - offset + NegexTagger.tagLen))
                    case "[conj]" =>
                    case "[pseu]" =>
                    case "" =>
                  }
                case "[post]" | "[posp]" =>
              }
            } else {
              prevEnd = m.end - offset
            }
            prevState = currState
        }
        buffer
    }
  }
}

object NegexTagger {
  private val tagLen: Int = 6

  val aType: String = "negex"

  val negTagTypes: Regex = Array("pren", "post", "prep", "posp", "conj", "pseu")
    .map(tt => s"\\[$tt\\]").mkString("|").r
}