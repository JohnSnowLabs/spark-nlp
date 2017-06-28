package com.jsl.nlp.clinical.negex

import com.jsl.nlp.annotators.clinical.negex.GenNegEx
import com.jsl.nlp.annotators.sbd.SentenceDetector
import com.jsl.nlp.{Annotation, Annotator, Document}
import org.apache.spark.ml.util.Identifiable

import scala.collection.mutable.ArrayBuffer
import scala.util.matching.Regex

/**
  * Created by alext on 6/21/17.
  */
class NegexTagger(override val uid: String) extends Annotator {
  /**
    * This is the annotation type
    */
  override val annotatorType: String = NegexTagger.annotatorType
  /**
    * This is the annotation types that this annotator expects to be present
    */
  override protected var requiredAnnotatorTypes: Array[String] = Array(SentenceDetector.annotatorType)

  def this() = this(Identifiable.randomUID(NegexTagger.annotatorType))

  /**
    * This takes a document and annotations and produces new annotations of this annotator's annotation type
    *
    * @return
    */
  override def annotate(document: Document, annotations: Seq[Annotation]): Seq[Annotation] = {
    annotations.flatMap {
      sentence =>
        val sentText = sentence.metadata(SentenceDetector.annotatorType)
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
                      buffer.append(Annotation(NegexTagger.annotatorType, prevEnd, m.start - offset + NegexTagger.tagLen, Map()))
                    case "[prep]" =>
                      buffer.append(Annotation(NegexTagger.annotatorType, prevEnd, m.start - offset + NegexTagger.tagLen, Map()))
                    case "[post]" =>
                    case "[posp]" =>
                    case "[conj]" =>
                      buffer.append(Annotation(NegexTagger.annotatorType, prevEnd, m.start - offset + NegexTagger.tagLen, Map()))
                    case "[pseu]" =>
                      buffer.append(Annotation(NegexTagger.annotatorType, prevEnd, m.start - offset + NegexTagger.tagLen, Map()))
                    case "" =>
                  }
                case "[conj]" | "[pseu]" | "" =>
                  currState match {
                    case "[pren]" =>
                    case "[prep]" =>
                    case "[post]" =>
                      buffer.append(Annotation(NegexTagger.annotatorType, prevEnd, m.start - offset + NegexTagger.tagLen, Map()))
                    case "[posp]" =>
                      buffer.append(Annotation(NegexTagger.annotatorType, prevEnd, m.start - offset + NegexTagger.tagLen, Map()))
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

  val annotatorType: String = "negex"

  val negTagTypes: Regex = Array("pren", "post", "prep", "posp", "conj", "pseu")
    .map(tt => s"\\[$tt\\]").mkString("|").r
}