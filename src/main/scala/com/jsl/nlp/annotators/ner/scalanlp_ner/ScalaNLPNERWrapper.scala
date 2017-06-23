package com.jsl.nlp.annotators.ner.scalanlp_ner

import java.util.concurrent.ConcurrentHashMap
import java.util.function.{Function => JFunction}

import epic.models.NerSelector
import epic.sequences.SemiCRF


/**
  * Created by alext on 6/14/17.
  */
object ScalaNLPNERWrapper {
  type NERModel = SemiCRF[Any, String]

  private val models: ConcurrentHashMap[String, NERModel] = new ConcurrentHashMap[String, NERModel]()

  private def getModel(language: String): NERModel = synchronized {
    models.computeIfAbsent(language, new JFunction[String, NERModel] {
      def apply(s: String): NERModel = NerSelector.loadNer(language) match {
        case Some(model) => model
        case None => throw new IllegalArgumentException(s"Language [$language] not found")
      }
    })
  }

  def ner(sentence: IndexedSeq[String], language: String = "en"): IndexedSeq[(String, SimpleSpan)] = {
    val model = getModel(language)
    val segments = model.bestSequence(sentence)
    segments.segments.map {
      case (tag, span) => (tag.toString, SimpleSpan(span.begin, span.end))
    }
  }
}

case class SimpleSpan(begin: Int, end: Int)
