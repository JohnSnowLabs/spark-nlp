package com.jsl.nlp.annotators.ner.scalanlp_ner

import java.util.concurrent.ConcurrentHashMap
import java.util.function.{Function => JFunction}

import epic.models.NerSelector
import epic.sequences.SemiCRF
import epic.trees.Span


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

  def ner(sentence: IndexedSeq[String], language: String = "en"): IndexedSeq[(String, Span)] = {
    val model = getModel(language)
    val segments = model.bestSequence(sentence)
    segments.segments.asInstanceOf[IndexedSeq[(String, Span)]]
  }
}
