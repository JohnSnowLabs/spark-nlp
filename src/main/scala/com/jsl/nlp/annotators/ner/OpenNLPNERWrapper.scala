package com.jsl.nlp.annotators.ner

import java.util.concurrent.ConcurrentHashMap
import java.util.function.{Function => JFunction}

import opennlp.tools.namefind.{NameFinderME, TokenNameFinderModel}
import opennlp.tools.util.Span

/**
  * Created by alext on 6/23/17.
  */
object OpenNLPNERWrapper {

  private val finders: ConcurrentHashMap[String, NameFinderME] =
    new ConcurrentHashMap[String, NameFinderME]()

  private def getFinder(modelType: String): NameFinderME = synchronized {
    finders.computeIfAbsent(modelType, new JFunction[String, NameFinderME] {
      def apply(s: String): NameFinderME = {
        val model = new TokenNameFinderModel(getClass.getResourceAsStream("/en-ner-" + modelType + ".bin"))
        new NameFinderME(model)
      }
    })
  }

  def ner(sentence: IndexedSeq[String], modelType: String): IndexedSeq[(String, Span)] = {
    val finder = getFinder(modelType)
    val spans = finder.find(sentence.toArray)
    spans.map {
      span => (modelType, span)
    }
  }
}
