package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.nlp.util.io.{ExternalResource, ResourceHelper}

case class DictionaryFeatures(dict: Map[String, String])
{
 def get(tokens: Seq[String]): Seq[String] = {
    val lower = new StringBuilder()

    tokens.take(DictionaryFeatures.maxTokens).flatMap{token =>
      if (lower.nonEmpty)
        lower.append(" ")

      lower.append(token.toLowerCase)
      dict.get(lower.toString)
    }
  }
}

object DictionaryFeatures {
  val maxTokens = 5

  def apply(text2Feature: Seq[(String, String)]) = {
    val dict = text2Feature.map(p => (p._1.replace("-", " ").trim.toLowerCase, p._2)).toMap
    new DictionaryFeatures(dict)
  }

  def read(possibleEr: Option[ExternalResource]): DictionaryFeatures = {
    possibleEr.map(er => DictionaryFeatures(ResourceHelper.parseTupleText(er)))
      .getOrElse(new DictionaryFeatures(Map.empty[String, String]))
  }
}