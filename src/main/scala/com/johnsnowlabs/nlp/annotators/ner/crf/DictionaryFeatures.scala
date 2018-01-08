package com.johnsnowlabs.nlp.annotators.ner.crf

import com.johnsnowlabs.nlp.util.io.ResourceHelper

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

  def read(paths: Seq[String]): DictionaryFeatures = {
    val items = paths.flatMap{path => read(path)}
    DictionaryFeatures(items)
  }

  private def read(path: String): Iterator[(String, String)] = {
    ResourceHelper.SourceStream(path)
      .content.getLines().map{line =>
        val items = line.split(":")
        require(items.size == 2)
        (items(0), items(1))
    }
  }
}