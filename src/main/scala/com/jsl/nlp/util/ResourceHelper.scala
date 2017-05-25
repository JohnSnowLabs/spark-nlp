package com.jsl.nlp.util

import java.io.{FileNotFoundException, InputStream}

import scala.io.Source
import scala.collection.mutable.{ListBuffer, Map => MMap}

/**
  * Created by saif on 28/04/17.
  */
object ResourceHelper {

  private type TaggedSentences = List[(List[String], List[String])]

  private case class SourceStream(resource: String) {
    val pipe: InputStream = try {
      getClass.getResourceAsStream("/" + resource)
    } catch {
      case _: FileNotFoundException =>
        throw new FileNotFoundException(s"Lemma dictionary $resource not found")
    }
    val content: Source = Source.fromInputStream(pipe)
  }

  private def wordTagSplitter(sentence: String, tagSeparator: String):
  (List[String], List[String]) = {
    val tokens: ListBuffer[String] = ListBuffer()
    val tags: ListBuffer[String] = ListBuffer()
    sentence.split("\\s+").foreach{token => {
      val tagSplit: Array[String] = token.split(tagSeparator)
      val word = tagSplit(0)
      val pos = tagSplit(1)
      tokens.append(word)
      tags.append(pos)
    }}
    (tokens.toList, tags.toList)
  }

  /**
    * Standard key value parser from source
    *
    * @param source File input to streamline
    * @param format format, for now only txt
    * @param keySep separator character
    * @param valueSep values separator in dictionary
    * @return
    */
  def parseKeyValuesText(
                         source: String,
                         format: String,
                         keySep: String,
                         valueSep: String): Map[String, Array[String]] = {
    format match {
      case "txt" =>
        val sourceStream = SourceStream(source)
        val res = sourceStream.content.getLines.map (line => {
          val kv = line.split (keySep).map (_.trim)
          val key = kv (0)
          val values = kv (1).split (valueSep).map (_.trim)
          (key, values)
        }).toMap
        sourceStream.pipe.close()
        res
    }
  }

  /**
    * Specific approach chosen is to generate quick reads of Lemma Dictionary
    *
    * @param source File input to streamline
    * @param format format, for now only txt
    * @param keySep separator cha
    * @param valueSep values separator in dictionary
    * @return
    */
  def flattenRevertValuesAsKeys(
                                 source: String,
                                 format: String,
                                 keySep: String,
                                 valueSep: String): Map[String, String] = {
    format match {
      case "txt" => {
        val m: MMap[String, String] = MMap()
        val sourceStream = SourceStream(source)
        sourceStream.content.getLines.foreach(line => {
          val kv = line.split(keySep).map(_.trim)
          val key = kv(0)
          val values = kv(1).split(valueSep).map(_.trim)
          values.foreach(m(_) = key)
        })
        sourceStream.pipe.close()
        m.toMap
      }
      case _ => throw new IllegalArgumentException("Only txt supported as a file format")
    }
  }

  def parsePOSCorpusFromText(
                              text: String,
                              tagSeparator: String
                            ): TaggedSentences = {
    val sentences: ListBuffer[(List[String], List[String])] = ListBuffer()
    text.split("\n").foreach{sentence =>
      sentences.append(wordTagSplitter(sentence, tagSeparator))
    }
    sentences.toList
  }

  def parsePOSCorpusFromSource(
                  source: String,
                  tagSeparator: String
                ): TaggedSentences = {
    val sourceStream = SourceStream(source)
    sourceStream.content.getLines()
      .filter(_.nonEmpty)
      .map(sentence => wordTagSplitter(sentence, tagSeparator))
      .toList
  }

  def parsePOSCorpusFromSources(sources: List[String], tagSeparator: String): TaggedSentences = {
    sources.flatMap(parsePOSCorpusFromSource(_, tagSeparator))
  }

}
