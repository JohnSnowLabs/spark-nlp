package com.jsl.nlp.util.io

import java.io.InputStream

import com.jsl.nlp.annotators.common.{TaggedSentence, TaggedWord}
import com.jsl.nlp.util.regex.RegexRule
import com.typesafe.config.{Config, ConfigFactory}

import scala.collection.mutable.{ArrayBuffer, Map => MMap}
import scala.io.Source

/**
  * Created by saif on 28/04/17.
  */

/**
  * Helper one-place for IO management. Streams, source and external input should be handled from here
  */
object ResourceHelper {

  /** uses config tunning parameters */
  private val config: Config = ConfigFactory.load

  /** Structure for a SourceStream coming from compiled content */
  private case class SourceStream(resource: String) {
    val pipe: InputStream =
      getClass.getResourceAsStream(resource)
    val content: Source = Source.fromInputStream(pipe)("UTF-8")
  }

  /**Standard splitter for general purpose sentences*/
  private def wordTagSplitter(sentence: String, tagSeparator: Char):
  Array[TaggedWord] = {
    val taggedWords: ArrayBuffer[TaggedWord] = ArrayBuffer()
      sentence.split("\\s+").foreach { token => {
        val tagSplit: Array[String] = token.split('|').filter(_.nonEmpty)
        if (tagSplit.length == 2) {
          val word = tagSplit(0)
          val tag = tagSplit(1)
          taggedWords.append(TaggedWord(word, tag))
        }
      }}
    taggedWords.toArray
  }

  /** ToDo: Place holder for defining a set of input rules for RegexMatcher */
  def loadRules: Array[RegexRule] = {
    ???
  }

  /**
    * General purpose key values parser from source
    * Currently only text files
    * @param source File input to streamline
    * @param format format, for now only txt
    * @param keySep separator character
    * @param valueSep values separator in dictionary
    * @return Dictionary of all values per key
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
    * General purpose key value parser from source
    * Currently read only text files
    * @param source File input to streamline
    * @param format format
    * @param keySep separator of keys, values taken as single
    * @return
    */
  def parseKeyValueText(
                          source: String,
                          format: String,
                          keySep: String
                        ): Map[String, String] = {
    format match {
      case "txt" =>
        val sourceStream = SourceStream(source)
        val res = sourceStream.content.getLines.map (line => {
          val kv = line.split (keySep).map (_.trim)
          (kv.head, kv.last)
        }).toMap
        sourceStream.pipe.close()
        res
    }
  }

  /**
    * For multiple values per keys, this optimizer flattens all values for keys to have constant access
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
      case "txt" =>
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
      case _ => throw new IllegalArgumentException("Only txt supported as a file format")
    }
  }

  /**
    * Parses CORPUS for tagged sentences
    * @param text String to process
    * @param tagSeparator Separator for provided String
    * @return A list of [[TaggedSentence]]
    */
  def parsePOSCorpusFromText(
                              text: String,
                              tagSeparator: Char
                            ): Array[TaggedSentence] = {
    val sentences: ArrayBuffer[Array[TaggedWord]] = ArrayBuffer()
    text.split("\n").filter(_.nonEmpty).foreach{sentence =>
      sentences.append(wordTagSplitter(sentence, tagSeparator))
    }
    sentences.map(TaggedSentence).toArray
  }

  /**
    * Parses CORPUS for tagged sentence from any compiled source
    * @param source for compiled corpuses, if any
    * @param tagSeparator Tag separator for processing
    * @return
    */
  def parsePOSCorpusFromSource(
                  source: String,
                  tagSeparator: Char
                ): Array[TaggedSentence] = {
    val sourceStream = SourceStream(source)
    val lines = try {
      sourceStream.content.getLines()
        .filter(_.nonEmpty)
        .map(sentence => wordTagSplitter(sentence, tagSeparator))
        .toArray
    } catch {
      case _: java.nio.charset.UnmappableCharacterException =>
        throw new Exception(s"file $source contains dirty characters or is not UTF-8")
    }
    sourceStream.pipe.close()
    lines.map(TaggedSentence)
  }

  /**
    * Reads POS Corpus from an entire directory of compiled sources
    * @param source compiled content only
    * @param tagSeparator tag separator for all corpuses
    * @param fileLimit limit of files to read. Can help clutter, overfitting
    * @return
    */
  def parsePOSCorpusFromDir(
                           source: String,
                           tagSeparator: Char,
                           fileLimit: Int
                           ): Array[TaggedSentence] = {
    Source.fromInputStream(getClass.getResourceAsStream(source))
      .getLines.take(fileLimit)
      .flatMap(fileName => parsePOSCorpusFromSource(source + "/" + fileName, tagSeparator))
      .toArray
  }

  /**
    * Retrieves Corpuses from configured compiled directory set in configuration
    * @param fileLimit files limit to read
    * @return TaggedSentences for POS training
    */
  def retrievePOSCorpus(fileLimit: Int = 50): Array[TaggedSentence] = {
    val posDirPath = config.getString("nlp.posDict.dir")
    //ToDo support multiple formats in corpus source
    val posFormat = config.getString("nlp.posDict.format")
    val posSeparator = config.getString("nlp.posDict.separator")
    parsePOSCorpusFromDir(posDirPath, posSeparator.head, fileLimit)
  }

  /**
    * Retrieves Lemma dictionary from configured compiled source set in configuration
    * @return a Dictionary for lemmas
    */
  def retrieveLemmaDict: Map[String, String] = {
    val lemmaFilePath = config.getString("nlp.lemmaDict.file")
    val lemmaFormat = config.getString("nlp.lemmaDict.format")
    val lemmaKeySep = config.getString("nlp.lemmaDict.kvSeparator")
    val lemmaValSep = config.getString("nlp.lemmaDict.vSeparator")
    val lemmaDict = ResourceHelper.flattenRevertValuesAsKeys(lemmaFilePath, lemmaFormat, lemmaKeySep, lemmaValSep)
    lemmaDict
  }

  /**
    * Sentiment dictionaries from compiled sources set in configuration
    * @return Sentiment dictionary
    */
  def retrieveSentimentDict: Map[String, String] = {
    val sentFilePath = config.getString("nlp.sentimentDict.file")
    val sentFormat = config.getString("nlp.sentimentDict.format")
    val sentSeparator = config.getString("nlp.sentimentDict.separator")
    ResourceHelper.parseKeyValueText(sentFilePath, sentFormat, sentSeparator)
  }

}
