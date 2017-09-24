package com.jsl.nlp.util.io

import java.io.{File, FileNotFoundException, InputStream}

import com.jsl.nlp.annotators.common.{TaggedSentence, TaggedWord}
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
    val pipe: Option[InputStream] = try {
      val touchSource = getClass.getResourceAsStream(resource)
      Source.fromInputStream(touchSource)("UTF-8").getLines().take(1)
      Some(touchSource)
    } catch {
      case _: NullPointerException => None
    }
    val content: Source = pipe.map(p => Source.fromInputStream(p)("UTF-8")).getOrElse(Source.fromFile(resource, "UTF-8"))
    def close(): Unit = {
      content.close()
      pipe.foreach(_.close())
    }
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

  /** Checks wether a path points to directory */
  private def pathIsDirectory(path: String): Boolean = {
    //ToDo: Improve me???
    if (path.contains(".txt")) false else true
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
        sourceStream.close()
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
        sourceStream.close()
        res
    }
  }

  /**
    * General purpose line parser from source
    * Currently read only text files
    * @param source File input to streamline
    * @param format format
    * @return
    */
  def parseLinesText(
                      source: String,
                      format: String
                     ): Array[String] = {
    format match {
      case "txt" =>
        val sourceStream = SourceStream(source)
        val res = sourceStream.content.getLines.toArray
        sourceStream.close()
        res
    }
  }

  /**
    * General purpose tuple parser from source
    * Currently read only text files
    * @param source File input to streamline
    * @param format format
    * @param keySep separator of tuples
    * @return
    */
  def parseTupleText(
                         source: String,
                         format: String,
                         keySep: String
                       ): Array[(String, String)] = {
    format match {
      case "txt" =>
        val sourceStream = SourceStream(source)
        val res = sourceStream.content.getLines.map (line => {
          val kv = line.split (keySep).map (_.trim)
          (kv.head, kv.last)
        }).toArray
        sourceStream.close()
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
        sourceStream.close()
        m.toMap
      case _ => throw new IllegalArgumentException("Only txt supported as a file format")
    }
  }

  def wordCount(
                 source: String,
                 format: String,
                 m: MMap[String, Int] = MMap.empty[String, Int].withDefaultValue(0),
                 clean: Boolean = true,
                 prefix: Option[String] = None,
                 f: Option[(List[String] => List[String])] = None
               ): MMap[String, Int] = {
    format match {
      case "txt" =>
        val regex = if (clean) "[a-zA-Z]+".r else "\\S+".r
        if (pathIsDirectory(source)) {
          try {
            new File(source).listFiles()
              .flatMap(fileName => wordCount(fileName.toString, format, m))
          } catch {
            case _: NullPointerException =>
              val sourceStream = SourceStream(source)
              sourceStream
                .content
                .getLines()
                .flatMap(fileName => wordCount(source + "/" + fileName, format, m))
                .toArray
              sourceStream.close()
          }
        } else {
          val sourceStream = SourceStream(source)
          sourceStream.content.getLines.foreach(line => {
            val words = regex.findAllMatchIn(line).map(_.matched).toList
            if (f.isDefined) {
              f.get.apply(words).foreach(w => {
                if (prefix.isDefined) {
                  m(prefix.get + w) += 1
                } else {
                  m(w) += 1
                }
              })
            } else {
              words.foreach(w => {
                if (prefix.isDefined) {
                  m(prefix.get + w) += 1
                } else {
                  m(w) += 1
                }
              })
            }
          })
          sourceStream.close()
        }
        if (m.isEmpty) throw new FileNotFoundException("Word count dictionary for spell checker does not exist or is empty")
        m
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
    sentences.map(TaggedSentence(_)).toArray
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
    val lines =
      sourceStream.content.getLines()
        .filter(_.nonEmpty)
        .map(sentence => wordTagSplitter(sentence, tagSeparator))
        .toArray
    sourceStream.close()
    lines.map(TaggedSentence(_))
  }

  /**
    * Reads POS Corpus from an entire directory of compiled sources
    * @param dirName compiled content only
    * @param tagSeparator tag separator for all corpuses
    * @param fileLimit limit of files to read. Can help clutter, overfitting
    * @return
    */
  def parsePOSCorpusFromDir(
                             dirName: String,
                             tagSeparator: Char,
                             fileLimit: Int
                           ): Array[TaggedSentence] = {
    try {
      new File(dirName).listFiles()
        .take(fileLimit)
        .flatMap(fileName => parsePOSCorpusFromSource(fileName.toString, tagSeparator))
    } catch {
      case _: NullPointerException =>
        val sourceStream = SourceStream(dirName)
        val res = sourceStream
          .content
          .getLines()
          .take(fileLimit)
          .flatMap(fileName => parsePOSCorpusFromSource(dirName + "/" + fileName, tagSeparator))
          .toArray
        sourceStream.close()
        res
    }
  }

  /**
    * Retrieves Corpuses from configured compiled directory set in configuration
    * @param fileLimit files limit to read
    * @return TaggedSentences for POS training
    */
  def retrievePOSCorpus(
                         posDirOrFilePath: String = "__default",
                         fileLimit: Int = 50
                       ): Array[TaggedSentence] = {
    //ToDo support multiple formats in corpus source
    val dirOrFilePath = if (posDirOrFilePath == "__default") config.getString("nlp.posDict.dir") else posDirOrFilePath
    val posFormat = config.getString("nlp.posDict.format")
    val posSeparator = config.getString("nlp.posDict.separator").head
    val result = {
      if (pathIsDirectory(dirOrFilePath)) parsePOSCorpusFromDir(dirOrFilePath, posSeparator, fileLimit)
      else parsePOSCorpusFromSource(dirOrFilePath, posSeparator)
    }
    if (result.isEmpty) throw new Exception("Empty corpus for POS")
    result
  }

  /**
    * Retrieves Lemma dictionary from configured compiled source set in configuration
    * @return a Dictionary for lemmas
    */
  def retrieveLemmaDict(
                         lemmaFilePath: String = "__default",
                         lemmaFormat: String = config.getString("nlp.lemmaDict.format"),
                         lemmaKeySep: String = config.getString("nlp.lemmaDict.kvSeparator"),
                         lemmaValSep: String = config.getString("nlp.lemmaDict.vSeparator")
                       ): Map[String, String] = {
    val filePath = if (lemmaFilePath == "__default") config.getString("nlp.lemmaDict.file") else lemmaFilePath
    ResourceHelper.flattenRevertValuesAsKeys(filePath, lemmaFormat, lemmaKeySep, lemmaValSep)
  }

  /**
    * Sentiment dictionaries from compiled sources set in configuration
    * @return Sentiment dictionary
    */
  def retrieveSentimentDict(sentFilePath: String = "__default"): Map[String, String] = {
    val filePath = if (sentFilePath == "__default") config.getString("nlp.sentimentDict.file") else sentFilePath
    val sentFormat = config.getString("nlp.sentimentDict.format")
    val sentSeparator = config.getString("nlp.sentimentDict.separator")
    ResourceHelper.parseKeyValueText(filePath, sentFormat, sentSeparator)
  }

  /**
    * Regex matcher rules
    * @param rulesFilePath
    * @param rulesFormat
    * @param rulesSeparator
    * @return
    */
  def retrieveRegexMatchRules(
                               rulesFilePath: String = "__default",
                               rulesFormat: String = config.getString("nlp.regexMatcher.format"),
                               rulesSeparator: String = config.getString("nlp.regexMatcher.separator")
                             ): Array[(String, String)] = {
    val filePath = if (rulesFilePath == "__default") config.getString("nlp.regexMatcher.file") else rulesFilePath
    ResourceHelper.parseTupleText(filePath, rulesFormat, rulesSeparator)
  }

  def retrieveEntityExtractorPhrases(
                                     entitiesPath: String = "__default",
                                     fileFormat: String = config.getString("nlp.entityExtractor.format")
                                   ): Array[String] = {
    val filePath = if (entitiesPath == "__default") config.getString("nlp.entityExtractor.file") else entitiesPath
    ResourceHelper.parseLinesText(filePath, fileFormat)
  }

  /**
    *
    * @param entitiesPath The path to load the dictionary from
    * @param fileFormat The format of the file specified at the path
    * @return The dictionary as a Map
    */
  def retrieveEntityDict(entitiesPath: String = "__default",
                         fileFormat: String = config.getString("nlp.entityRecognition.format")
                        ): Map[String, String] = {
     val filePath = if (entitiesPath == "__default") config.getString("nlp.entityRecognition.file") else entitiesPath
    ResourceHelper.parseKeyValueText(filePath, fileFormat, ":")
  }

  def retrieveEntityDicts(files: Array[String],
                          fileFormat: String = config.getString("nlp.entityRecognition.format")
                         ): Map[String, String] = {
    
     files.map( f => ResourceHelper.parseKeyValueText(f, fileFormat, ":") ).foldRight(Map[String, String]())( (m1, m2) => m1 ++ m2) 
  }

}
