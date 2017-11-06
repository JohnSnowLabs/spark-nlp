package com.johnsnowlabs.nlp.util.io

import java.io.{File, FileNotFoundException, InputStream}

import com.johnsnowlabs.nlp.annotators.{Normalizer, RegexTokenizer}
import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher}
import com.johnsnowlabs.nlp.annotators.common.{TaggedSentence, TaggedWord}
import com.johnsnowlabs.nlp.util.io.ResourceFormat._
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.{ArrayBuffer, Map => MMap}
import scala.io.Source

import java.io.IOException
import java.net.URISyntaxException
import java.net.URL
import java.net.URLDecoder
import java.util
import java.util.jar.JarEntry
import java.util.jar.JarFile


/**
  * Created by saif on 28/04/17.
  */

/**
  * Helper one-place for IO management. Streams, source and external input should be handled from here
  */
object ResourceHelper {

  private val spark: SparkSession = SparkSession.builder().getOrCreate()


  def listDirectory(path: String): Seq[String] = {
    var dirURL = getClass.getResource(path)

    if (dirURL == null)
      dirURL = getClass.getClassLoader.getResource(path)

    if (dirURL != null && dirURL.getProtocol.equals("file")) {
      /* A file path: easy enough */
      return new File(dirURL.toURI).list().sorted
    }

    if (dirURL.getProtocol().equals("jar")) {
      /* A JAR path */
      val jarPath = dirURL.getPath().substring(5, dirURL.getPath().indexOf("!")) //strip out only the JAR file
      val jar = new JarFile(URLDecoder.decode(jarPath, "UTF-8"))
      val entries = jar.entries()
      val result = new ArrayBuffer[String]()

      val pathToCheck = path.replaceFirst("/", "")
      while(entries.hasMoreElements()) {
        val name = entries.nextElement().getName().replaceFirst("/", "")
        if (name.startsWith(pathToCheck)) { //filter according to the path
          var entry = name.substring(pathToCheck.length())
          val checkSubdir = entry.indexOf("/")
          if (checkSubdir >= 0) {
            // if it is a subdirectory, we just return the directory name
            entry = entry.substring(0, checkSubdir)
          }
          if (entry.nonEmpty)
            result.append(entry)
        }
      }
      return result.distinct.sorted
    }

    throw new UnsupportedOperationException(s"Cannot list files for URL $dirURL")
  }

  /** Structure for a SourceStream coming from compiled content */
  case class SourceStream(resource: String) {
    val pipe: Option[InputStream] = try {
      var stream = getClass.getResourceAsStream(resource)
      if (stream == null)
        stream = getClass.getClassLoader.getResourceAsStream(resource)

      Some(stream)
    } catch {
      case _: NullPointerException => None
    }
    val content: Source = pipe.map(p => Source.fromInputStream(p)("UTF-8")).getOrElse(Source.fromFile(resource, "UTF-8"))
    def close(): Unit = {
      content.close()
      pipe.foreach(_.close())
    }
  }

  /** Checks whether a path points to directory */
  def pathIsDirectory(path: String): Boolean = {
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
                         format: Format,
                         keySep: String,
                         valueSep: String): Map[String, Array[String]] = {
    format match {
      case TXT =>
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
                          format: Format,
                          keySep: String
                        ): Map[String, String] = {
    format match {
      case TXT =>
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
                      format: Format
                     ): Array[String] = {
    format match {
      case TXT =>
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
                         format: Format,
                         keySep: String
                       ): Array[(String, String)] = {
    format match {
      case TXT =>
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
                                 format: Format,
                                 keySep: String,
                                 valueSep: String): Map[String, String] = {
    format match {
      case TXT =>
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
                 format: Format,
                 m: MMap[String, Int] = MMap.empty[String, Int].withDefaultValue(0),
                 clean: Boolean = true,
                 prefix: Option[String] = None,
                 f: Option[(List[String] => List[String])] = None
               ): MMap[String, Int] = {
    format match {
      case TXT =>
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
      case TXTDS =>
        import spark.implicits._
        val dataset = spark.read.textFile(source)
        val wordCount = spark.sparkContext.broadcast(MMap.empty[String, Int].withDefaultValue(0))
        val documentAssembler = new DocumentAssembler()
          .setInputCol("value")
        val tokenizer = new RegexTokenizer()
          .setInputCols("document")
          .setOutputCol("token")
        val normalizer = new Normalizer()
          .setInputCols("token")
          .setOutputCol("normal")
        val finisher = new Finisher()
          .setInputCols("normal")
          .setOutputCols("finished")
          .setAnnotationSplitSymbol("--")
        new Pipeline()
          .setStages(Array(documentAssembler, tokenizer, normalizer, finisher))
          .fit(dataset)
          .transform(dataset)
          .select("finished").as[String]
          .foreach(text => text.split("--").foreach(t => {
            wordCount.value(t) += 1
          }))
        val result = wordCount.value
        wordCount.destroy()
        result
      case _ => throw new IllegalArgumentException("format not available for word count")
    }
  }

}
