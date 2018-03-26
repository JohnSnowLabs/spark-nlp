package com.johnsnowlabs.nlp.util.io

import java.io._
import java.net.{URL, URLDecoder}
import java.util.jar.JarFile

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.common.{TaggedSentence, TaggedWord}
import com.johnsnowlabs.nlp.util.io.ReadAs._
import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher}
import org.apache.hadoop.fs.{FileSystem, LocatedFileStatus, Path, RemoteIterator}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{Dataset, SparkSession}

import scala.collection.mutable.{ArrayBuffer, Map => MMap}
import scala.io.BufferedSource


/**
  * Created by saif on 28/04/17.
  */

/**
  * Helper one-place for IO management. Streams, source and external input should be handled from here
  */
object ResourceHelper {

  val spark: SparkSession = SparkSession.builder().getOrCreate()

  private def inputStreamOrSequence(fs: FileSystem, files: RemoteIterator[LocatedFileStatus]): InputStream = {
    val firstFile = files.next
    if (files.hasNext) {
      new SequenceInputStream(fs.open(firstFile.getPath), inputStreamOrSequence(fs, files))
    } else {
      fs.open(firstFile.getPath)
    }
  }

  /** Structure for a SourceStream coming from compiled content */
  case class SourceStream(resource: String) {
    val pipe: Option[InputStream] =
        /** Check whether it exists in file system */
      Option {
        val path = new Path(resource)
        val fs = FileSystem.get(path.toUri, spark.sparkContext.hadoopConfiguration)
        val files = fs.listFiles(new Path(resource), true)
        if (files.hasNext) inputStreamOrSequence(fs, files) else null
      }
    val content: BufferedSource = pipe.map(p => {
      new BufferedSource(p)("UTF-8")
    }).getOrElse(throw new FileNotFoundException(s"file or folder: $resource not found"))
    def close(): Unit = {
      content.close()
      pipe.foreach(_.close)
    }
  }

  private def fixTarget(path: String): String = {
    val toSearch = s"^.*target\\${File.separator}.*scala-.*\\${File.separator}.*classes\\${File.separator}"
    if (path.matches(toSearch + ".*")) {
      path.replaceFirst(toSearch, "")
    }
    else {
      path
    }
  }

  def getResourceStream(path: String): InputStream = {
    Option(getClass.getResourceAsStream(path))
      .getOrElse{
        getClass.getClassLoader().getResourceAsStream(path)
      }
  }

  def getResourceFile(path: String): URL = {
    var dirURL = getClass.getResource(path)

    if (dirURL == null)
      dirURL = getClass.getClassLoader.getResource(path)

    dirURL
  }

  def listResourceDirectory(path: String): Seq[String] = {
    val dirURL = getResourceFile(path)

    if (dirURL != null && dirURL.getProtocol.equals("file") && new File(dirURL.toURI).exists()) {
      /* A file path: easy enough */
      return new File(dirURL.toURI).listFiles.sorted.map(_.getPath).map(fixTarget(_))
    } else if (dirURL == null) {
        /* path not in resources and not in disk */
        throw new FileNotFoundException(path)
    }

    if (dirURL.getProtocol.equals("jar")) {
      /* A JAR path */
      val jarPath = dirURL.getPath.substring(5, dirURL.getPath.indexOf("!")) //strip out only the JAR file
      val jar = new JarFile(URLDecoder.decode(jarPath, "UTF-8"))
      val entries = jar.entries()
      val result = new ArrayBuffer[String]()

      val pathToCheck = path.replaceFirst("/", "")
      while(entries.hasMoreElements) {
        val name = entries.nextElement().getName//.replaceFirst("/", "")
        if (name.startsWith(pathToCheck)) { //filter according to the path
          var entry = name.substring(pathToCheck.length())
          val checkSubdir = entry.indexOf("/")
          if (checkSubdir >= 0) {
            // if it is a subdirectory, we just return the directory name
            entry = entry.substring(0, checkSubdir)
          }
          if (entry.nonEmpty) {
            result.append(pathToCheck + entry)
          }
        }
      }
      return result.distinct.sorted
    }

    throw new UnsupportedOperationException(s"Cannot list files for URL $dirURL")
  }

  def createDatasetFromText(
                             path: String, clean: Boolean = true,
                             includeFilename: Boolean = false,
                             includeRowNumber: Boolean = false,
                             aggregateByFile: Boolean = false
                           ): Dataset[_] = {
    require((includeFilename && aggregateByFile) || (!includeFilename && !aggregateByFile), "AggregateByFile requires includeFileName")
    import org.apache.spark.sql.functions._
    import spark.implicits._
    var data: Dataset[_] = spark.read.textFile(path)
    if (clean) data = data.as[String].map(_.trim()).filter(_.nonEmpty)
    if (includeFilename) data = data.withColumn("filename", input_file_name())
    if (aggregateByFile) data = data.groupBy("filename").agg(collect_list($"value").as("value"))
      .withColumn("text", concat_ws(" ", $"value"))
      .drop("value")
    if (includeRowNumber) {
      if (includeFilename && !aggregateByFile) {
        import org.apache.spark.sql.expressions.Window
        val w = Window.partitionBy("filename").orderBy("filename")
        data = data.withColumn("id", row_number().over(w))
      } else {
        data = data.withColumn("id", monotonically_increasing_id())
      }
    }
    data.withColumnRenamed("value", "text")
  }

  /**
    * General purpose key value parser from source
    * Currently read only text files
    * @return
    */
  def parseKeyValueText(
                         er: ExternalResource
                        ): Map[String, String] = {
    er.readAs match {
      case LINE_BY_LINE =>
        val sourceStream = SourceStream(er.path)
        val res = sourceStream.content.getLines.map (line => {
          val kv = line.split (er.options("delimiter")).map (_.trim)
          (kv.head, kv.last)
        }).toMap
        sourceStream.close()
        res
      case SPARK_DATASET =>
        import spark.implicits._
        val dataset = spark.read.options(er.options).format(er.options("format"))
          .options(er.options)
          .option("delimiter", er.options("delimiter"))
          .load(er.path)
          .toDF("key", "value")
        val keyValueStore = MMap.empty[String, String]
        dataset.as[(String, String)].foreach{kv => keyValueStore(kv._1) = kv._2}
        keyValueStore.toMap
      case _ =>
        throw new Exception("Unsupported readAs")
    }
  }

  /**
    * General purpose line parser from source
    * Currently read only text files
    * @return
    */
  def parseLines(
                      er: ExternalResource
                     ): Array[String] = {
    er.readAs match {
      case LINE_BY_LINE =>
        val sourceStream = SourceStream(er.path)
        val res = sourceStream.content.getLines.toArray
        sourceStream.close()
        res
      case SPARK_DATASET =>
        import spark.implicits._
        val dataset = spark.read.options(er.options).format(er.options("format")).load(er.path)
        val lineStore = spark.sparkContext.collectionAccumulator[String]
        dataset.as[String].foreach(l => lineStore.add(l))
        val result = lineStore.value.toArray.map(_.toString)
        lineStore.reset()
        result
      case _ =>
        throw new Exception("Unsupported readAs")
    }
  }

  /**
    * General purpose tuple parser from source
    * Currently read only text files
    * @return
    */
  def parseTupleText(
                         er: ExternalResource
                       ): Array[(String, String)] = {
    er.readAs match {
      case LINE_BY_LINE =>
        val sourceStream = SourceStream(er.path)
        val res = sourceStream.content.getLines.filter(_.nonEmpty).map (line => {
          val kv = line.split (er.options("delimiter")).map (_.trim)
          (kv.head, kv.last)
        }).toArray
        sourceStream.close()
        res
      case SPARK_DATASET =>
        import spark.implicits._
        val dataset = spark.read.options(er.options).format(er.options("format")).load(er.path)
        val lineStore = spark.sparkContext.collectionAccumulator[String]
        dataset.as[String].foreach(l => lineStore.add(l))
        val result = lineStore.value.toArray.map(line => {
          val kv = line.toString.split (er.options("delimiter")).map (_.trim)
          (kv.head, kv.last)
        })
        lineStore.reset()
        result
      case _ =>
        throw new Exception("Unsupported readAs")
    }
  }

  /**
    * General purpose tuple parser from source
    * Currently read only text files
    * @return
    */
  def parseTupleSentences(
                      er: ExternalResource
                    ): Array[TaggedSentence] = {
    er.readAs match {
      case LINE_BY_LINE =>
        val sourceStream = SourceStream(er.path)
        val result = sourceStream.content.getLines.filter(_.nonEmpty).map(line => {
          line.split("\\s+").filter(kv => {
            val s = kv.split(er.options("delimiter").head)
            s.length == 2 && s(0).nonEmpty && s(1).nonEmpty
          }).map(kv => {
            val p = kv.split(er.options("delimiter").head)
            TaggedWord(p(0), p(1))
          })
        }).toArray
        sourceStream.close()
        result.map(TaggedSentence(_))
      case SPARK_DATASET =>
        import spark.implicits._
        val dataset = spark.read.options(er.options).format(er.options("format")).load(er.path)
        val result = dataset.as[String].filter(_.nonEmpty).map(line => {
          line.split("\\s+").filter(kv => {
            val s = kv.split(er.options("delimiter").head)
            s.length == 2 && s(0).nonEmpty && s(1).nonEmpty
          }).map(kv => {
            val p = kv.split(er.options("delimiter").head)
            TaggedWord(p(0), p(1))
          })
        }).collect
        result.map(TaggedSentence(_))
      case _ =>
        throw new Exception("Unsupported readAs")
    }
  }

  /**
    * For multiple values per keys, this optimizer flattens all values for keys to have constant access
    */
  def flattenRevertValuesAsKeys(er: ExternalResource): Map[String, String] = {
    er.readAs match {
      case LINE_BY_LINE =>
        val m: MMap[String, String] = MMap()
        val sourceStream = SourceStream(er.path)
        sourceStream.content.getLines.foreach(line => {
          val kv = line.split(er.options("keyDelimiter")).map(_.trim)
          val key = kv(0)
          val values = kv(1).split(er.options("valueDelimiter")).map(_.trim)
          values.foreach(m(_) = key)
        })
        sourceStream.close()
        m.toMap
      case SPARK_DATASET =>
        import spark.implicits._
        val dataset = spark.read.options(er.options).format(er.options("format")).load(er.path)
        val valueAsKeys = MMap.empty[String, String]
        dataset.as[String].foreach(line => {
          val kv = line.split(er.options("keyDelimiter")).map(_.trim)
          val key = kv(0)
          val values = kv(1).split(er.options("valueDelimiter")).map(_.trim)
          values.foreach(v => valueAsKeys(v) = key)
        })
        valueAsKeys.toMap
      case _ =>
        throw new Exception("Unsupported readAs")
    }
  }

  def wordCount(
                 er: ExternalResource,
                 m: MMap[String, Long] = MMap.empty[String, Long].withDefaultValue(0),
                 p: Option[PipelineModel] = None
               ): MMap[String, Long] = {
    er.readAs match {
      case LINE_BY_LINE =>
        val sourceStream = SourceStream(er.path)
        val regex = er.options("tokenPattern").r
        sourceStream.content.getLines.foreach(line => {
          val words = regex.findAllMatchIn(line).map(_.matched).toList
            words.foreach(w => {
              m(w) += 1
            })
        })
        sourceStream.close()
        if (m.isEmpty) throw new FileNotFoundException("Word count dictionary for spell checker does not exist or is empty")
        m
      case SPARK_DATASET =>
        import spark.implicits._
        val dataset = spark.read.options(er.options).format(er.options("format")).load(er.path)
        val transformation = {
          if (p.isDefined) {
            p.get.transform(dataset)
          } else {
            val documentAssembler = new DocumentAssembler()
              .setInputCol("value")
            val tokenizer = new Tokenizer()
              .setInputCols("document")
              .setOutputCol("token")
              .setTargetPattern(er.options("tokenPattern"))
            val finisher = new Finisher()
              .setInputCols("token")
              .setOutputCols("finished")
              .setAnnotationSplitSymbol("--")
            new Pipeline()
              .setStages(Array(documentAssembler, tokenizer, finisher))
              .fit(dataset)
              .transform(dataset)
          }
        }
        val wordCount = MMap.empty[String, Long].withDefaultValue(0)
        transformation
          .select("finished").as[String]
          .foreach(text => text.split("--").foreach(t => {
            wordCount(t) += 1
          }))
        wordCount
      case _ => throw new IllegalArgumentException("format not available for word count")
    }
  }
}
