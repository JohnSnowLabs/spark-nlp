package com.johnsnowlabs.nlp.util.io

import java.io._
import java.net.{URL, URLDecoder}
import java.nio.file.{Files, Paths}
import java.util.jar.JarFile

import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.common.{TaggedSentence, TaggedWord}
import com.johnsnowlabs.nlp.util.io.ReadAs._
import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

import scala.collection.mutable.{ArrayBuffer, Map => MMap}
import scala.io.BufferedSource

/**
  * Created by saif on 28/04/17.
  */

/**
  * Helper one-place for IO management. Streams, source and external input should be handled from here
  */
object ResourceHelper {

  def getActiveSparkSession: SparkSession =
    SparkSession.getActiveSession.getOrElse(SparkSession.builder()
      .appName("SparkNLP Default Session")
      .master("local[*]")
      .config("spark.driver.memory","12G")
      .config("spark.driver.maxResultSize", "2G")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "500m")
      .getOrCreate()
    )

  lazy val spark: SparkSession = getActiveSparkSession

  /** Structure for a SourceStream coming from compiled content */
  case class SourceStream(resource: String) {
    val path = new Path(resource)
    val fs = FileSystem.get(path.toUri, spark.sparkContext.hadoopConfiguration)
    if (!fs.exists(path))
      throw new FileNotFoundException(s"file or folder: $resource not found")
    val pipe: Seq[InputStream] = {
      /** Check whether it exists in file system */
      val files = fs.listFiles(path, true)
      val buffer = ArrayBuffer.empty[InputStream]
      while (files.hasNext) buffer.append(fs.open(files.next().getPath))
      buffer
    }
    val openBuffers: Seq[BufferedSource] = pipe.map(pp => { new BufferedSource(pp)("UTF-8")})
    val content: Seq[Iterator[String]] = openBuffers.map(c => c.getLines())

    def copyToLocal(prefix: String = "sparknlp_tmp_"): String = {
      if (fs.getScheme == "file")
        return resource
      val files = fs.listFiles(path, false)
      val dst: Path = new Path(Files.createTempDirectory(prefix).toUri)
      while (files.hasNext) {
        fs.copyFromLocalFile(files.next.getPath, dst)
      }
      dst.toString
    }
    def close(): Unit = {
      openBuffers.foreach(_.close())
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

  def copyToLocal(path: String): String = {
    val resource = SourceStream(path)
    resource.copyToLocal()
  }

  /** NOT thread safe. Do not call from executors. */
  def getResourceStream(path: String): InputStream = {
    if (new File(path).exists())
      new FileInputStream(new File(path))
    else {
      Option(getClass.getResourceAsStream(path))
        .getOrElse {
          getClass.getClassLoader.getResourceAsStream(path)
        }
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

      val pathToCheck = path
        .stripPrefix(File.separator.replaceAllLiterally("\\", "/"))
        .stripSuffix(File.separator) +
        File.separator.replaceAllLiterally("\\", "/")

      while(entries.hasMoreElements) {
        val name = entries.nextElement().getName.stripPrefix(File.separator)
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
        val res = sourceStream.content.flatMap(c => c.map (line => {
          val kv = line.split (er.options("delimiter"))
          (kv.head.trim, kv.last.trim)
        })).toMap
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
        val res = sourceStream.content.flatten.toArray
        sourceStream.close()
        res
      case SPARK_DATASET =>
        import spark.implicits._
        spark.read.options(er.options).format(er.options("format")).load(er.path).as[String].collect
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
        val res = sourceStream.content.flatMap(c => c.filter(_.nonEmpty).map (line => {
          val kv = line.split (er.options("delimiter")).map (_.trim)
          (kv.head, kv.last)
        })).toArray
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
        val result = sourceStream.content.flatMap(c => c.filter(_.nonEmpty).map(line => {
          line.split("\\s+").filter(kv => {
            val s = kv.split(er.options("delimiter").head)
            s.length == 2 && s(0).nonEmpty && s(1).nonEmpty
          }).map(kv => {
            val p = kv.split(er.options("delimiter").head)
            TaggedWord(p(0), p(1))
          })
        })).toArray
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

  def parseTupleSentencesDS(
                             er: ExternalResource
                           ): Dataset[TaggedSentence] = {
    er.readAs match {
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
        })
        result.map(TaggedSentence(_))
      case _ =>
        throw new Exception("Unsupported readAs. If you're training POS with large dataset, consider PerceptronApproachDistributed")
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
        sourceStream.content.foreach(c => c.foreach(line => {
          val kv = line.split(er.options("keyDelimiter")).map(_.trim)
          val key = kv(0)
          val values = kv(1).split(er.options("valueDelimiter")).map(_.trim)
          values.foreach(m(_) = key)
        }))
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

  /**
    * General purpose read saved Parquet
    * Currently read only Parquet format
    * @return
    */
  def readParquetSparkDatFrame(
                           er: ExternalResource
                         ): DataFrame = {
    er.readAs match {
      case SPARK_DATASET =>
        val dataset = spark.read.options(er.options).format(er.options("format")).load(er.path)
        dataset
      case _ =>
        throw new Exception("Unsupported readAs - only accepts SPARK_DATASET")
    }
  }

  def getWordCount(externalResource: ExternalResource,
                   wordCount: MMap[String, Long] = MMap.empty[String, Long].withDefaultValue(0),
                   pipeline: Option[PipelineModel] = None
               ): MMap[String, Long] = {
    externalResource.readAs match {
      case LINE_BY_LINE =>
        val sourceStream = SourceStream(externalResource.path)
        val regex = externalResource.options("tokenPattern").r
        sourceStream.content.foreach(c => c.foreach{line => {
          val words: List[String] = regex.findAllMatchIn(line).map(_.matched).toList
          words.foreach(w =>
            // Creates a Map of frequency words: word -> frequency based on ExternalResource
            wordCount(w) += 1
          )
        }})
        sourceStream.close()
        if (wordCount.isEmpty)
          throw new FileNotFoundException("Word count dictionary for spell checker does not exist or is empty")
        wordCount
      case SPARK_DATASET =>
        import spark.implicits._
        val dataset = spark.read.options(externalResource.options).format(externalResource.options("format"))
          .load(externalResource.path)
        val transformation = {
          if (pipeline.isDefined) {
            pipeline.get.transform(dataset)
          } else {
            val documentAssembler = new DocumentAssembler()
              .setInputCol("value")
            val tokenizer = new Tokenizer()
              .setInputCols("document")
              .setOutputCol("token")
              .setTargetPattern(externalResource.options("tokenPattern"))
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

  def getFilesContentBuffer(externalResource: ExternalResource): Seq[Iterator[String]] = {
    externalResource.readAs match {
      case LINE_BY_LINE =>
          SourceStream(externalResource.path).content
      case _ =>
        throw new Exception("Unsupported readAs")
    }
  }

  def listLocalFiles(path: String): List[File] = {
    val filesPath = Option(new File(path).listFiles())
    val files = filesPath.getOrElse(throw new FileNotFoundException(s"folder: $path not found"))
    files.toList
  }

  def validFile(path: String): Boolean = {
    val isValid = Files.exists(Paths.get(path))

    if (isValid) {
      isValid
    } else {
      throw new FileNotFoundException(path)
    }

  }

}
