/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.util.io

import com.amazonaws.AmazonServiceException
import com.johnsnowlabs.client.CloudResources
import com.johnsnowlabs.client.aws.AWSGateway
import com.johnsnowlabs.client.util.CloudHelper
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.common.{TaggedSentence, TaggedWord}
import com.johnsnowlabs.nlp.util.io.ReadAs._
import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher}
import com.johnsnowlabs.util.ConfigHelper
import org.apache.commons.io.FileUtils
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

import java.io._
import java.net.{URI, URL, URLDecoder}
import java.nio.file
import java.nio.file.{Files, Paths}
import java.util.jar.JarFile
import scala.collection.mutable.{ArrayBuffer, Map => MMap}
import scala.io.BufferedSource
import scala.util.{Failure, Success, Try}

/** Helper one-place for IO management. Streams, source and external input should be handled from
  * here
  */
object ResourceHelper {

  def getActiveSparkSession: SparkSession =
    SparkSession.getActiveSession.getOrElse(
      SparkSession
        .builder()
        .appName("SparkNLP Default Session")
        .master("local[*]")
        .config("spark.driver.memory", "22G")
        .config("spark.driver.maxResultSize", "0")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "1000m")
        .getOrCreate())

  def getSparkSessionWithS3(
      awsAccessKeyId: String,
      awsSecretAccessKey: String,
      hadoopAwsVersion: String = ConfigHelper.hadoopAwsVersion,
      AwsJavaSdkVersion: String = ConfigHelper.awsJavaSdkVersion,
      region: String = "us-east-1",
      s3Impl: String = "org.apache.hadoop.fs.s3a.S3AFileSystem",
      pathStyleAccess: Boolean = true,
      credentialsProvider: String = "TemporaryAWSCredentialsProvider",
      awsSessionToken: Option[String] = None): SparkSession = {

    require(
      SparkSession.getActiveSession.isEmpty,
      "Spark session already running, can't apply new configuration for S3.")

    val sparkSession = SparkSession
      .builder()
      .appName("SparkNLP Session with S3 Support")
      .master("local[*]")
      .config("spark.driver.memory", "22G")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.kryoserializer.buffer.max", "1000M")
      .config("spark.driver.maxResultSize", "0")
      .config("spark.hadoop.fs.s3a.access.key", awsAccessKeyId)
      .config("spark.hadoop.fs.s3a.secret.key", awsSecretAccessKey)
      .config(ConfigHelper.awsExternalRegion, region)
      .config(
        "spark.hadoop.fs.s3a.aws.credentials.provider",
        s"org.apache.hadoop.fs.s3a.$credentialsProvider")
      .config("spark.hadoop.fs.s3a.impl", s3Impl)
      .config(
        "spark.jars.packages",
        "org.apache.hadoop:hadoop-aws:" + hadoopAwsVersion + ",com.amazonaws:aws-java-sdk:" + AwsJavaSdkVersion)
      .config("spark.hadoop.fs.s3a.path.style.access", pathStyleAccess.toString)

    if (credentialsProvider == "TemporaryAWSCredentialsProvider") {
      require(
        awsSessionToken.isDefined,
        "AWS Session token needs to be provided for TemporaryAWSCredentialsProvider.")
      sparkSession.config("spark.hadoop.fs.s3a.session.token", awsSessionToken.get)
    }

    sparkSession.getOrCreate()
  }

  lazy val spark: SparkSession = getActiveSparkSession

  /** Structure for a SourceStream coming from compiled content */
  case class SourceStream(resource: String) {

    var fileSystem: Option[FileSystem] = None
    private val (pathExists: Boolean, path: Option[Path]) = OutputHelper.doesPathExists(resource)
    if (!pathExists) {
      throw new FileNotFoundException(s"file or folder: $resource not found")
    } else {
      fileSystem = Some(OutputHelper.getFileSystem(resource))
    }

    val pipe: Seq[InputStream] = getPipe(fileSystem.get)
    private val openBuffers: Seq[BufferedSource] = pipe.map(pp => {
      new BufferedSource(pp)("UTF-8")
    })
    val content: Seq[Iterator[String]] = openBuffers.map(c => c.getLines())

    private def getPipe(fileSystem: FileSystem): Seq[InputStream] = {
      if (fileSystem.getScheme == "s3a") {
        val awsGateway = new AWSGateway()
        val (bucket, s3Path) = CloudHelper.parseS3URI(path.get.toString)
        val inputStreams = awsGateway.listS3Files(bucket, s3Path).map { summary =>
          val s3Object = awsGateway.getS3Object(bucket, summary.getKey)
          s3Object.getObjectContent
        }
        inputStreams
      } else {
        val files = fileSystem.listFiles(path.get, true)
        val buffer = ArrayBuffer.empty[InputStream]
        while (files.hasNext) buffer.append(fileSystem.open(files.next().getPath))
        buffer
      }
    }

    /** Copies the resource into a local temporary folder and returns the folders URI.
      *
      * @param prefix
      *   Prefix for the temporary folder.
      * @return
      *   URI of the created temporary folder with the resource
      */
    def copyToLocal(prefix: String = "sparknlp_tmp_"): URI = {
      if (fileSystem.get.getScheme == "file")
        return URI.create(resource)

      val destination: file.Path = Files.createTempDirectory(prefix)

      val destinationUri = fileSystem.get.getScheme match {
        case "hdfs" =>
          fileSystem.get.copyToLocalFile(false, path.get, new Path(destination.toUri), true)
          if (fileSystem.get.getFileStatus(path.get).isDirectory)
            Paths.get(destination.toString, path.get.getName).toUri
          else destination.toUri
        case "dbfs" =>
          val dbfsPath = path.get.toString.replace("dbfs:/", "/dbfs/")
          val sourceFile = new File(dbfsPath)
          val targetFile = new File(destination.toString)
          if (sourceFile.isFile) FileUtils.copyFileToDirectory(sourceFile, targetFile)
          else FileUtils.copyDirectory(sourceFile, targetFile)
          targetFile.toURI
        case _ =>
          val files = fileSystem.get.listFiles(path.get, false)
          while (files.hasNext) {
            fileSystem.get.copyFromLocalFile(files.next.getPath, new Path(destination.toUri))
          }
          destination.toUri
      }

      destinationUri
    }

    def close(): Unit = {
      openBuffers.foreach(_.close())
      pipe.foreach(_.close)
    }
  }

  private def fixTarget(path: String): String = {
    val toSearch =
      s"^.*target\\${File.separator}.*scala-.*\\${File.separator}.*classes\\${File.separator}"
    if (path.matches(toSearch + ".*")) {
      path.replaceFirst(toSearch, "")
    } else {
      path
    }
  }

  /** Copies the remote resource to a local temporary folder and returns its absolute path.
    *
    * Currently, file:/, s3:/, hdfs:/ and dbfs:/ are supported.
    *
    * If the file is already on the local file system just the absolute path will be returned
    * instead.
    * @param path
    *   Path to the resource
    * @return
    *   Absolute path to the temporary or local folder of the resource
    */
  def copyToLocal(path: String): String = try {
    val localUri =
      if (CloudHelper.isCloudPath(path)) { // Download directly from Cloud Buckets
        CloudResources.downloadBucketToLocalTmp(path)
      } else { // Use Source Stream
        val pathWithProtocol: String =
          if (URI.create(path).getScheme == null) new File(path).toURI.toURL.toString else path
        val resource = SourceStream(pathWithProtocol)
        resource.copyToLocal()
      }

    new File(localUri).getAbsolutePath // Platform independent path
  } catch {
    case awsE: AmazonServiceException =>
      println("Error while retrieving folder from S3. Make sure you have set the right " +
        "access keys with proper permissions in your configuration. For an example please see " +
        "https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/training/english/dl-ner/mfa_ner_graphs_s3.ipynb")
      throw awsE
    case e: Exception =>
      val copyToLocalErrorMessage: String =
        "Please make sure the provided path exists and is accessible while keeping in mind only file:/, hdfs:/, dbfs:/ and s3:/ protocols are supported at the moment."
      println(
        s"$e \n Therefore, could not create temporary local directory for provided path $path. $copyToLocalErrorMessage")
      throw e
  }

  /** NOT thread safe. Do not call from executors. */
  def getResourceStream(path: String): InputStream = {
    if (new File(path).exists())
      new FileInputStream(new File(path))
    else {
      Option(getClass.getResourceAsStream(path))
        .getOrElse {
          Option(getClass.getClassLoader.getResourceAsStream(path))
            .getOrElse(throw new IllegalArgumentException(f"Wrong resource path $path"))
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
      return new File(dirURL.toURI).listFiles.sorted.map(_.getPath).map(fixTarget)
    } else if (dirURL == null) {
      /* path not in resources and not in disk */
      throw new FileNotFoundException(path)
    }

    if (dirURL.getProtocol.equals("jar")) {
      /* A JAR path */
      val jarPath =
        dirURL.getPath.substring(5, dirURL.getPath.indexOf("!")) // strip out only the JAR file
      val jar = new JarFile(URLDecoder.decode(jarPath, "UTF-8"))
      val entries = jar.entries()
      val result = new ArrayBuffer[String]()

      val pathToCheck = path
        .stripPrefix(File.separator.replaceAllLiterally("\\", "/"))
        .stripSuffix(File.separator) +
        File.separator.replaceAllLiterally("\\", "/")

      while (entries.hasMoreElements) {
        val name = entries.nextElement().getName.stripPrefix(File.separator)
        if (name.startsWith(pathToCheck)) { // filter according to the path
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

  /** General purpose key value parser from source Currently read only text files
    *
    * @return
    */
  def parseKeyValueText(er: ExternalResource): Map[String, String] = {
    er.readAs match {
      case TEXT =>
        val sourceStream = SourceStream(er.path)
        val res = sourceStream.content
          .flatMap(c =>
            c.map(line => {
              val kv = line.split(er.options("delimiter"))
              (kv.head.trim, kv.last.trim)
            }))
          .toMap
        sourceStream.close()
        res
      case SPARK =>
        import spark.implicits._
        val dataset = spark.read
          .options(er.options)
          .format(er.options("format"))
          .options(er.options)
          .option("delimiter", er.options("delimiter"))
          .load(er.path)
          .toDF("key", "value")
        val keyValueStore = MMap.empty[String, String]
        dataset.as[(String, String)].foreach { kv =>
          keyValueStore(kv._1) = kv._2
        }
        keyValueStore.toMap
      case _ =>
        throw new Exception("Unsupported readAs")
    }
  }

  def parseKeyListValues(externalResource: ExternalResource): Map[String, List[String]] = {
    externalResource.readAs match {
      case TEXT =>
        val sourceStream = SourceStream(externalResource.path)
        val keyValueStore = MMap.empty[String, List[String]]
        sourceStream.content.foreach(content =>
          content.foreach { line =>
            {
              val keyValues = line.split(externalResource.options("delimiter"))
              val key = keyValues.head
              val value = keyValues.drop(1).toList
              val storedValue = keyValueStore.get(key)
              if (storedValue.isDefined && !storedValue.contains(value)) {
                keyValueStore.update(key, storedValue.get ++ value)
              } else keyValueStore(key) = value
            }
          })
        sourceStream.close()
        keyValueStore.toMap
    }
  }

  def parseKeyArrayValues(externalResource: ExternalResource): Map[String, Array[Float]] = {
    externalResource.readAs match {
      case TEXT =>
        val sourceStream = SourceStream(externalResource.path)
        val keyValueStore = MMap.empty[String, Array[Float]]
        sourceStream.content.foreach(content =>
          content.foreach { line =>
            {
              val keyValues = line.split(externalResource.options("delimiter"))
              val key = keyValues.head
              val value = keyValues.drop(1).map(x => x.toFloat)
              if (value.length > 1) {
                keyValueStore(key) = value
              }
            }
          })
        sourceStream.close()
        keyValueStore.toMap
    }
  }

  /** General purpose line parser from source Currently read only text files
    *
    * @return
    */
  def parseLines(er: ExternalResource): Array[String] = {
    er.readAs match {
      case TEXT =>
        val sourceStream = SourceStream(er.path)
        val res = sourceStream.content.flatten.toArray
        sourceStream.close()
        res
      case SPARK =>
        import spark.implicits._
        spark.read
          .options(er.options)
          .format(er.options("format"))
          .load(er.path)
          .as[String]
          .collect
      case _ =>
        throw new Exception("Unsupported readAs")
    }
  }

  /** General purpose line parser from source Currently read only text files
    *
    * @return
    */
  def parseLinesIterator(er: ExternalResource): Seq[Iterator[String]] = {
    er.readAs match {
      case TEXT =>
        val sourceStream = SourceStream(er.path)
        sourceStream.content
      case _ =>
        throw new Exception("Unsupported readAs")
    }
  }

  /** General purpose tuple parser from source Currently read only text files
    *
    * @return
    */
  def parseTupleText(er: ExternalResource): Array[(String, String)] = {
    er.readAs match {
      case TEXT =>
        val sourceStream = SourceStream(er.path)
        val res = sourceStream.content
          .flatMap(c =>
            c.filter(_.nonEmpty)
              .map(line => {
                val kv = line.split(er.options("delimiter")).map(_.trim)
                (kv.head, kv.last)
              }))
          .toArray
        sourceStream.close()
        res
      case SPARK =>
        import spark.implicits._
        val dataset = spark.read.options(er.options).format(er.options("format")).load(er.path)
        val lineStore = spark.sparkContext.collectionAccumulator[String]
        dataset.as[String].foreach(l => lineStore.add(l))
        val result = lineStore.value.toArray.map(line => {
          val kv = line.toString.split(er.options("delimiter")).map(_.trim)
          (kv.head, kv.last)
        })
        lineStore.reset()
        result
      case _ =>
        throw new Exception("Unsupported readAs")
    }
  }

  /** General purpose tuple parser from source Currently read only text files
    *
    * @return
    */
  def parseTupleSentences(er: ExternalResource): Array[TaggedSentence] = {
    er.readAs match {
      case TEXT =>
        val sourceStream = SourceStream(er.path)
        val result = sourceStream.content
          .flatMap(c =>
            c.filter(_.nonEmpty)
              .map(line => {
                line
                  .split("\\s+")
                  .filter(kv => {
                    val s = kv.split(er.options("delimiter").head)
                    s.length == 2 && s(0).nonEmpty && s(1).nonEmpty
                  })
                  .map(kv => {
                    val p = kv.split(er.options("delimiter").head)
                    TaggedWord(p(0), p(1))
                  })
              }))
          .toArray
        sourceStream.close()
        result.map(TaggedSentence(_))
      case SPARK =>
        import spark.implicits._
        val dataset = spark.read.options(er.options).format(er.options("format")).load(er.path)
        val result = dataset
          .as[String]
          .filter(_.nonEmpty)
          .map(line => {
            line
              .split("\\s+")
              .filter(kv => {
                val s = kv.split(er.options("delimiter").head)
                s.length == 2 && s(0).nonEmpty && s(1).nonEmpty
              })
              .map(kv => {
                val p = kv.split(er.options("delimiter").head)
                TaggedWord(p(0), p(1))
              })
          })
          .collect
        result.map(TaggedSentence(_))
      case _ =>
        throw new Exception("Unsupported readAs")
    }
  }

  def parseTupleSentencesDS(er: ExternalResource): Dataset[TaggedSentence] = {
    er.readAs match {
      case SPARK =>
        import spark.implicits._
        val dataset = spark.read.options(er.options).format(er.options("format")).load(er.path)
        val result = dataset
          .as[String]
          .filter(_.nonEmpty)
          .map(line => {
            line
              .split("\\s+")
              .filter(kv => {
                val s = kv.split(er.options("delimiter").head)
                s.length == 2 && s(0).nonEmpty && s(1).nonEmpty
              })
              .map(kv => {
                val p = kv.split(er.options("delimiter").head)
                TaggedWord(p(0), p(1))
              })
          })
        result.map(TaggedSentence(_))
      case _ =>
        throw new Exception(
          "Unsupported readAs. If you're training POS with large dataset, consider PerceptronApproachDistributed")
    }
  }

  /** For multiple values per keys, this optimizer flattens all values for keys to have constant
    * access
    */
  def flattenRevertValuesAsKeys(er: ExternalResource): Map[String, String] = {
    er.readAs match {
      case TEXT =>
        val m: MMap[String, String] = MMap()
        val sourceStream = SourceStream(er.path)
        sourceStream.content.foreach(c =>
          c.foreach(line => {
            val kv = line.split(er.options("keyDelimiter")).map(_.trim)
            if (kv.length > 1) {
              val key = kv(0)
              val values = kv(1).split(er.options("valueDelimiter")).map(_.trim)
              values.foreach(m(_) = key)
            }
          }))
        sourceStream.close()
        m.toMap
      case SPARK =>
        import spark.implicits._
        val dataset = spark.read.options(er.options).format(er.options("format")).load(er.path)
        val valueAsKeys = MMap.empty[String, String]
        dataset
          .as[String]
          .foreach(line => {
            val kv = line.split(er.options("keyDelimiter")).map(_.trim)
            if (kv.length > 1) {
              val key = kv(0)
              val values = kv(1).split(er.options("valueDelimiter")).map(_.trim)
              values.foreach(v => valueAsKeys(v) = key)
            }
          })
        valueAsKeys.toMap
      case _ =>
        throw new Exception("Unsupported readAs")
    }
  }

  /** General purpose read saved Parquet Currently read only Parquet format
    *
    * @return
    */
  def readSparkDataFrame(er: ExternalResource): DataFrame = {
    er.readAs match {
      case SPARK =>
        val dataset = spark.read.options(er.options).format(er.options("format")).load(er.path)
        dataset
      case _ =>
        throw new Exception("Unsupported readAs - only accepts SPARK")
    }
  }

  def getWordCount(
      externalResource: ExternalResource,
      wordCount: MMap[String, Long] = MMap.empty[String, Long].withDefaultValue(0),
      pipeline: Option[PipelineModel] = None): MMap[String, Long] = {
    externalResource.readAs match {
      case TEXT =>
        val sourceStream = SourceStream(externalResource.path)
        val regex = externalResource.options("tokenPattern").r
        sourceStream.content.foreach(c =>
          c.foreach { line =>
            {
              val words: List[String] = regex.findAllMatchIn(line).map(_.matched).toList
              words.foreach(w =>
                // Creates a Map of frequency words: word -> frequency based on ExternalResource
                wordCount(w) += 1)
            }
          })
        sourceStream.close()
        if (wordCount.isEmpty)
          throw new FileNotFoundException(
            "Word count dictionary for spell checker does not exist or is empty")
        wordCount
      case SPARK =>
        import spark.implicits._
        val dataset = spark.read
          .options(externalResource.options)
          .format(externalResource.options("format"))
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
          .select("finished")
          .as[String]
          .foreach(text =>
            text
              .split("--")
              .foreach(t => {
                wordCount(t) += 1
              }))
        wordCount
      case _ => throw new IllegalArgumentException("format not available for word count")
    }
  }

  def getFilesContentBuffer(externalResource: ExternalResource): Seq[Iterator[String]] = {
    externalResource.readAs match {
      case TEXT =>
        SourceStream(externalResource.path).content
      case _ =>
        throw new Exception("Unsupported readAs")
    }
  }

  def listLocalFiles(path: String): List[File] = {
    val fileSystem = OutputHelper.getFileSystem(path)

    val filesPath = fileSystem.getScheme match {
      case "hdfs" =>
        if (path.startsWith("file:")) {
          Option(new File(path.replace("file:", "")).listFiles())
        } else {
          try {
            val filesIterator = fileSystem.listFiles(new Path(path), false)
            val files: ArrayBuffer[File] = ArrayBuffer()

            while (filesIterator.hasNext) {
              val file = new File(filesIterator.next().getPath.toString)
              files.append(file)
            }

            Option(files.toArray)
          } catch {
            case _: FileNotFoundException =>
              Option(new File(path).listFiles())
          }

        }
      case "dbfs" if path.startsWith("dbfs:") =>
        Option(new File(path.replace("dbfs:", "/dbfs/")).listFiles())
      case _ => Option(new File(path).listFiles())
    }

    val files = filesPath.getOrElse(throw new FileNotFoundException(s"folder: $path not found"))
    files.toList
  }

  def getFileFromPath(pathToFile: String): File = {
    val fileSystem = OutputHelper.getFileSystem
    val filePath = fileSystem.getScheme match {
      case "hdfs" =>
        if (pathToFile.startsWith("file:")) {
          new File(pathToFile.replace("file:", ""))
        } else new File(pathToFile)
      case "dbfs" if pathToFile.startsWith("dbfs:") =>
        new File(pathToFile.replace("dbfs:", "/dbfs/"))
      case _ => new File(pathToFile)
    }

    filePath
  }

  def validFile(path: String): Boolean = {
    if (path.isEmpty) return false

    if (path.contains(",")) {
      return path.split(",").map(_.trim).forall(p => validFile(p))
    }

    var isValid = validLocalFile(path) match {
      case Success(value) => value
      case Failure(_) => false
    }

    if (!isValid) {
      validHadoopFile(path) match {
        case Success(value) => isValid = value
        case Failure(_) => isValid = false
      }
    }

    if (!isValid) {
      validDbfsFile(path) match {
        case Success(value) => isValid = value
        case Failure(_) => isValid = false
      }
    }

    isValid
  }

  private def validLocalFile(path: String): Try[Boolean] = Try {
    Files.exists(Paths.get(path))
  }

  private def validHadoopFile(path: String): Try[Boolean] = Try {
    val hadoopPath = new Path(path)
    val fileSystem = OutputHelper.getFileSystem
    fileSystem.exists(hadoopPath)
  }

  private def validDbfsFile(path: String): Try[Boolean] = Try {
    getFileFromPath(path).exists()
  }

  def isValidURL(url: String): Boolean = {
    try {
      new URI(url).parseServerAuthority()
      true
    } catch {
      case _: Exception => false
    }
  }

  def fileSystemFromPath(path: String): FileSystem = {
    val uri = new URI(path.replaceAllLiterally("\\", "/"))
    FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
  }
  def isHTTPProtocol(urlStr: String): Boolean = {
    try {
      val url = new URL(urlStr)
      url.getProtocol == "http" || url.getProtocol == "https"
    } catch {
      case _: Exception => false
    }
  }

}
