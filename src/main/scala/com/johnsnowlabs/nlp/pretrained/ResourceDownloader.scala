/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.pretrained

import com.johnsnowlabs.nlp.{DocumentAssembler, pretrained}
import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.nlp.annotators.classifier.dl.{ClassifierDLModel, MultiClassifierDLModel, SentimentDLModel}
import com.johnsnowlabs.nlp.annotators.ld.dl.LanguageDetectorDL
import com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfModel
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetectorModel
import com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentModel
import com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel
import com.johnsnowlabs.nlp.annotators.seq2seq.{MarianTransformer, T5Transformer}
import com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerModel
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingModel
import com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteModel
import com.johnsnowlabs.nlp.annotators.ws.WordSegmenterModel
import com.johnsnowlabs.nlp.embeddings.{AlbertEmbeddings, BertEmbeddings, BertSentenceEmbeddings, DistilBertEmbeddings, ElmoEmbeddings, RoBertaEmbeddings, UniversalSentenceEncoder, WordEmbeddingsModel, XlmRoBertaEmbeddings, XlnetEmbeddings}
import com.johnsnowlabs.nlp.pretrained.ResourceDownloader.{listPretrainedResources, publicLoc, showString}
import com.johnsnowlabs.nlp.pretrained.ResourceType.ResourceType
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.{Build, ConfigHelper, FileHelper, Version}
import org.apache.spark.ml.util.DefaultParamsReadable
import org.apache.spark.ml.{PipelineModel, PipelineStage}

import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.util.{Failure, Success}
import com.amazonaws.AmazonClientException
import com.amazonaws.auth.profile.ProfileCredentialsProvider
import com.amazonaws.auth.{DefaultAWSCredentialsProviderChain, _}
import org.apache.hadoop.fs.FileSystem


trait ResourceDownloader {

  /**
   * Download resource to local file
   *
   * @param request Resource request
   * @return downloaded file or None if resource is not found
   */
  def download(request: ResourceRequest): Option[String]

  def getDownloadSize(request: ResourceRequest): Option[Long]

  def clearCache(request: ResourceRequest): Unit

  def downloadMetadataIfNeed(folder: String): List[ResourceMetadata]

  val fs: FileSystem = ResourceDownloader.fs

}

object ResourceDownloader {

  val fs: FileSystem = FileSystem.get(ResourceHelper.spark.sparkContext.hadoopConfiguration)

  def s3Bucket: String = ConfigHelper.getConfigValueOrElse(ConfigHelper.pretrainedS3BucketKey, "auxdata.johnsnowlabs.com")

  def s3BucketCommunity: String = ConfigHelper.getConfigValueOrElse(ConfigHelper.pretrainedCommunityS3BucketKey, "community.johnsnowlabs.com")

  def s3Path: String = ConfigHelper.getConfigValueOrElse(ConfigHelper.pretrainedS3PathKey, "")

  def cacheFolder: String = ConfigHelper.getConfigValueOrElse(ConfigHelper.pretrainedCacheFolder, fs.getHomeDirectory + "/cache_pretrained")

  def credentials: Option[AWSCredentials] = if (ConfigHelper.hasPath(ConfigHelper.awsCredentials)) {
    val accessKeyId = ConfigHelper.getConfigValue(ConfigHelper.accessKeyId)
    val secretAccessKey = ConfigHelper.getConfigValue(ConfigHelper.secretAccessKey)
    val awsProfile = ConfigHelper.getConfigValue(ConfigHelper.awsProfileName)
    if (awsProfile.isDefined) {
      return Some(new ProfileCredentialsProvider(awsProfile.get).getCredentials)
    }
    if (accessKeyId.isEmpty || secretAccessKey.isEmpty) {
      fetchcredentials()
    }
    else
      Some(new BasicAWSCredentials(accessKeyId.get, secretAccessKey.get))
  }
  else {
    fetchcredentials()
  }

  private def fetchcredentials(): Option[AWSCredentials] = {
    try {
      //check if default profile name works if not try 
      Some(new ProfileCredentialsProvider("spark_nlp").getCredentials)
    } catch {
      case _: Exception =>
        try {

          Some(new DefaultAWSCredentialsProviderChain().getCredentials)
        } catch {
          case _: AmazonClientException =>
            if (ResourceHelper.spark.sparkContext.hadoopConfiguration.get("fs.s3a.access.key") != null) {

              val key = ResourceHelper.spark.sparkContext.hadoopConfiguration.get("fs.s3a.access.key")
              val secret = ResourceHelper.spark.sparkContext.hadoopConfiguration.get("fs.s3a.secret.key")

              Some(new BasicAWSCredentials(key, secret))
            } else {
              Some(new AnonymousAWSCredentials())
            }
          case e: Exception => throw e

        }
    }

  }

  val publicLoc = "public/models"

  private val cache = mutable.Map[ResourceRequest, PipelineStage]()

  lazy val sparkVersion: Version = {
    val spark_version = if (ResourceHelper.spark.version.startsWith("2.3")) "2.4.4" else ResourceHelper.spark.version
    Version.parse(spark_version)
  }

  lazy val libVersion: Version = {
    Version.parse(Build.version)
  }

  var defaultDownloader: ResourceDownloader = new S3ResourceDownloader(s3Bucket, s3Path, cacheFolder, credentials)
  var publicDownloader: ResourceDownloader = new S3ResourceDownloader(s3Bucket, s3Path, cacheFolder, Some(new AnonymousAWSCredentials()))
  var communityDownloader: ResourceDownloader = new S3ResourceDownloader(s3BucketCommunity, s3Path, cacheFolder, Some(new AnonymousAWSCredentials()))

  /**
   * Reset the cache and recreate ResourceDownloader S3 credentials
   */
  def resetResourceDownloader(): Unit = {
    cache.empty
    this.defaultDownloader = new S3ResourceDownloader(s3Bucket, s3Path, cacheFolder, credentials)
  }

  /**
   * List all pretrained models in public name_lang
   */
  def listPublicModels(): List[String] = {
    listPretrainedResources(folder = publicLoc, ResourceType.MODEL)
  }


  def showPublicModels(lang: String): Unit = {
    println(showString(listPretrainedResources(folder = publicLoc, ResourceType.MODEL, lang), ResourceType.MODEL))
  }

  def showPublicModels(lang: String, version: String): Unit = {
    println(showString(listPretrainedResources(folder = publicLoc, ResourceType.MODEL, lang, Version.parse(version)), ResourceType.MODEL))
  }

  /**
   * List all pretrained pipelines in public
   */
  def listPublicPipelines(): List[String] = {
    listPretrainedResources(folder = publicLoc, ResourceType.PIPELINE)
  }


  def showPublicPipelines(lang: String): Unit = {
    println(showString(listPretrainedResources(folder = publicLoc, ResourceType.PIPELINE, lang), ResourceType.PIPELINE))
  }

  def showPublicPipelines(lang: String, version: String): Unit = {
    println(showString(listPretrainedResources(folder = publicLoc, ResourceType.PIPELINE, lang, Version.parse(version)), ResourceType.PIPELINE))
  }

  /**
   * Returns models or pipelines in metadata json which has not been categorized yet.
   *
   * @return list of models or pipelines which are not categorized in metadata json
   */
  def listUnCategorizedResources(): List[String] = {
    listPretrainedResources(folder = publicLoc, ResourceType.NOT_DEFINED)
  }


  def showUnCategorizedResources(lang: String): Unit = {
    println(showString(listPretrainedResources(folder = publicLoc, ResourceType.NOT_DEFINED, lang), ResourceType.NOT_DEFINED))
  }

  def showUnCategorizedResources(lang: String, version: String): Unit = {
    println(showString(listPretrainedResources(folder = publicLoc, ResourceType.NOT_DEFINED, lang, Version.parse(version)), ResourceType.NOT_DEFINED))
  }

  def showString(list: List[String], resourceType: ResourceType): String = {
    val sb = new StringBuilder
    var max_length = 14
    var max_length_version = 7
    for (data <- list) {
      val temp = data.split(":")
      max_length = scala.math.max(temp(0).length, max_length)
      max_length_version = scala.math.max(temp(2).length, max_length_version)
    }
    //adding head
    sb.append("+")
    sb.append("-" * (max_length + 2))
    sb.append("+")
    sb.append("-" * 6)
    sb.append("+")
    sb.append("-" * (max_length_version + 2))
    sb.append("+\n")
    if (resourceType.equals(ResourceType.PIPELINE))
      sb.append("| " + "Pipeline" + (" " * (max_length - 8)) + " | " + "lang" + " | " + "version" + " " * (max_length_version - 7) + " |\n")
    else if (resourceType.equals(ResourceType.MODEL))
      sb.append("| " + "Model" + (" " * (max_length - 5)) + " | " + "lang" + " | " + "version" + " " * (max_length_version - 7) + " |\n")
    else
      sb.append("| " + "Pipeline/Model" + (" " * (max_length - 14)) + " | " + "lang" + " | " + "version" + " " * (max_length_version - 7) + " |\n")

    sb.append("+")
    sb.append("-" * (max_length + 2))
    sb.append("+")
    sb.append("-" * 6)
    sb.append("+")
    sb.append("-" * (max_length_version + 2))
    sb.append("+\n")
    for (data <- list) {
      val temp = data.split(":")
      sb.append("| " + temp(0) + (" " * (max_length - temp(0).length)) + " |  " + temp(1) + "  | " + temp(2) + " " * (max_length_version - temp(2).length) + " |\n")
    }
    //adding bottom
    sb.append("+")
    sb.append("-" * (max_length + 2))
    sb.append("+")
    sb.append("-" * 6)
    sb.append("+")
    sb.append("-" * (max_length_version + 2))
    sb.append("+\n")
    sb.toString()

  }

  /**
   * List all resources after parsing the metadata json from the given folder in the S3 location
   *
   * @param folder       folder inside S3 bucket
   * @param resourceType type of the resources, ml for models and pl for pipelines
   * @return list of pipelines if resourceType is Pipeline or list of models if resourceType is Model
   */
  def listPretrainedResources(folder: String, resourceType: ResourceType): List[String] = {
    val resourceList = new ListBuffer[String]()
    val resourceMetaData = defaultDownloader.downloadMetadataIfNeed(folder)
    for (meta <- resourceMetaData) {
      if (meta.category.getOrElse(ResourceType.NOT_DEFINED).toString.equals(resourceType.toString)) {
        resourceList += meta.name + ":" + meta.language.getOrElse("-") + ":" + meta.libVersion.getOrElse("-")
      }

    }
    resourceList.result()
  }

  def listPretrainedResources(folder: String, resourceType: ResourceType, lang: String): List[String] = {
    val resourceList = new ListBuffer[String]()
    val resourceMetaData = defaultDownloader.downloadMetadataIfNeed(folder)
    for (meta <- resourceMetaData) {
      if (meta.category.getOrElse(ResourceType.NOT_DEFINED).toString.equals(resourceType.toString) & meta.language.getOrElse("").equalsIgnoreCase(lang)) {
        resourceList += meta.name + ":" + meta.language.getOrElse("-") + ":" + meta.libVersion.getOrElse("-")
      }

    }
    resourceList.result()
  }

  def listPretrainedResources(folder: String, resourceType: ResourceType, lang: String, version: Version): List[String] = {
    val resourceList = new ListBuffer[String]()
    val resourceMetaData = defaultDownloader.downloadMetadataIfNeed(folder)
    for (meta <- resourceMetaData) {

      if (meta.category.getOrElse(ResourceType.NOT_DEFINED).toString.equals(resourceType.toString) & meta.language.getOrElse("").equalsIgnoreCase(lang) & Version.isCompatible(version, meta.libVersion)) {
        resourceList += meta.name + ":" + meta.language.getOrElse("-") + ":" + meta.libVersion.getOrElse("-")
      }

    }
    resourceList.result()
  }

  def listPretrainedResources(folder: String, resourceType: ResourceType, version: Version): List[String] = {
    val resourceList = new ListBuffer[String]()
    val resourceMetaData = defaultDownloader.downloadMetadataIfNeed(folder)
    for (meta <- resourceMetaData) {

      if (meta.category.getOrElse(ResourceType.NOT_DEFINED).toString.equals(resourceType.toString) & Version.isCompatible(version, meta.libVersion)) {
        resourceList += meta.name + ":" + meta.language.getOrElse("-") + ":" + meta.libVersion.getOrElse("-")
      }

    }
    resourceList.result()
  }

  /**
   * Loads resource to path
   *
   * @param name     Name of Resource
   * @param folder   Subfolder in s3 where to search model (e.g. medicine)
   * @param language Desired language of Resource
   * @return path of downloaded resource
   */
  def downloadResource(name: String, language: Option[String] = None, folder: String = publicLoc): String = {
    downloadResource(ResourceRequest(name, language, folder))
  }

  /**
   * Loads resource to path
   *
   * @param request Request for resource
   * @return path of downloaded resource
   */
  def downloadResource(request: ResourceRequest): String = {
    val f = Future {
      if (request.folder.equals(publicLoc)) {
        publicDownloader.download(request)
      } else if (request.folder.startsWith("@")) {
        val actualLoc = request.folder.replace("@", "")
        val updatedRequest = ResourceRequest(request.name, request.language, folder = actualLoc, request.libVersion, request.sparkVersion)
        communityDownloader.download(updatedRequest)
      } else {
        defaultDownloader.download(request)
      }
    }

    var download_finished = false
    var path: Option[String] = None
    val file_size = getDownloadSize(request.name, request.language, request.folder)
    require(!file_size.equals("-1"), s"Can not find ${request.name} inside ${request.folder} to download. Please make sure the name and location are correct!")
    println(request.name + " download started this may take some time.")
    println("Approximate size to download " + file_size)

    var nextc = 0
    while (!download_finished) {
      nextc += 1
      f.onComplete {
        case Success(value) =>
          download_finished = true
          path = value
        case Failure(_) =>
          download_finished = true
          path = None
      }
      Thread.sleep(1000)

    }

    require(path.isDefined, s"Was not found appropriate resource to download for request: $request with downloader: $defaultDownloader")
    println("Download done! Loading the resource.")
    path.get
  }

  def downloadModel[TModel <: PipelineStage](reader: DefaultParamsReadable[TModel],
                                             name: String,
                                             language: Option[String] = None,
                                             folder: String = publicLoc
                                            ): TModel = {
    downloadModel(reader, ResourceRequest(name, language, folder))
  }

  def downloadModel[TModel <: PipelineStage](reader: DefaultParamsReadable[TModel], request: ResourceRequest): TModel = {
    if (!cache.contains(request)) {
      val path = downloadResource(request)
      val model = reader.read.load(path)
      cache(request) = model
      model
    }
    else {
      cache(request).asInstanceOf[TModel]
    }
  }

  def downloadPipeline(name: String, language: Option[String] = None, folder: String = publicLoc): PipelineModel = {
    downloadPipeline(ResourceRequest(name, language, folder))
  }

  def downloadPipeline(request: ResourceRequest): PipelineModel = {
    if (!cache.contains(request)) {
      val path = downloadResource(request)
      val model = PipelineModel.read.load(path)
      cache(request) = model
      model
    }
    else {
      cache(request).asInstanceOf[PipelineModel]
    }
  }

  def clearCache(name: String, language: Option[String] = None, folder: String = publicLoc): Unit = {
    clearCache(ResourceRequest(name, language, folder))
  }

  def clearCache(request: ResourceRequest): Unit = {
    defaultDownloader.clearCache(request)
    publicDownloader.clearCache(request)
    communityDownloader.clearCache(request)
    cache.remove(request)
  }

  def getDownloadSize(name: String, language: Option[String] = None, folder: String = publicLoc): String = {
    var size: Option[Long] = None
    if (folder.equals(publicLoc)) {
      size = publicDownloader.getDownloadSize(ResourceRequest(name, language, folder))
    } else if (folder.startsWith("@")) {
      val actualLoc = folder.replace("@", "")
      size = communityDownloader.getDownloadSize(ResourceRequest(name, language, actualLoc))
    } else {
      size = defaultDownloader.getDownloadSize(ResourceRequest(name, language, folder))
    }
    size match {
      case Some(downloadBytes) => FileHelper.getHumanReadableFileSize(downloadBytes)
      case None => "-1"


    }
  }
}

object ResourceType extends Enumeration {
  type ResourceType = Value
  val MODEL: pretrained.ResourceType.Value = Value("ml")
  val PIPELINE: pretrained.ResourceType.Value = Value("pl")
  val NOT_DEFINED: pretrained.ResourceType.Value = Value("nd")
}

case class ResourceRequest
(
  name: String,
  language: Option[String] = None,
  folder: String = ResourceDownloader.publicLoc,
  libVersion: Version = ResourceDownloader.libVersion,
  sparkVersion: Version = ResourceDownloader.sparkVersion
)

/* convenience accessor for Py4J calls */
object PythonResourceDownloader {

  val keyToReader: mutable.Map[String, DefaultParamsReadable[_]] = mutable.Map(
    "DocumentAssembler" -> DocumentAssembler,
    "SentenceDetector" -> SentenceDetector,
    "TokenizerModel" -> TokenizerModel,
    "PerceptronModel" -> PerceptronModel,
    "NerCrfModel" -> NerCrfModel,
    "Stemmer" -> Stemmer,
    "NormalizerModel" -> NormalizerModel,
    "RegexMatcherModel" -> RegexMatcherModel,
    "LemmatizerModel" -> LemmatizerModel,
    "DateMatcher" -> DateMatcher,
    "TextMatcherModel" -> TextMatcherModel,
    "SentimentDetectorModel" -> SentimentDetectorModel,
    "ViveknSentimentModel" -> ViveknSentimentModel,
    "NorvigSweetingModel" -> NorvigSweetingModel,
    "SymmetricDeleteModel" -> SymmetricDeleteModel,
    "NerDLModel" -> NerDLModel,
    "WordEmbeddingsModel" -> WordEmbeddingsModel,
    "BertEmbeddings" -> BertEmbeddings,
    "DependencyParserModel" -> DependencyParserModel,
    "TypedDependencyParserModel" -> TypedDependencyParserModel,
    "UniversalSentenceEncoder" -> UniversalSentenceEncoder,
    "ElmoEmbeddings" -> ElmoEmbeddings,
    "ClassifierDLModel" -> ClassifierDLModel,
    "ContextSpellCheckerModel" -> ContextSpellCheckerModel,
    "AlbertEmbeddings" -> AlbertEmbeddings,
    "XlnetEmbeddings" -> XlnetEmbeddings,
    "SentimentDLModel" -> SentimentDLModel,
    "LanguageDetectorDL" -> LanguageDetectorDL,
    "StopWordsCleaner" -> StopWordsCleaner,
    "BertSentenceEmbeddings" -> BertSentenceEmbeddings,
    "MultiClassifierDLModel" -> MultiClassifierDLModel,
    "SentenceDetectorDLModel" -> SentenceDetectorDLModel,
    "T5Transformer" -> T5Transformer,
    "MarianTransformer" -> MarianTransformer,
    "WordSegmenterModel" -> WordSegmenterModel,
    "DistilBertEmbeddings" -> DistilBertEmbeddings,
    "RoBertaEmbeddings" -> RoBertaEmbeddings,
    "XlmRoBertaEmbeddings" -> XlmRoBertaEmbeddings
  )

  def downloadModel(readerStr: String, name: String, language: String = null, remoteLoc: String = null): PipelineStage = {
    val reader = keyToReader.getOrElse(readerStr, throw new RuntimeException(s"Unsupported Model: $readerStr"))
    val correctedFolder = Option(remoteLoc).getOrElse(ResourceDownloader.publicLoc)
    ResourceDownloader.downloadModel(reader.asInstanceOf[DefaultParamsReadable[PipelineStage]], name, Option(language), correctedFolder)
  }

  def downloadPipeline(name: String, language: String = null, remoteLoc: String = null): PipelineModel = {
    val correctedFolder = Option(remoteLoc).getOrElse(ResourceDownloader.publicLoc)
    ResourceDownloader.downloadPipeline(name, Option(language), correctedFolder)
  }

  def clearCache(name: String, language: String = null, remoteLoc: String = null): Unit = {
    val correctedFolder = Option(remoteLoc).getOrElse(ResourceDownloader.publicLoc)
    ResourceDownloader.clearCache(name, Option(language), correctedFolder)
  }

  def showUnCategorizedResources(): Unit = {
    println(showString(listPretrainedResources(folder = publicLoc, ResourceType.NOT_DEFINED), ResourceType.NOT_DEFINED))
  }

  def showPublicPipelines(): Unit = {
    println(showString(listPretrainedResources(folder = publicLoc, ResourceType.PIPELINE), ResourceType.PIPELINE))
  }

  def showPublicModels(): Unit = {
    println(showString(listPretrainedResources(folder = publicLoc, ResourceType.MODEL), ResourceType.MODEL))
  }

  def getDownloadSize(name: String, language: String = "en", remoteLoc: String = null): String = {
    val correctedFolder = Option(remoteLoc).getOrElse(ResourceDownloader.publicLoc)
    ResourceDownloader.getDownloadSize(name, Option(language), correctedFolder)
  }
}

