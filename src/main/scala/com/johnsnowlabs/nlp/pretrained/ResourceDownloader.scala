package com.johnsnowlabs.nlp.pretrained

import com.amazonaws.auth.{AWSCredentials, AnonymousAWSCredentials, BasicAWSCredentials}
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfModel
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.{Build, ConfigHelper, Version}
import org.apache.spark.ml.{PipelineModel, PipelineStage}
import org.apache.spark.ml.util.DefaultParamsReadable
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetectorModel
import com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentModel
import com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerModel
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingModel
import org.apache.hadoop.fs.FileSystem

import scala.collection.mutable


trait ResourceDownloader {

  /**
    * Download resource to local file
    * @param request      Resource request
    * @return             downloaded file or None if resource is not found
    */
  def download(request: ResourceRequest): Option[String]

  def clearCache(request: ResourceRequest): Unit

  val fs = ResourceDownloader.fs

}

object ResourceDownloader {

  val fs = FileSystem.get(ResourceHelper.spark.sparkContext.hadoopConfiguration)

  def s3Bucket = ConfigHelper.getConfigValueOrElse(ConfigHelper.pretrainedS3BucketKey, "auxdata.johnsnowlabs.com")
  def s3Path = ConfigHelper.getConfigValueOrElse(ConfigHelper.pretrainedS3PathKey, "")
  def cacheFolder = ConfigHelper.getConfigValueOrElse(ConfigHelper.pretrainedCacheFolder, fs.getHomeDirectory + "/cache_pretrained")

  def credentials: Option[AWSCredentials] = if (ConfigHelper.hasPath(ConfigHelper.awsCredentials)) {
    val accessKeyId = ConfigHelper.getConfigValue(ConfigHelper.accessKeyId)
    val secretAccessKey = ConfigHelper.getConfigValue(ConfigHelper.secretAccessKey)
    if (accessKeyId.isEmpty || secretAccessKey.isEmpty)
      Some(new AnonymousAWSCredentials())
    else
      Some(new BasicAWSCredentials(accessKeyId.get, secretAccessKey.get))
    }
  else {
    None
  }


  val publicLoc = "public/models"

  private val cache = mutable.Map[ResourceRequest, PipelineStage]()

  lazy val sparkVersion: Version = {
    Version.parse(ResourceHelper.spark.version)
  }

  lazy val libVersion: Version = {
    Version.parse(Build.version)
  }

  var defaultDownloader: ResourceDownloader = new S3ResourceDownloader(s3Bucket, s3Path, cacheFolder, credentials)

  /**
    * Reset the cache and recreate ResourceDownloader S3 credentials
    */
  def resetResourceDownloader(): Unit ={
    cache.empty
    this.defaultDownloader = new S3ResourceDownloader(s3Bucket, s3Path, cacheFolder, credentials)
  }

  /**
    * Loads resource to path
    * @param name Name of Resource
    * @param folder Subfolder in s3 where to search model (e.g. medicine)
    * @param language Desired language of Resource
    * @return path of downloaded resource
    */
  def downloadResource(name: String, language: Option[String] = None, folder: String = publicLoc): String = {
    downloadResource(ResourceRequest(name, language, folder))
  }

  /**
    * Loads resource to path
    * @param request Request for resource
    * @return path of downloaded resource
    */
  def downloadResource(request: ResourceRequest): String = {
    val path = defaultDownloader.download(request)
    println("path:"+ path)
    require(path.isDefined, s"Was not found appropriate resource to download for request: $request with downloader: $defaultDownloader")

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
    cache.remove(request)
  }
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

  val keyToReader : Map[String, DefaultParamsReadable[_]] = Map(
    "DocumentAssembler" -> DocumentAssembler,
    "SentenceDetector" -> SentenceDetector,
    "Tokenizer" -> Tokenizer,
    "PerceptronModel" -> PerceptronModel,
    "NerCrfModel" -> NerCrfModel,
    "Stemmer" -> Stemmer,
    "Normalizer" -> Normalizer,
    "RegexMatcherModel" -> RegexMatcherModel,
    "LemmatizerModel" -> LemmatizerModel,
    "DateMatcher" -> DateMatcher,
    "TextMatcherModel" -> TextMatcherModel,
    "SentimentDetectorModel" -> SentimentDetectorModel,
    "ViveknSentimentModel" -> ViveknSentimentModel,
    "NorvigSweetingModel" -> NorvigSweetingModel,
    "NerDLModel" -> NerDLModel,
    "ContextSpellCheckerModel" -> ContextSpellCheckerModel
    )

  def downloadModel(readerStr: String, name: String, language: String = null, remoteLoc: String  = null): PipelineStage = {
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
}

