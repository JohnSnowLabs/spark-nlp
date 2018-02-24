package com.johnsnowlabs.pretrained

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.{Build, ConfigHelper, Version}
import org.apache.spark.ml.{PipelineModel, PipelineStage}
import org.apache.spark.ml.util.DefaultParamsReadable
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector

import scala.collection.mutable


trait ResourceDownloader {
  
  /**
    * Download resource to local file
    *
    * @param name         Resource Name. ner_fast for example
    * @param language     Language of the model
    * @param libVersion   spark-nlp library version
    * @param sparkVersion spark library version
    * @return             downloaded file or None if resource is not found
    */
  def download(name: String,
                        language: Option[String],
                        libVersion: Version,
                        sparkVersion: Version): Option[String]

  def clearCache(name: String,
                 language: Option[String],
                 libVersion: Version,
                 sparkVersion: Version): Unit
}

object ResourceDownloader {

  val s3Bucket = ConfigHelper.getConfigValueOrElse(ConfigHelper.pretrainedS3BucketKey, "data.johnsnowlabs.com")
  val s3Path = ConfigHelper.getConfigValueOrElse(ConfigHelper.pretrainedS3PathKey, "models/spark-nlp-public")
  val cacheFolder = ConfigHelper.getConfigValueOrElse(ConfigHelper.pretrainedCacheFolder, "cache_pretrained")

  private val cache = mutable.Map[ResourceRequest, PipelineStage]()

  lazy val sparkVersion: Version = {
    Version.parse(ResourceHelper.spark.version)
  }

  lazy val libVersion: Version = {
    Version.parse(Build.version)
  }

  var defaultDownloader: ResourceDownloader = new S3ResourceDownloader(s3Bucket, s3Path, cacheFolder)

  /**
    * Loads resource to path
    * @param name Name of Resource
    * @param language Desired language of Resource
    * @return path of downloaded resource
    */
  def downloadResource(name: String, language: Option[String]): String = {
    val path = defaultDownloader.download(name, language, libVersion, sparkVersion)
    require(path.isDefined, s"Was not able to download: $name for language: " +
      s"$language, with libVersion: $libVersion and spark version: $sparkVersion " +
      s"with downlader: $defaultDownloader")

    path.get
  }

  def downloadModel[TModel <: PipelineStage](reader: DefaultParamsReadable[TModel], name: String, language: Option[String]): TModel = {
    val key = ResourceRequest(name, language)

    if (!cache.contains(key)) {
      val path = downloadResource(name, language)
      val model = reader.read.load(path)
      cache(key) = model
      model
    }
    else {
      cache(key).asInstanceOf[TModel]
    }
  }

  def downloadPipeline(name: String, language: Option[String]): PipelineModel = {
    val key = ResourceRequest(name, language)

    if (!cache.contains(key)) {
      val path = downloadResource(name, language)
      val model = PipelineModel.read.load(path)
      cache(key) = model
      model
    }
    else {
      cache(key).asInstanceOf[PipelineModel]
    }
  }

  def clearCache(name: String, language: Option[String]): Unit = {
    defaultDownloader.clearCache(name, language, libVersion, sparkVersion)
    cache.remove(ResourceRequest(name, language))
  }

  case class ResourceRequest(name: String, language: Option[String])

}

/* convenience accessor for Py4J calls */
object PythonResourceDownloader {

  //TODO: fill this!

  val keyToReader : Map[String, DefaultParamsReadable[_]] = Map(
    "documentAssembler" -> DocumentAssembler,
    "sentenceDetector" -> SentenceDetector
    )

  def downloadModel(readerStr: String, name: String, language: String) = {
    val reader = keyToReader.get(readerStr).getOrElse(throw new RuntimeException("Unsupported Model."))
    ResourceDownloader.downloadModel(reader.asInstanceOf[DefaultParamsReadable[PipelineStage]], name, Some(language))
  }


  def downloadPipeline(name: String, language: String):PipelineModel =
    ResourceDownloader.downloadPipeline(name, Some(language))

}

