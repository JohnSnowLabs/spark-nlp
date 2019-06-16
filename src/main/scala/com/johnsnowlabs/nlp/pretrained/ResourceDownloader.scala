package com.johnsnowlabs.nlp.pretrained

import com.amazonaws.auth.{AWSCredentials, AnonymousAWSCredentials, BasicAWSCredentials}
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfModel
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
import com.johnsnowlabs.nlp.annotators.parser.dep.DependencyParserModel
import com.johnsnowlabs.nlp.annotators.parser.typdep.TypedDependencyParserModel
import com.johnsnowlabs.nlp.annotators.pos.perceptron.PerceptronModel
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.annotators.sda.pragmatic.SentimentDetectorModel
import com.johnsnowlabs.nlp.annotators.sda.vivekn.ViveknSentimentModel
import com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerModel
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingModel
import com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteModel
import com.johnsnowlabs.nlp.embeddings.{BertEmbeddings, WordEmbeddingsModel}
import com.johnsnowlabs.nlp.pretrained.ResourceType.ResourceType
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.{Build, ConfigHelper, Version}
import org.apache.hadoop.fs.FileSystem
import org.apache.spark.ml.util.DefaultParamsReadable
import org.apache.spark.ml.{PipelineModel, PipelineStage}

import scala.collection.mutable.{ListBuffer, Map}



trait ResourceDownloader {

  /**
    * Download resource to local file
    * @param request      Resource request
    * @return             downloaded file or None if resource is not found
    */
  def download(request: ResourceRequest): Option[String]

  def clearCache(request: ResourceRequest): Unit

  def downloadMetadataIfNeed(folder: String): List[ResourceMetadata]
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

  private val cache = Map[ResourceRequest, PipelineStage]()

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
    * List all pretrained models in public
    */
  def listPublicModels(): List[String] = {
    listPretrainedResources(folder = publicLoc, ResourceType.MODEL)
  }

  /**
    * List all pretrained pipelines in public
    */
  def listPublicPipelines(): List[String] = {
    listPretrainedResources(folder = publicLoc, ResourceType.PIPELINE)
  }

  /**
    * Returns models or pipelines in metadata json which has not been categorized yet.
    *
    * @return list of models or piplelines which are not categorized in metadata json
    */
  def listUnCategoriedResources(): List[String] = {
    listPretrainedResources(folder = publicLoc, ResourceType.NOT_DEFINED)
  }

  /**
    * List all resources after parsing the metadata json from the given folder in the S3 location
    *
    * @param folder
    * @param resourceType
    * @return list of pipelines if resourceType is Pipeline or list of models if resourceType is Model
    */
  def listPretrainedResources(folder: String, resourceType: ResourceType): List[String] = {
    val resourceList = new ListBuffer[String]()
    val resourceMetaData = defaultDownloader.downloadMetadataIfNeed(folder)
    for (meta <- resourceMetaData) {
      if (meta.category.getOrElse(ResourceType.NOT_DEFINED).toString.equals(resourceType.toString)) {
        resourceList += meta.name + "_" + meta.language.getOrElse("no_lang")
      }

    }
    resourceList.result()
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

object ResourceType extends Enumeration {
  type ResourceType = Value
  val MODEL = Value("ml")
  val PIPELINE = Value("pl")
  val NOT_DEFINED = Value("nd")
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
    "SymmetricDeleteModel" -> SymmetricDeleteModel,
    "NerDLModel" -> NerDLModel,
    "ContextSpellCheckerModel" -> ContextSpellCheckerModel,
    "WordEmbeddingsModel" -> WordEmbeddingsModel,
    "BertEmbeddings" -> BertEmbeddings,
    "DependencyParserModel" -> DependencyParserModel,
    "TypedDependencyParserModel" -> TypedDependencyParserModel
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

