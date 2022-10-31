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

package com.johnsnowlabs.nlp.pretrained

import com.johnsnowlabs.client.aws.AWSGateway
import com.johnsnowlabs.nlp.annotators._
import com.johnsnowlabs.nlp.annotators.audio.Wav2Vec2ForCTC
import com.johnsnowlabs.nlp.annotators.classifier.dl._
import com.johnsnowlabs.nlp.annotators.coref.SpanBertCorefModel
import com.johnsnowlabs.nlp.annotators.cv.ViTForImageClassification
import com.johnsnowlabs.nlp.annotators.er.EntityRulerModel
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
import com.johnsnowlabs.nlp.annotators.seq2seq.{GPT2Transformer, MarianTransformer, T5Transformer}
import com.johnsnowlabs.nlp.annotators.spell.context.ContextSpellCheckerModel
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingModel
import com.johnsnowlabs.nlp.annotators.spell.symmetric.SymmetricDeleteModel
import com.johnsnowlabs.nlp.annotators.ws.WordSegmenterModel
import com.johnsnowlabs.nlp.embeddings._
import com.johnsnowlabs.nlp.pretrained.ResourceType.ResourceType
import com.johnsnowlabs.nlp.util.io.{OutputHelper, ResourceHelper}
import com.johnsnowlabs.nlp.{DocumentAssembler, TableAssembler, pretrained}
import com.johnsnowlabs.util._
import org.apache.hadoop.fs.FileSystem
import org.apache.spark.SparkFiles
import org.apache.spark.ml.util.DefaultParamsReadable
import org.apache.spark.ml.{PipelineModel, PipelineStage}
import org.slf4j.LoggerFactory

import java.io.File
import java.net.URI
import java.nio.file.Paths
import scala.collection.mutable
import scala.collection.mutable.ListBuffer
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.Future
import scala.util.{Failure, Success}

trait ResourceDownloader {

  /** Download resource to local file
    *
    * @param request
    *   Resource request
    * @return
    *   downloaded file or None if resource is not found
    */
  def download(request: ResourceRequest): Option[String]

  def getDownloadSize(request: ResourceRequest): Option[Long]

  def clearCache(request: ResourceRequest): Unit

  def downloadMetadataIfNeed(folder: String): List[ResourceMetadata]

  def downloadAndUnzipFile(s3FilePath: String): Option[String]

  val fileSystem: FileSystem = ResourceDownloader.fileSystem

}

object ResourceDownloader {

  private val logger = LoggerFactory.getLogger(this.getClass.toString)

  val fileSystem: FileSystem = OutputHelper.getFileSystem

  def s3Bucket: String = ConfigLoader.getConfigStringValue(ConfigHelper.pretrainedS3BucketKey)

  def s3BucketCommunity: String =
    ConfigLoader.getConfigStringValue(ConfigHelper.pretrainedCommunityS3BucketKey)

  def s3Path: String = ConfigLoader.getConfigStringValue(ConfigHelper.pretrainedS3PathKey)

  def cacheFolder: String = ConfigLoader.getConfigStringValue(ConfigHelper.pretrainedCacheFolder)

  val publicLoc = "public/models"

  private val cache = mutable.Map[ResourceRequest, PipelineStage]()

  lazy val sparkVersion: Version = {
    val spark_version = ResourceHelper.spark.version
    Version.parse(spark_version)
  }

  lazy val libVersion: Version = {
    Version.parse(Build.version)
  }

  var defaultDownloader: ResourceDownloader =
    new S3ResourceDownloader(s3Bucket, s3Path, cacheFolder, "default")
  var publicDownloader: ResourceDownloader =
    new S3ResourceDownloader(s3Bucket, s3Path, cacheFolder, "public")
  var communityDownloader: ResourceDownloader =
    new S3ResourceDownloader(s3BucketCommunity, s3Path, cacheFolder, "community")

  /** Reset the cache and recreate ResourceDownloader S3 credentials */
  def resetResourceDownloader(): Unit = {
    cache.empty
    this.defaultDownloader = new S3ResourceDownloader(s3Bucket, s3Path, cacheFolder, "default")
  }

  /** List all pretrained models in public name_lang */
  def listPublicModels(): List[String] = {
    listPretrainedResources(folder = publicLoc, ResourceType.MODEL)
  }

  /** Prints all pretrained models for a particular annotator model, that are compatible with a
    * version of Spark NLP. If any of the optional arguments are not set, the filter is not
    * considered.
    *
    * @param annotator
    *   Name of the model class, for example "NerDLModel"
    * @param lang
    *   Language of the pretrained models to display, for example "en"
    * @param version
    *   Version of Spark NLP that the model should be compatible with, for example "3.2.3"
    */
  def showPublicModels(
      annotator: Option[String] = None,
      lang: Option[String] = None,
      version: Option[String] = Some(Build.version)): Unit = {
    println(
      publicResourceString(
        annotator = annotator,
        lang = lang,
        version = version,
        resourceType = ResourceType.MODEL))
  }

  /** Prints all pretrained models for a particular annotator model, that are compatible with this
    * version of Spark NLP.
    *
    * @param annotator
    *   Name of the annotator class
    */
  def showPublicModels(annotator: String): Unit = showPublicModels(Some(annotator))

  /** Prints all pretrained models for a particular annotator model, that are compatible with this
    * version of Spark NLP.
    *
    * @param annotator
    *   Name of the annotator class
    * @param lang
    *   Language of the pretrained models to display
    */
  def showPublicModels(annotator: String, lang: String): Unit =
    showPublicModels(Some(annotator), Some(lang))

  /** Prints all pretrained models for a particular annotator, that are compatible with a version
    * of Spark NLP.
    *
    * @param annotator
    *   Name of the model class, for example "NerDLModel"
    * @param lang
    *   Language of the pretrained models to display, for example "en"
    * @param version
    *   Version of Spark NLP that the model should be compatible with, for example "3.2.3"
    */
  def showPublicModels(annotator: String, lang: String, version: String): Unit =
    showPublicModels(Some(annotator), Some(lang), Some(version))

  /** List all pretrained pipelines in public */
  def listPublicPipelines(): List[String] = {
    listPretrainedResources(folder = publicLoc, ResourceType.PIPELINE)
  }

  /** Prints all Pipelines available for a language and a version of Spark NLP. By default shows
    * all languages and uses the current version of Spark NLP.
    *
    * @param lang
    *   Language of the Pipeline
    * @param version
    *   Version of Spark NLP
    */
  def showPublicPipelines(
      lang: Option[String] = None,
      version: Option[String] = Some(Build.version)): Unit = {
    println(
      publicResourceString(
        annotator = None,
        lang = lang,
        version = version,
        resourceType = ResourceType.PIPELINE))
  }

  /** Prints all Pipelines available for a language and this version of Spark NLP.
    *
    * @param lang
    *   Language of the Pipeline
    */
  def showPublicPipelines(lang: String): Unit = showPublicPipelines(Some(lang))

  /** Prints all Pipelines available for a language and a version of Spark NLP.
    *
    * @param lang
    *   Language of the Pipeline
    * @param version
    *   Version of Spark NLP
    */
  def showPublicPipelines(lang: String, version: String): Unit =
    showPublicPipelines(Some(lang), Some(version))

  /** Returns models or pipelines in metadata json which has not been categorized yet.
    *
    * @return
    *   list of models or pipelines which are not categorized in metadata json
    */
  def listUnCategorizedResources(): List[String] = {
    listPretrainedResources(folder = publicLoc, ResourceType.NOT_DEFINED)
  }

  def showUnCategorizedResources(lang: String): Unit = {
    println(publicResourceString(None, Some(lang), None, resourceType = ResourceType.NOT_DEFINED))
  }

  def showUnCategorizedResources(lang: String, version: String): Unit = {
    println(
      publicResourceString(
        None,
        Some(lang),
        Some(version),
        resourceType = ResourceType.NOT_DEFINED))

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
    // adding head
    sb.append("+")
    sb.append("-" * (max_length + 2))
    sb.append("+")
    sb.append("-" * 6)
    sb.append("+")
    sb.append("-" * (max_length_version + 2))
    sb.append("+\n")
    if (resourceType.equals(ResourceType.PIPELINE))
      sb.append(
        "| " + "Pipeline" + (" " * (max_length - 8)) + " | " + "lang" + " | " + "version" + " " * (max_length_version - 7) + " |\n")
    else if (resourceType.equals(ResourceType.MODEL))
      sb.append(
        "| " + "Model" + (" " * (max_length - 5)) + " | " + "lang" + " | " + "version" + " " * (max_length_version - 7) + " |\n")
    else
      sb.append(
        "| " + "Pipeline/Model" + (" " * (max_length - 14)) + " | " + "lang" + " | " + "version" + " " * (max_length_version - 7) + " |\n")

    sb.append("+")
    sb.append("-" * (max_length + 2))
    sb.append("+")
    sb.append("-" * 6)
    sb.append("+")
    sb.append("-" * (max_length_version + 2))
    sb.append("+\n")
    for (data <- list) {
      val temp = data.split(":")
      sb.append(
        "| " + temp(0) + (" " * (max_length - temp(0).length)) + " |  " + temp(1) + "  | " + temp(
          2) + " " * (max_length_version - temp(2).length) + " |\n")
    }
    // adding bottom
    sb.append("+")
    sb.append("-" * (max_length + 2))
    sb.append("+")
    sb.append("-" * 6)
    sb.append("+")
    sb.append("-" * (max_length_version + 2))
    sb.append("+\n")
    sb.toString()

  }

  def publicResourceString(
      annotator: Option[String] = None,
      lang: Option[String] = None,
      version: Option[String] = Some(Build.version),
      resourceType: ResourceType): String = {
    showString(
      listPretrainedResources(
        folder = publicLoc,
        resourceType,
        annotator = annotator,
        lang = lang,
        version = version match {
          case Some(ver) => Some(Version.parse(ver))
          case None => None
        }),
      resourceType)
  }

  /** Lists pretrained resource from metadata.json, depending on the set filters. The folder in
    * the S3 location and the resourceType is necessary. The other filters are optional and will
    * be ignored if not set.
    *
    * @param folder
    *   Folder in the S3 location
    * @param resourceType
    *   Type of the Resource. Can Either `ResourceType.MODEL`, `ResourceType.PIPELINE` or
    *   `ResourceType.NOT_DEFINED`
    * @param annotator
    *   Name of the model class
    * @param lang
    *   Language of the model
    * @param version
    *   Version that the model should be compatible with
    * @return
    *   A list of the available resources
    */
  def listPretrainedResources(
      folder: String,
      resourceType: ResourceType,
      annotator: Option[String] = None,
      lang: Option[String] = None,
      version: Option[Version] = None): List[String] = {
    val resourceList = new ListBuffer[String]()
    val resourceMetaData = defaultDownloader.downloadMetadataIfNeed(folder)
    for (meta <- resourceMetaData) {
      val isSameResourceType =
        meta.category.getOrElse(ResourceType.NOT_DEFINED).toString.equals(resourceType.toString)
      val isCompatibleWithVersion = version match {
        case Some(ver) => Version.isCompatible(ver, meta.libVersion)
        case None => true
      }
      val isSameAnnotator = annotator match {
        case Some(cls) => meta.annotator.getOrElse("").equalsIgnoreCase(cls)
        case None => true
      }
      val isSameLanguage = lang match {
        case Some(l) => meta.language.getOrElse("").equalsIgnoreCase(l)
        case None => true
      }

      if (isSameResourceType & isCompatibleWithVersion & isSameAnnotator & isSameLanguage) {
        resourceList += meta.name + ":" + meta.language.getOrElse("-") + ":" + meta.libVersion
          .getOrElse("-")
      }
    }
    resourceList.result()
  }

  def listPretrainedResources(
      folder: String,
      resourceType: ResourceType,
      lang: String): List[String] =
    listPretrainedResources(folder, resourceType, lang = Some(lang))

  def listPretrainedResources(
      folder: String,
      resourceType: ResourceType,
      version: Version): List[String] =
    listPretrainedResources(folder, resourceType, version = Some(version))

  def listPretrainedResources(
      folder: String,
      resourceType: ResourceType,
      lang: String,
      version: Version): List[String] =
    listPretrainedResources(folder, resourceType, lang = Some(lang), version = Some(version))

  def listAvailableAnnotators(folder: String = publicLoc): List[String] = {
    val resourceMetaData = defaultDownloader.downloadMetadataIfNeed(folder)
    resourceMetaData
      .map(_.annotator.getOrElse(""))
      .toSet
      .filter { a =>
        !a.equals("")
      }
      .toList
      .sorted
  }

  def showAvailableAnnotators(folder: String = publicLoc): Unit = {
    println(listAvailableAnnotators(folder).mkString("\n"))
  }

  /** Loads resource to path
    *
    * @param name
    *   Name of Resource
    * @param folder
    *   Subfolder in s3 where to search model (e.g. medicine)
    * @param language
    *   Desired language of Resource
    * @return
    *   path of downloaded resource
    */
  def downloadResource(
      name: String,
      language: Option[String] = None,
      folder: String = publicLoc): String = {
    downloadResource(ResourceRequest(name, language, folder))
  }

  /** Loads resource to path
    *
    * @param request
    *   Request for resource
    * @return
    *   path of downloaded resource
    */
  def downloadResource(request: ResourceRequest): String = {
    val future = Future {
      if (request.folder.equals(publicLoc)) {
        publicDownloader.download(request)
      } else if (request.folder.startsWith("@")) {
        val actualLoc = request.folder.replace("@", "")
        val updatedRequest = ResourceRequest(
          request.name,
          request.language,
          folder = actualLoc,
          request.libVersion,
          request.sparkVersion)
        communityDownloader.download(updatedRequest)
      } else {
        defaultDownloader.download(request)
      }
    }

    var downloadFinished = false
    var path: Option[String] = None
    val fileSize = getDownloadSize(request)
    require(
      !fileSize.equals("-1"),
      s"Can not find ${request.name} inside ${request.folder} to download. Please make sure the name and location are correct!")
    println(request.name + " download started this may take some time.")
    println("Approximate size to download " + fileSize)

    while (!downloadFinished) {
      future.onComplete {
        case Success(value) =>
          downloadFinished = true
          path = value
        case Failure(exception) =>
          println(s"Error: ${exception.getMessage}")
          logger.error(exception.getMessage)
          downloadFinished = true
          path = None
      }
      Thread.sleep(1000)

    }

    require(
      path.isDefined,
      s"Was not found appropriate resource to download for request: $request with downloader: $defaultDownloader")
    println("Download done! Loading the resource.")
    path.get
  }

  /** Downloads a resource from the default S3 bucket in the cache pretrained folder.
    * @param model
    *   the name of the key in the S3 bucket
    * @param folder
    *   the language of the model
    * @return
    *   the path to the downloaded file
    */
  def downloadModelDirectly(model: String, folder: String = publicLoc): Unit = {
    if (folder.equals(publicLoc)) {
      publicDownloader.downloadAndUnzipFile(model)
    } else if (folder.startsWith("@")) {
      communityDownloader.downloadAndUnzipFile(model)
    } else {
      defaultDownloader.downloadAndUnzipFile(model)
    }
  }

  def downloadModel[TModel <: PipelineStage](
      reader: DefaultParamsReadable[TModel],
      name: String,
      language: Option[String] = None,
      folder: String = publicLoc): TModel = {
    downloadModel(reader, ResourceRequest(name, language, folder))
  }

  def downloadModel[TModel <: PipelineStage](
      reader: DefaultParamsReadable[TModel],
      request: ResourceRequest): TModel = {
    if (!cache.contains(request)) {
      val path = downloadResource(request)
      val model = reader.read.load(path)
      cache(request) = model
      model
    } else {
      cache(request).asInstanceOf[TModel]
    }
  }

  def downloadPipeline(
      name: String,
      language: Option[String] = None,
      folder: String = publicLoc): PipelineModel = {
    downloadPipeline(ResourceRequest(name, language, folder))
  }

  def downloadPipeline(request: ResourceRequest): PipelineModel = {
    if (!cache.contains(request)) {
      val path = downloadResource(request)
      val model = PipelineModel.read.load(path)
      cache(request) = model
      model
    } else {
      cache(request).asInstanceOf[PipelineModel]
    }
  }

  def clearCache(
      name: String,
      language: Option[String] = None,
      folder: String = publicLoc): Unit = {
    clearCache(ResourceRequest(name, language, folder))
  }

  def clearCache(request: ResourceRequest): Unit = {
    defaultDownloader.clearCache(request)
    publicDownloader.clearCache(request)
    communityDownloader.clearCache(request)
    cache.remove(request)
  }

  def getDownloadSize(resourceRequest: ResourceRequest): String = {
    var size: Option[Long] = None
    val folder = resourceRequest.folder
    if (folder.equals(publicLoc)) {
      size = publicDownloader.getDownloadSize(resourceRequest)
    } else if (folder.startsWith("@")) {
      val actualLoc = folder.replace("@", "")
      size = communityDownloader.getDownloadSize(
        ResourceRequest(resourceRequest.name, resourceRequest.language, actualLoc))
    } else {
      size = defaultDownloader.getDownloadSize(resourceRequest)
    }
    size match {
      case Some(downloadBytes) => FileHelper.getHumanReadableFileSize(downloadBytes)
      case None => "-1"

    }
  }

  /** Downloads the provided S3 path to a local temporary directory and returns the location of
    * the folder.
    *
    * @param path
    *   S3 URL to the resource
    * @return
    *   URI of the local path to the temporary folder of the resource
    */
  def downloadS3Directory(path: String): URI = {

    val (bucketName, keyPrefix) = ResourceHelper.parseS3URI(path)

    val tmpDirectory = SparkFiles.getRootDirectory()

    val awsGateway = new AWSGateway()
    awsGateway.downloadFilesFromDirectory(bucketName, keyPrefix, new File(tmpDirectory))

    Paths.get(tmpDirectory, keyPrefix).toUri
  }

}

object ResourceType extends Enumeration {
  type ResourceType = Value
  val MODEL: pretrained.ResourceType.Value = Value("ml")
  val PIPELINE: pretrained.ResourceType.Value = Value("pl")
  val NOT_DEFINED: pretrained.ResourceType.Value = Value("nd")
}

case class ResourceRequest(
    name: String,
    language: Option[String] = None,
    folder: String = ResourceDownloader.publicLoc,
    libVersion: Version = ResourceDownloader.libVersion,
    sparkVersion: Version = ResourceDownloader.sparkVersion)

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
    "XlmRoBertaEmbeddings" -> XlmRoBertaEmbeddings,
    "LongformerEmbeddings" -> LongformerEmbeddings,
    "RoBertaSentenceEmbeddings" -> RoBertaSentenceEmbeddings,
    "XlmRoBertaSentenceEmbeddings" -> XlmRoBertaSentenceEmbeddings,
    "AlbertForTokenClassification" -> AlbertForTokenClassification,
    "BertForTokenClassification" -> BertForTokenClassification,
    "DeBertaForTokenClassification" -> DeBertaForTokenClassification,
    "DistilBertForTokenClassification" -> DistilBertForTokenClassification,
    "LongformerForTokenClassification" -> LongformerForTokenClassification,
    "RoBertaForTokenClassification" -> RoBertaForTokenClassification,
    "XlmRoBertaForTokenClassification" -> XlmRoBertaForTokenClassification,
    "XlnetForTokenClassification" -> XlnetForTokenClassification,
    "AlbertForSequenceClassification" -> AlbertForSequenceClassification,
    "BertForSequenceClassification" -> BertForSequenceClassification,
    "DeBertaForSequenceClassification" -> DeBertaForSequenceClassification,
    "DistilBertForSequenceClassification" -> DistilBertForSequenceClassification,
    "LongformerForSequenceClassification" -> LongformerForSequenceClassification,
    "RoBertaForSequenceClassification" -> RoBertaForSequenceClassification,
    "XlmRoBertaForSequenceClassification" -> XlmRoBertaForSequenceClassification,
    "XlnetForSequenceClassification" -> XlnetForSequenceClassification,
    "GPT2Transformer" -> GPT2Transformer,
    "EntityRulerModel" -> EntityRulerModel,
    "Doc2VecModel" -> Doc2VecModel,
    "Word2VecModel" -> Word2VecModel,
    "DeBertaEmbeddings" -> DeBertaEmbeddings,
    "DeBertaForSequenceClassification" -> DeBertaForSequenceClassification,
    "DeBertaForTokenClassification" -> DeBertaForTokenClassification,
    "CamemBertEmbeddings" -> CamemBertEmbeddings,
    "AlbertForQuestionAnswering" -> AlbertForQuestionAnswering,
    "BertForQuestionAnswering" -> BertForQuestionAnswering,
    "DeBertaForQuestionAnswering" -> DeBertaForQuestionAnswering,
    "DistilBertForQuestionAnswering" -> DistilBertForQuestionAnswering,
    "LongformerForQuestionAnswering" -> LongformerForQuestionAnswering,
    "RoBertaForQuestionAnswering" -> RoBertaForQuestionAnswering,
    "XlmRoBertaForQuestionAnswering" -> XlmRoBertaForQuestionAnswering,
    "SpanBertCorefModel" -> SpanBertCorefModel,
    "ViTForImageClassification" -> ViTForImageClassification,
    "Wav2Vec2ForCTC" -> Wav2Vec2ForCTC,
    "CamemBertForTokenClassification" -> CamemBertForTokenClassification,
    "TableAssembler" -> TableAssembler,
    "TapasForQuestionAnswering" -> TapasForQuestionAnswering)

  def downloadModel(
      readerStr: String,
      name: String,
      language: String = null,
      remoteLoc: String = null): PipelineStage = {
    val reader = keyToReader.getOrElse(
      readerStr,
      throw new RuntimeException(s"Unsupported Model: $readerStr"))
    val correctedFolder = Option(remoteLoc).getOrElse(ResourceDownloader.publicLoc)
    ResourceDownloader.downloadModel(
      reader.asInstanceOf[DefaultParamsReadable[PipelineStage]],
      name,
      Option(language),
      correctedFolder)
  }

  def downloadPipeline(
      name: String,
      language: String = null,
      remoteLoc: String = null): PipelineModel = {
    val correctedFolder = Option(remoteLoc).getOrElse(ResourceDownloader.publicLoc)
    ResourceDownloader.downloadPipeline(name, Option(language), correctedFolder)
  }

  def clearCache(name: String, language: String = null, remoteLoc: String = null): Unit = {
    val correctedFolder = Option(remoteLoc).getOrElse(ResourceDownloader.publicLoc)
    ResourceDownloader.clearCache(name, Option(language), correctedFolder)
  }

  def downloadModelDirectly(model: String, remoteLoc: String = null): Unit = {
    val correctedFolder = Option(remoteLoc).getOrElse(ResourceDownloader.publicLoc)
    ResourceDownloader.downloadModelDirectly(model, correctedFolder)
  }

  def showUnCategorizedResources(): String = {
    ResourceDownloader.publicResourceString(
      annotator = None,
      lang = None,
      version = None,
      resourceType = ResourceType.NOT_DEFINED)
  }

  def showPublicPipelines(lang: String, version: String): String = {
    val ver: Option[String] = version match {
      case null => Some(Build.version)
      case _ => Some(version)
    }
    ResourceDownloader.publicResourceString(
      annotator = None,
      lang = Option(lang),
      version = ver,
      resourceType = ResourceType.PIPELINE)
  }

  def showPublicModels(annotator: String, lang: String, version: String): String = {
    val ver: Option[String] = version match {
      case null => Some(Build.version)
      case _ => Some(version)
    }
    ResourceDownloader.publicResourceString(
      annotator = Option(annotator),
      lang = Option(lang),
      version = ver,
      resourceType = ResourceType.MODEL)
  }

  def showAvailableAnnotators(): String = {
    ResourceDownloader.listAvailableAnnotators().mkString("\n")
  }

  def getDownloadSize(name: String, language: String = "en", remoteLoc: String = null): String = {
    val correctedFolder = Option(remoteLoc).getOrElse(ResourceDownloader.publicLoc)
    ResourceDownloader.getDownloadSize(ResourceRequest(name, Option(language), correctedFolder))
  }
}
