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

package com.johnsnowlabs.nlp.annotators.er

import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.common._
import com.johnsnowlabs.nlp.annotators.er.EntityRulerModel.{
  AUTO_MODES,
  ENTITY_PRESETS,
  describeAutoMode,
  getPatternByName
}
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasPretrained, HasSimpleAnnotate}
import com.johnsnowlabs.storage.Database.{ENTITY_REGEX_PATTERNS, Name}
import com.johnsnowlabs.storage._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, Param, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.slf4j.{Logger, LoggerFactory}

import java.util.Locale
import scala.util.Try

/** Instantiated model of the [[EntityRulerApproach]]. For usage and examples see the
  * documentation of the main class.
  *
  * @param uid
  *   internally renquired UID to make it writable
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class EntityRulerModel(override val uid: String)
    extends AnnotatorModel[EntityRulerModel]
    with HasSimpleAnnotate[EntityRulerModel]
    with HasStorageModel {

  def this() = this(Identifiable.randomUID("ENTITY_RULER_MODEL"))

  @transient
  private val logger: Logger = LoggerFactory.getLogger("EntityRulerModel")

  // Keeping this parameter for backward compatibility
  private[er] val enablePatternRegex =
    new BooleanParam(this, "enablePatternRegex", "Enables regex pattern match")

  private[er] val useStorage =
    new BooleanParam(this, "useStorage", "Whether to use RocksDB storage to serialize patterns")

  private[er] val regexEntities =
    new StringArrayParam(this, "regexEntities", "entities defined in regex patterns")

  private[er] val entityRulerFeatures: StructFeature[EntityRulerFeatures] =
    new StructFeature[EntityRulerFeatures](
      this,
      "Structure to store data when RocksDB is not used")

  private[er] val sentenceMatch = new BooleanParam(
    this,
    "sentenceMatch",
    "Whether to find match at sentence level (regex only). True: sentence level. False: token level")

  private[er] val ahoCorasickAutomaton: StructFeature[Option[AhoCorasickAutomaton]] =
    new StructFeature[Option[AhoCorasickAutomaton]](this, "AhoCorasickAutomaton")

  private[er] val extractEntities = new StringArrayParam(
    this,
    "extractEntities",
    "List of entity labels to extract (empty = extract all entities)")

  /** AutoMode defines logical bundles of regex patterns to activate (e.g., "network_entities",
    * "email_entities"). When set, overrides activeEntities selection.
    */
  private[er] val autoMode: Param[String] = new Param[String](
    this,
    "autoMode",
    "Predefined bundle of entity patterns to use (e.g. 'network_entities', 'email_entities', 'all_entities').")

  private[er] def setRegexEntities(value: Array[String]): this.type = set(regexEntities, value)

  private[er] def setEntityRulerFeatures(value: EntityRulerFeatures): this.type =
    set(entityRulerFeatures, value)

  private[er] def setUseStorage(value: Boolean): this.type = set(useStorage, value)

  private[er] def setSentenceMatch(value: Boolean): this.type = set(sentenceMatch, value)

  private[er] def setAhoCorasickAutomaton(value: Option[AhoCorasickAutomaton]): this.type =
    set(ahoCorasickAutomaton, value)

  def setExtractEntities(value: Array[String]): this.type = set(extractEntities, value)

  def setAutoMode(value: String): this.type = set(autoMode, value)

  private var automatonModel: Option[Broadcast[AhoCorasickAutomaton]] = None

  val hasAutoMode: Boolean = {
    val value = if (isDefined(autoMode)) Try($(autoMode)).toOption else None
    val result = value.exists(AUTO_MODES.contains)
    result
  }

  def setAutomatonModelIfNotSet(
      spark: SparkSession,
      automaton: Option[AhoCorasickAutomaton]): this.type = {
    if (automatonModel.isEmpty && automaton.isDefined) {
      automatonModel = Some(spark.sparkContext.broadcast(automaton.get))
    }
    this
  }

  def getAutomatonModelIfNotSet: Option[AhoCorasickAutomaton] = {
    if (automatonModel.isDefined) {
      Some(automatonModel.get.value)
    } else {
      if (this.get(ahoCorasickAutomaton).isDefined && $$(ahoCorasickAutomaton).isDefined)
        $$(ahoCorasickAutomaton)
      else None
    }
  }

  setDefault(
    useStorage -> false,
    sentenceMatch -> false,
    caseSensitive -> true,
    enablePatternRegex -> false,
    extractEntities -> Array(),
    regexEntities -> Array(),
    autoMode -> "")

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)
  override val optionalInputAnnotatorTypes: Array[String] = Array(TOKEN)
  val outputAnnotatorType: AnnotatorType = CHUNK

  override def _transform(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): DataFrame = {
    if ($(regexEntities).nonEmpty) {
      val structFields = dataset.schema.fields
        .filter(field => field.metadata.contains("annotatorType"))
        .filter(field => field.metadata.getString("annotatorType") == TOKEN)
      if (structFields.isEmpty) {
        throw new IllegalArgumentException(
          s"Missing $TOKEN annotator. Regex patterns requires it in your pipeline")
      } else {
        super._transform(dataset, recursivePipeline)
      }
    } else {
      super._transform(dataset, recursivePipeline)
    }
  }

  private def getActiveEntitiesFromAutoMode: Array[String] = {
    if (isDefined(autoMode) && Option($(autoMode)).exists(_.nonEmpty)) {
      val modeKey = $(autoMode).toUpperCase(Locale.ROOT)
      val validEntities = EntityRulerModel.AUTO_MODES.getOrElse(modeKey, Seq.empty)
      validEntities.toArray
    } else {
      // Fallback for legacy regex-based pipelines
      $(regexEntities)
    }
  }

  override def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    getAutomatonModelIfNotSet.foreach { automaton =>
      this.setAutomatonModelIfNotSet(dataset.sparkSession, Some(automaton))
    }
    dataset
  }

  /** takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param annotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    var annotatedEntitiesByKeywords: Seq[Annotation] = Seq()
    val sentences = SentenceSplit.unpack(annotations)
    val annotatedEntitiesByRegex = computeAnnotatedEntitiesByRegex(annotations, sentences)

    if (getAutomatonModelIfNotSet.isDefined) {
      annotatedEntitiesByKeywords = sentences.flatMap { sentence =>
        getAutomatonModelIfNotSet.get.searchPatternsInText(sentence)
      }
    }

    annotatedEntitiesByRegex ++ annotatedEntitiesByKeywords
  }

  private def computeAnnotatedEntitiesByRegex(
      annotations: Seq[Annotation],
      sentences: Seq[Sentence]): Seq[Annotation] = {
    val entitiesToUse = {
      val baseEntities =
        if ($(extractEntities).nonEmpty) {
          $(regexEntities).filter(e => $(extractEntities).contains(e.split(",")(0)))
        } else if (isDefined(autoMode)) getActiveEntitiesFromAutoMode
        else $(regexEntities)

      baseEntities
    }

    if (entitiesToUse.nonEmpty) {
      val regexPatternsReader =
        if ($(useStorage))
          Some(getReader(Database.ENTITY_REGEX_PATTERNS).asInstanceOf[RegexPatternsReader])
        else None

      if ($(sentenceMatch)) {
        annotateEntitiesFromRegexPatternsBySentence(sentences, regexPatternsReader, entitiesToUse)
      } else {
        val tokenizedWithSentences = TokenizedWithSentence.unpack(annotations)
        annotateEntitiesFromRegexPatterns(
          tokenizedWithSentences,
          regexPatternsReader,
          entitiesToUse)
      }
    } else Seq()
  }

  private def annotateEntitiesFromRegexPatterns(
      tokenizedWithSentences: Seq[TokenizedSentence],
      regexPatternsReader: Option[RegexPatternsReader],
      activeEntities: Array[String]): Seq[Annotation] = {

    val annotatedEntities = tokenizedWithSentences.flatMap { tokenizedWithSentence =>
      tokenizedWithSentence.indexedTokens.flatMap { indexedToken =>
        val entity = getMatchedEntity(indexedToken.token, regexPatternsReader, activeEntities)
        if (entity.isDefined) {
          val entityMetadata = getEntityMetadata(entity)
          Some(
            Annotation(
              CHUNK,
              indexedToken.begin,
              indexedToken.end,
              indexedToken.token,
              entityMetadata ++ Map("sentence" -> tokenizedWithSentence.sentenceIndex.toString)))
        } else {
          None
        }
      }
    }

    annotatedEntities
  }

  private def getMatchedEntity(
      token: String,
      regexPatternsReader: Option[RegexPatternsReader],
      activeEntities: Array[String]): Option[String] = {

    val hasActiveEntities = activeEntities != null && activeEntities.nonEmpty

    val selectedEntities: Seq[(String, String)] = if (hasAutoMode) {
      describeAutoMode($(autoMode)).flatMap(name => getPatternByName(name).map(name -> _))

    } else if (hasActiveEntities) {
      activeEntities.flatMap { regexEntity =>
        // load from reader if available, else from local model storage
        val regexPatterns: Option[Seq[String]] = regexPatternsReader match {
          case Some(rpr) => {
            rpr.lookup(regexEntity)
          }
          case None => {
            // fallback if entityRulerFeatures is not set
            if (get(entityRulerFeatures).isDefined)
              $$(entityRulerFeatures).regexPatterns.get(regexEntity)
            else
              EntityRulerModel.ENTITY_PRESETS.get(regexEntity).map(Seq(_))
          }
        }

        regexPatterns match {
          case Some(patterns) => patterns.map(p => regexEntity -> p)
          case None => Seq.empty
        }
      }

    } else {
      ENTITY_PRESETS.toSeq
    }

    val matchesByEntity = selectedEntities.flatMap { case (entityName, regexPattern) =>
      regexPattern.r.findFirstIn(token).map(_ => entityName)
    }

    if (matchesByEntity.size > 1)
      logger.warn(
        s"[EntityRulerModel] Multiple entities matched token '$token': ${matchesByEntity.mkString(", ")}. " +
          s"Returning first entity '${matchesByEntity.head}'.")

    matchesByEntity.headOption
  }

  /** Extracts all regex-matched entities within a sentence, supporting both autoMode and manual
    * patterns.
    */
  private def getMatchedEntityBySentence(
      sentence: Sentence,
      regexPatternsReader: Option[RegexPatternsReader],
      activeEntities: Array[String]): Array[(IndexedToken, String)] = {

    import EntityRulerModel._

    val hasActiveEntities = activeEntities != null && activeEntities.nonEmpty

    val selectedEntities: Seq[(String, String)] = if (hasAutoMode) {
      describeAutoMode($(autoMode)).flatMap(name => getPatternByName(name).map(name -> _))

    } else if (hasActiveEntities) {
      activeEntities.flatMap { regexEntity =>
        val regexPatterns: Option[Seq[String]] = regexPatternsReader match {
          case Some(rpr) => rpr.lookup(regexEntity)
          case None => {
            // fallback if entityRulerFeatures is not set
            if (get(entityRulerFeatures).isDefined)
              $$(entityRulerFeatures).regexPatterns.get(regexEntity)
            else
              EntityRulerModel.ENTITY_PRESETS.get(regexEntity).map(Seq(_))
          }
        }

        regexPatterns match {
          case Some(patterns) => patterns.map(p => regexEntity -> p)
          case None => Seq.empty
        }
      }

    } else {
      ENTITY_PRESETS.toSeq
    }

    val matchesByEntity = selectedEntities
      .flatMap { case (regexEntity, regexPattern) =>
        val allMatches = regexPattern.r
          .findAllMatchIn(sentence.content)
          .map { matched =>
            val begin = matched.start + sentence.start
            val end = matched.end + sentence.start - 1
            (matched.matched, begin, end, regexEntity)
          }
          .toSeq

        // Merge overlapping intervals for same pattern
        val intervals = allMatches.map { case (_, b, e, _) => List(b, e) }.toList
        val mergedIntervals = EntityRulerUtil.mergeIntervals(intervals)

        val filteredMatches =
          allMatches.filter { case (_, b, e, _) => mergedIntervals.contains(List(b, e)) }

        if (filteredMatches.nonEmpty) Some(filteredMatches) else None
      }
      .flatten
      .sortBy(_._2) // sort by begin position

    matchesByEntity.map { case (matchedText, begin, end, entityLabel) =>
      (IndexedToken(matchedText, begin, end), entityLabel)
    }.toArray
  }

  private def annotateEntitiesFromRegexPatternsBySentence(
      sentences: Seq[Sentence],
      patternsReader: Option[RegexPatternsReader],
      activeEntities: Array[String]): Seq[Annotation] = {

    val annotatedEntities = sentences.flatMap { sentence =>
      val matchedEntities = getMatchedEntityBySentence(sentence, patternsReader, activeEntities)
      matchedEntities.map { case (indexedToken, label) =>
        val entityMetadata = getEntityMetadata(Some(label))
        Annotation(
          CHUNK,
          indexedToken.begin,
          indexedToken.end,
          indexedToken.token,
          entityMetadata ++ Map("sentence" -> sentence.index.toString))
      }
    }
    annotatedEntities
  }

  private def getEntityMetadata(labelData: Option[String]): Map[String, String] = {

    val entityMetadata = labelData.get
      .split(",")
      .zipWithIndex
      .flatMap { case (metadata, index) =>
        if (index == 0) {
          Map("entity" -> metadata)
        } else Map("id" -> metadata)
      }
      .toMap

    entityMetadata
  }

  override def copy(extra: ParamMap): EntityRulerModel = {
    val copied = defaultCopy(extra)
    if (isDefined(autoMode)) this.setAutoMode($(autoMode))
    copied
  }

  override def deserializeStorage(path: String, spark: SparkSession): Unit = {
    if ($(useStorage)) {
      super.deserializeStorage(path: String, spark: SparkSession)
    }
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    if ($(useStorage)) {
      super.onWrite(path, spark)
    }
  }

  protected val databases: Array[Name] = EntityRulerModel.databases

  protected def createReader(database: Name, connection: RocksDBConnection): StorageReader[_] = {
    new RegexPatternsReader(connection)
  }
}

trait ReadablePretrainedEntityRuler
    extends StorageReadable[EntityRulerModel]
    with HasPretrained[EntityRulerModel] {

  override val databases: Array[Name] = Array(ENTITY_REGEX_PATTERNS)

  override val defaultModelName: Option[String] = None

  override def pretrained(): EntityRulerModel = super.pretrained()

  override def pretrained(name: String): EntityRulerModel = super.pretrained(name)

  override def pretrained(name: String, lang: String): EntityRulerModel =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): EntityRulerModel =
    super.pretrained(name, lang, remoteLoc)

}

object EntityRulerModel extends ReadablePretrainedEntityRuler {

  val EMAIL_DATETIMETZ_PATTERN = "EMAIL_DATETIMETZ_PATTERN"
  val EMAIL_ADDRESS_PATTERN = "EMAIL_ADDRESS_PATTERN"
  val IPV4_PATTERN = "IPV4_PATTERN"
  val IPV6_PATTERN = "IPV6_PATTERN"
  val IP_ADDRESS_PATTERN = "IP_ADDRESS_PATTERN"
  val IP_ADDRESS_NAME_PATTERN = "IP_ADDRESS_NAME_PATTERN"
  val MAPI_ID_PATTERN = "MAPI_ID_PATTERN"
  val US_PHONE_NUMBERS_PATTERN = "US_PHONE_NUMBERS_PATTERN"
  val IMAGE_URL_PATTERN = "IMAGE_URL_PATTERN"

  val NETWORK_ENTITIES = "NETWORK_ENTITIES"
  val EMAIL_ENTITIES = "EMAIL_ENTITIES"
  val COMMUNICATION_ENTITIES = "COMMUNICATION_ENTITIES"
  val CONTACT_ENTITIES = "CONTACT_ENTITIES"
  val MEDIA_ENTITIES = "MEDIA_ENTITIES"
  val ALL_ENTITIES = "ALL_ENTITIES"

  private lazy val ENTITY_PRESETS: Map[String, String] = Map(
    EMAIL_DATETIMETZ_PATTERN -> "[A-Za-z]{3},\\s\\d{1,2}\\s[A-Za-z]{3}\\s\\d{4}\\s\\d{2}:\\d{2}:\\d{2}\\s[+-]\\d{4}",
    EMAIL_ADDRESS_PATTERN -> "(?i)[a-z0-9\\.\\-+_]+@[a-z0-9\\.\\-+_]+\\.[a-z]+",
    IPV4_PATTERN -> "(?:25[0-5]|2[0-4]\\d|1\\d{2}|[1-9]?\\d)(?:\\.(?:25[0-5]|2[0-4]\\d|1\\d{2}|[1-9]?\\d)){3}",
    IPV6_PATTERN -> "[a-z0-9]{4}::[a-z0-9]{4}:[a-z0-9]{4}:[a-z0-9]{4}:[a-z0-9]{4}%?[0-9]*",
    IP_ADDRESS_NAME_PATTERN -> "[a-zA-Z0-9-]*\\.[a-zA-Z]*\\.[a-zA-Z]*",
    MAPI_ID_PATTERN -> "[0-9]*\\.[0-9]*\\.[0-9]*\\.[0-9]*",
    US_PHONE_NUMBERS_PATTERN -> "(?:\\+?(\\d{1,3}))?[-. (]*(\\d{3})?[-. )]*(\\d{3})[-. ]*(\\d{4})(?: *x(\\d+))?\\s*$",
    IMAGE_URL_PATTERN -> "(?i)https?://(?:[a-z0-9$_@.&+!*\\(\\),%-])+(?:/[a-z0-9$_@.&+!*\\(\\),%-]*)*\\.(?:jpg|jpeg|png|gif|bmp|heic)")

  private lazy val AUTO_MODES: Map[String, Seq[String]] = Map(
    NETWORK_ENTITIES
      .toUpperCase(Locale.ROOT) -> Seq(IPV4_PATTERN, IPV6_PATTERN, IP_ADDRESS_NAME_PATTERN),
    EMAIL_ENTITIES.toUpperCase(Locale.ROOT) -> Seq(
      EMAIL_ADDRESS_PATTERN,
      EMAIL_DATETIMETZ_PATTERN,
      MAPI_ID_PATTERN),
    COMMUNICATION_ENTITIES.toUpperCase(Locale.ROOT) -> Seq(
      EMAIL_ADDRESS_PATTERN,
      US_PHONE_NUMBERS_PATTERN),
    CONTACT_ENTITIES.toUpperCase(Locale.ROOT) -> (
      Seq(EMAIL_ADDRESS_PATTERN, US_PHONE_NUMBERS_PATTERN) ++
        Seq(IPV4_PATTERN, IPV6_PATTERN, IP_ADDRESS_NAME_PATTERN)
    ),
    MEDIA_ENTITIES.toUpperCase(Locale.ROOT) -> Seq(IMAGE_URL_PATTERN),
    ALL_ENTITIES.toUpperCase(Locale.ROOT) -> ENTITY_PRESETS.keys.toSeq)

  def getPatternByName(name: String): Option[String] = ENTITY_PRESETS.get(name)

  def describeAutoMode(mode: String): Seq[String] = AUTO_MODES.getOrElse(mode, Seq.empty)

}
