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
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasPretrained, HasSimpleAnnotate}
import com.johnsnowlabs.storage.Database.{ENTITY_REGEX_PATTERNS, Name}
import com.johnsnowlabs.storage._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{BooleanParam, StringArrayParam}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.slf4j.{Logger, LoggerFactory}

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

  def this() = this(Identifiable.randomUID("ENTITY_RULER"))

  private val logger: Logger = LoggerFactory.getLogger("Credentials")

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

  val extractEntities = new StringArrayParam(
    this,
    "extractEntities",
    "List of entity labels to extract (empty = extract all entities)")

  private[er] def setRegexEntities(value: Array[String]): this.type = set(regexEntities, value)

  private[er] def setEntityRulerFeatures(value: EntityRulerFeatures): this.type =
    set(entityRulerFeatures, value)

  private[er] def setUseStorage(value: Boolean): this.type = set(useStorage, value)

  private[er] def setSentenceMatch(value: Boolean): this.type = set(sentenceMatch, value)

  private[er] def setAhoCorasickAutomaton(value: Option[AhoCorasickAutomaton]): this.type =
    set(ahoCorasickAutomaton, value)

  def setExtractEntities(value: Array[String]): this.type = set(extractEntities, value)

  private var automatonModel: Option[Broadcast[AhoCorasickAutomaton]] = None

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
      if ($$(ahoCorasickAutomaton).isDefined) $$(ahoCorasickAutomaton) else None
    }
  }

  setDefault(
    useStorage -> false,
    caseSensitive -> true,
    enablePatternRegex -> false,
    extractEntities -> Array())

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

  override def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    this.setAutomatonModelIfNotSet(dataset.sparkSession, $$(ahoCorasickAutomaton))
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
    val entitiesToUse =
      if ($(extractEntities).nonEmpty)
        $(regexEntities).filter(e => $(extractEntities).contains(e.split(",")(0)))
      else
        $(regexEntities)

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
        } else None
      }
    }

    annotatedEntities
  }

  private def getMatchedEntity(
      token: String,
      regexPatternsReader: Option[RegexPatternsReader],
      activeEntities: Array[String]): Option[String] = {

    val matchesByEntity = activeEntities.flatMap { regexEntity =>
      val regexPatterns: Option[Seq[String]] = regexPatternsReader match {
        case Some(rpr) => rpr.lookup(regexEntity)
        case None => $$(entityRulerFeatures).regexPatterns.get(regexEntity)
      }
      if (regexPatterns.isDefined) {
        val matches = regexPatterns.get.flatMap(regexPattern => regexPattern.r.findFirstIn(token))
        if (matches.nonEmpty) Some(regexEntity) else None
      } else None
    }.toSeq

    if (matchesByEntity.size > 1) {
      logger.warn("More than one entity found. Sending the first element of the array")
    }

    matchesByEntity.headOption
  }

  private def getMatchedEntityBySentence(
      sentence: Sentence,
      regexPatternsReader: Option[RegexPatternsReader],
      activeEntities: Array[String]): Array[(IndexedToken, String)] = {

    val matchesByEntity = activeEntities
      .flatMap { regexEntity =>
        val regexPatterns: Option[Seq[String]] = regexPatternsReader match {
          case Some(rpr) => rpr.lookup(regexEntity)
          case None => $$(entityRulerFeatures).regexPatterns.get(regexEntity)
        }
        if (regexPatterns.isDefined) {

          val resultMatches = regexPatterns.get.flatMap { regexPattern =>
            val matchedResult = regexPattern.r.findFirstMatchIn(sentence.content)
            if (matchedResult.isDefined) {
              val begin = matchedResult.get.start + sentence.start
              val end = matchedResult.get.end + sentence.start - 1
              Some(matchedResult.get.toString(), begin, end, regexEntity)
            } else None
          }

          val intervals =
            resultMatches.map(resultMatch => List(resultMatch._2, resultMatch._3)).toList
          val mergedIntervals = EntityRulerUtil.mergeIntervals(intervals)
          val filteredMatches =
            resultMatches.filter(x => mergedIntervals.contains(List(x._2, x._3)))

          if (filteredMatches.nonEmpty) Some(filteredMatches) else None
        } else None
      }
      .flatten
      .sortBy(_._2)

    // Convert to (IndexedToken, EntityLabel) tuples
    matchesByEntity.map { case (matchedText, begin, end, entityLabel) =>
      (IndexedToken(matchedText, begin, end), entityLabel)
    }
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

object EntityRulerModel extends ReadablePretrainedEntityRuler
