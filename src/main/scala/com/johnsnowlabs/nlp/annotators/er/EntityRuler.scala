/*
 * Copyright 2017-2021 John Snow Labs
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

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.storage.Database.Name
import com.johnsnowlabs.storage.{Database, HasStorage, RocksDBConnection, StorageReadWriter, StorageWriter}
import com.johnsnowlabs.util.JsonParser
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.functions.collect_set

import scala.io.Source

class EntityRuler(override val uid: String) extends AnnotatorApproach[EntityRulerModel] with HasStorage {

  def this() = this(Identifiable.randomUID("ENTITY_RULER"))

  override val description: String = "Entity Ruler matches entities based on text patterns"

  val patternsResource = new ExternalResourceParam(this, "patternsResource",
    "Resource in JSON or CSV format to map entities to patterns")

  val enablePatternRegex = new BooleanParam(this, "enablePatternRegex",
    "Enables regex pattern match")

  def setEnablePatternRegex(value: Boolean) : this.type = set(enablePatternRegex, value)

  private var entities: Array[String] = Array()

  def setPatternsResource(path: String, readAs: ReadAs.Format,
                          options: Map[String, String] = Map("format" -> "JSON")): this.type =
    set(patternsResource, ExternalResource(path, readAs, options))

  setDefault(storagePath -> ExternalResource("", ReadAs.TEXT, Map()),
    patternsResource -> null, enablePatternRegex -> false
  )

  private val AVAILABLE_FORMATS = Array("JSON", "JSONL", "CSV")

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): EntityRulerModel = {

    val entityRuler = new EntityRulerModel()
      .setStorageRef($(storageRef))
      .setEnablePatternRegex($(enablePatternRegex))

    if ($(enablePatternRegex)) {
      entityRuler.setRegexEntities(entities)
    }
    entityRuler
  }

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN)
  override val outputAnnotatorType: AnnotatorType = CHUNK
  override protected val databases: Array[Name] = EntityRulerModel.databases

  protected def index(fitDataset: Dataset[_], storageSourcePath: Option[String], readAs: Option[ReadAs.Value],
                               writers: Map[Name, StorageWriter[_]], readOptions: Option[Map[String, String]]): Unit = {

    require($(patternsResource) != null, "patternsResource parameter required")
    require($(patternsResource).path != "", "path for a patternsResource file is required")
    require(AVAILABLE_FORMATS.contains($(patternsResource).options.getOrElse("format", "").toUpperCase()),
      "format option parameter required with either JSON or CSV values")
    if ($(patternsResource).options("format").toUpperCase() == "CSV") {
      require($(patternsResource).options.getOrElse("delimiter", "") != "", "delimiter option parameter required")
    }
    require($(patternsResource).readAs != null, "readAs parameter required")

    var storageWriter: StorageReadWriter[_] = null

    if ($(enablePatternRegex)) {
      storageWriter = writers(Database.ENTITY_REGEX_PATTERNS).asInstanceOf[RegexPatternsReadWriter]
    } else {
      storageWriter = writers(Database.ENTITY_PATTERNS).asInstanceOf[PatternsReadWriter]
    }

    resourceFormats match {
      case "JSON&TEXT" => storePatternsFromJson(storageWriter)
      case "JSONL&TEXT" => storePatternsFromJsonl(storageWriter)
      case "JSON&SPARK" => storePatternsFromJSONDataFrame(storageWriter, "JSON")
      case "JSONL&SPARK" => storePatternsFromJSONDataFrame(storageWriter, "JSONL")
      case "CSV&TEXT" => storePatternsFromCSV(storageWriter)
      case "CSV&SPARK" => storeEntityPatternsFromCSVDataFrame(storageWriter)
      case _ @ format => throw new IllegalArgumentException(s"format $format not available")
    }

  }

  private lazy val resourceFormats: String = $(patternsResource).options("format").toUpperCase() + "&" +  $(patternsResource).readAs

  private def storePatternsFromJson(storageReadWriter: StorageReadWriter[_]): Unit = {
    import io.circe.generic.auto._

    val stream =  ResourceHelper.getResourceStream($(patternsResource).path)
    val jsonContent = Source.fromInputStream(stream).mkString
    val jsonParser = new JsonParser[EntityPattern]
    val entityPatterns: Array[EntityPattern] = jsonParser.readJsonArray(jsonContent)

    entityPatterns.foreach{ entityPattern =>
      val entity = if (entityPattern.id.isDefined) s"${entityPattern.label},${entityPattern.id.get}" else entityPattern.label
      storageReadWriter match {
        case patternsWriter: PatternsReadWriter => storePatterns(entityPattern.patterns.toIterator, entity, patternsWriter)
        case regexPatternsWriter: RegexPatternsReadWriter => storeRegexPattern(entityPattern.patterns, entity, regexPatternsWriter)
      }
    }
  }

  private def storePatternsFromJsonl(storageReadWriter: StorageReadWriter[_]): Unit = {

    import io.circe.generic.auto._
    val jsonParser = new JsonParser[EntityPattern]

    val sourceStream = ResourceHelper.SourceStream($(patternsResource).path)
    sourceStream.content.foreach( content => content.foreach{ line =>
      val entityPattern: EntityPattern = jsonParser.readJson(line)
      val entity = if (entityPattern.id.isDefined) s"${entityPattern.label},${entityPattern.id.get}" else entityPattern.label
      storageReadWriter match {
        case patternsWriter: PatternsReadWriter => storePatterns(entityPattern.patterns.toIterator, entity, patternsWriter)
        case regexPatternsWriter: RegexPatternsReadWriter => storeRegexPattern(entityPattern.patterns, entity, regexPatternsWriter)
      }
    })
  }

  private def storePatternsFromCSV(storageReadWriter: StorageReadWriter[_]): Unit = {
    storageReadWriter match {
      case patternsWriter: PatternsReadWriter =>
        val entityPatterns: Map[String, String] = ResourceHelper.parseKeyValueText($(patternsResource))
        entityPatterns.foreach(entityPattern => storePattern(entityPattern._2, entityPattern._1, patternsWriter))
      case regexPatternsWriter: RegexPatternsReadWriter =>
        val entityPatterns: Map[String, List[String]] = ResourceHelper.parseKeyListValues($(patternsResource))
        entityPatterns.foreach(entityPattern => storeRegexPattern(entityPattern._2, entityPattern._1, regexPatternsWriter))
    }
  }

  private def storeEntityPatternsFromCSVDataFrame(storageReadWriter: StorageReadWriter[_]): Unit = {

    val patternOptions = $(patternsResource).options
    var patternsDataFrame = spark.read.options(patternOptions)
      .format(patternOptions("format"))
      .options(patternOptions)
      .option("delimiter", patternOptions("delimiter"))
      .load($(patternsResource).path)
      .toDF("label", "pattern")

    patternsDataFrame = patternsDataFrame.groupBy("label")
      .agg(collect_set("pattern").alias("patterns"))

    storeFromDataFrame(patternsDataFrame, storageReadWriter)

  }

  private def storePatternsFromJSONDataFrame(storageReadWriter: StorageReadWriter[_], format: String): Unit = {

    val path = $(patternsResource).path
    var patternsDataFrame = spark.read.option("multiline", "true").json(path)

    if (format.equals("JSONL")) {
      patternsDataFrame = spark.read.json(path)
    }

    storeFromDataFrame(patternsDataFrame, storageReadWriter)
  }

  private def storeFromDataFrame(patternsDataFrame: DataFrame, storageReadWriter: StorageReadWriter[_]): Unit = {
    val fieldId = patternsDataFrame.schema.fields.filter(field => field.name == "id")
    var id = ""

    patternsDataFrame.rdd.toLocalIterator.foreach{ row =>
      val patterns = row.getAs[Seq[String]]("patterns")
      val label = row.getAs[String]("label")
      if (fieldId.nonEmpty) {
        id = row.getAs[String]("id")
      }
      val entity = if (fieldId.nonEmpty) s"$label,$id" else label
      storageReadWriter match {
        case patternsWriter: PatternsReadWriter => storePatterns(patterns.toIterator, entity, patternsWriter)
        case regexPatternsWriter: RegexPatternsReadWriter => storeRegexPattern(patterns, entity, regexPatternsWriter)
      }
    }
  }

  private def storePatterns(patterns: Iterator[String], entity: String, patternsReaderWriter: PatternsReadWriter): Unit = {
    patterns.foreach(pattern => storePattern(pattern, entity, patternsReaderWriter))
  }

  private def storePattern(pattern: String, entity: String, patternsReaderWriter: PatternsReadWriter): Unit = {
    patternsReaderWriter.lookup(pattern).getOrElse(patternsReaderWriter.add(pattern, entity))
  }

  private def storeRegexPattern(pattern: Seq[String], entity: String,
                                regexPatternsReaderWriter: RegexPatternsReadWriter): Unit = {

    if (!entities.contains(entity)) {
      entities = entities ++ Array(entity)
    }
    regexPatternsReaderWriter.lookup(entity).getOrElse(regexPatternsReaderWriter.add(entity, pattern))
  }

  protected def createWriter(database: Name, connection: RocksDBConnection): StorageWriter[_] = {
    database match {
      case Database.ENTITY_PATTERNS => new PatternsReadWriter(connection)
      case Database.ENTITY_REGEX_PATTERNS => new RegexPatternsReadWriter(connection)
    }
  }

}