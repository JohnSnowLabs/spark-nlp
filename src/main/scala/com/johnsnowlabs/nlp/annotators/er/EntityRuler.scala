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
import com.johnsnowlabs.storage.{Database, HasStorage, RocksDBConnection, StorageWriter}
import com.johnsnowlabs.util.JsonParser
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

import scala.io.Source

class EntityRuler(override val uid: String) extends AnnotatorApproach[EntityRulerModel] with HasStorage {

  def this() = this(Identifiable.randomUID("ENTITY_RULER"))

  override val description: String = "Entity Ruler matches entities based on text patterns"

  val patterns = new ExternalResourceParam(this, "patterns",
    "Resource in JSON or CSV format to map entities to patterns")

  def setPatterns(path: String, readAs: ReadAs.Format, options: Map[String, String] = Map("format" -> "JSON")): this.type =
    set(patterns, ExternalResource(path, readAs, options))

  setDefault(storagePath -> ExternalResource("", ReadAs.TEXT, Map()),
    patterns -> null
  )

  private val AVAILABLE_FORMATS = Array("JSON", "CSV")

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): EntityRulerModel = {

    new EntityRulerModel()
      .setStorageRef($(storageRef))
  }

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN)
  override val outputAnnotatorType: AnnotatorType = CHUNK
  override protected val databases: Array[Name] = EntityRulerModel.databases

  override protected def index(fitDataset: Dataset[_], storageSourcePath: Option[String], readAs: Option[ReadAs.Value],
                               writers: Map[Name, StorageWriter[_]], readOptions: Option[Map[String, String]]): Unit = {

    require($(patterns) != null, "patterns parameter required")
    require(AVAILABLE_FORMATS.contains($(patterns).options.getOrElse("format", "").toUpperCase()),
      "format option parameter required with either JSON or CSV values")
    if ($(patterns).options("format") != "JSON") {
      require($(patterns).options.getOrElse("delimiter", "") != "", "delimiter option parameter required")
    }
    require($(patterns).readAs != null, "readAs parameter required")

    val patternsReaderWriter = writers(Database.ENTITY_PATTERNS).asInstanceOf[PatternsReadWriter]

    resourceFormats match {
      case "JSON&TEXT" => storeEntityPatternsFromJson(patternsReaderWriter)
      case "JSON&SPARK" => storeEntityPatternsFromJSONDataFrame(patternsReaderWriter)
      case "CSV&TEXT" => storeEntityPatternsFromCSV(patternsReaderWriter)
      case "CSV&SPARK" => storeEntityPatternsFromCSVDataFrame(patternsReaderWriter)
    }

  }

  private lazy val resourceFormats: String = $(patterns).options("format").toUpperCase() + "&" +  $(patterns).readAs

  private def storeEntityPatternsFromJson(patternsReaderWriter: PatternsReadWriter): Unit = {
    import io.circe.generic.auto._

    val stream =  ResourceHelper.getResourceStream($(patterns).path)
    val jsonContent = Source.fromInputStream(stream).mkString
    val jsonParser = new JsonParser[EntityPattern]
    val entityPatterns: Array[EntityPattern] = jsonParser.readJsonArray(jsonContent)

    entityPatterns.foreach{ entityPattern =>
      storePatterns(entityPattern.pattern, entityPattern.label, patternsReaderWriter)
    }
  }

  private def storeEntityPatternsFromCSV(patternsReaderWriter: PatternsReadWriter): Unit = {
    val entityPatterns: Map[String, String] = ResourceHelper.parseKeyValueText($(patterns))
    entityPatterns.foreach(entityPattern => storePatterns(entityPattern._1, entityPattern._2, patternsReaderWriter))
  }

  private def storeEntityPatternsFromCSVDataFrame(patternsReaderWriter: PatternsReadWriter): Unit = {

    val patternsResource = $(patterns)
    val patternsDataFrame = spark.read.options(patternsResource.options)
      .format(patternsResource.options("format"))
      .options(patternsResource.options)
      .option("delimiter", patternsResource.options("delimiter"))
      .load(patternsResource.path)
      .toDF("pattern", "entity")

    patternsDataFrame.rdd.toLocalIterator.foreach{ row =>
      val pattern = row.getAs[String]("pattern")
      val entity = row.getAs[String]("entity")
      storePatterns(pattern, entity, patternsReaderWriter)
    }

  }

  private def storeEntityPatternsFromJSONDataFrame(patternsReaderWriter: PatternsReadWriter): Unit = {

    val patternsResource = $(patterns)
    val patternsDataFrame = spark.read.option("multiline", "true")
      .json(patternsResource.path)

    patternsDataFrame.printSchema()

    patternsDataFrame.rdd.toLocalIterator.foreach{ row =>
      val pattern = row.getAs[String]("pattern")
      val entity = row.getAs[String]("label")
      storePatterns(pattern, entity, patternsReaderWriter)
    }

  }

  private def storePatterns(pattern: String, entity: String, patternsReaderWriter: PatternsReadWriter): Unit = {
    pattern.split("\\s").foreach(p => patternsReaderWriter.lookup(p).getOrElse(patternsReaderWriter.add(p, entity)))
  }

  override protected def createWriter(database: Name, connection: RocksDBConnection): StorageWriter[_] = {
    new PatternsReadWriter(connection)
  }
}