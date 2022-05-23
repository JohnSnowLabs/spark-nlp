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

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{CHUNK, DOCUMENT, TOKEN}
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.storage.Database.Name
import com.johnsnowlabs.storage._
import com.johnsnowlabs.util.spark.SparkUtil
import com.johnsnowlabs.util.{JsonParser, Version}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.{col, collect_set, concat, lit}
import org.apache.spark.sql.types.StructField
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.io.Source

/** Fits an Annotator to match exact strings or regex patterns provided in a file against a
  * Document and assigns them an named entity. The definitions can contain any number of named
  * entities.
  *
  * There are multiple ways and formats to set the extraction resource. It is possible to set it
  * either as a "JSON", "JSONL" or "CSV" file. A path to the file needs to be provided to
  * `setPatternsResource`. The file format needs to be set as the "format" field in the `option`
  * parameter map and depending on the file type, additional parameters might need to be set.
  *
  * To enable regex extraction, `setEnablePatternRegex(true)` needs to be called.
  *
  * If the file is in a JSON format, then the rule definitions need to be given in a list with the
  * fields "id", "label" and "patterns":
  * {{{
  *  [
  *   {
  *     "id": "person-regex",
  *     "label": "PERSON",
  *     "patterns": ["\\w+\\s\\w+", "\\w+-\\w+"]
  *   },
  *   {
  *     "id": "locations-words",
  *     "label": "LOCATION",
  *     "patterns": ["Winterfell"]
  *   }
  * ]
  * }}}
  *
  * The same fields also apply to a file in the JSONL format:
  * {{{
  * {"id": "names-with-j", "label": "PERSON", "patterns": ["Jon", "John", "John Snow"]}
  * {"id": "names-with-s", "label": "PERSON", "patterns": ["Stark", "Snow"]}
  * {"id": "names-with-e", "label": "PERSON", "patterns": ["Eddard", "Eddard Stark"]}
  * }}}
  *
  * In order to use a CSV file, an additional parameter "delimiter" needs to be set. In this case,
  * the delimiter might be set by using `.setPatternsResource("patterns.csv", ReadAs.TEXT,
  * Map("format"->"csv", "delimiter" -> "\\|"))`
  * {{{
  * PERSON|Jon
  * PERSON|John
  * PERSON|John Snow
  * LOCATION|Winterfell
  * }}}
  *
  * ==Example==
  * In this example, the entities file as the form of
  * {{{
  * PERSON|Jon
  * PERSON|John
  * PERSON|John Snow
  * LOCATION|Winterfell
  * }}}
  * where each line represents an entity and the associated string delimited by "|".
  *
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.annotators.er.EntityRulerApproach
  * import com.johnsnowlabs.nlp.util.io.ReadAs
  *
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols("document")
  *   .setOutputCol("token")
  *
  * val entityRuler = new EntityRulerApproach()
  *   .setInputCols("document", "token")
  *   .setOutputCol("entities")
  *   .setPatternsResource(
  *     path = "src/test/resources/entity-ruler/patterns.csv",
  *     readAs = ReadAs.TEXT,
  *     options = Map("format" -> "csv", "delimiter" -> "\\|")
  *   )
  *   .setEnablePatternRegex(true)
  *
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   tokenizer,
  *   entityRuler
  * ))
  *
  * val data = Seq("Jon Snow wants to be lord of Winterfell.").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(entities)").show(false)
  * +--------------------------------------------------------------------+
  * |col                                                                 |
  * +--------------------------------------------------------------------+
  * |[chunk, 0, 2, Jon, [entity -> PERSON, sentence -> 0], []]           |
  * |[chunk, 29, 38, Winterfell, [entity -> LOCATION, sentence -> 0], []]|
  * +--------------------------------------------------------------------+
  * }}}
  *
  * @param uid
  *   required uid for storing annotator to disk
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
class EntityRulerApproach(override val uid: String)
    extends AnnotatorApproach[EntityRulerModel]
    with HasStorage {

  def this() = this(Identifiable.randomUID("ENTITY_RULER"))

  override val description: String = "Entity Ruler matches entities based on text patterns"

  private var entities: Array[String] = Array()
  private var patterns: Map[String, String] = Map()
  private var regexPatterns: Map[String, Seq[String]] = Map()

  /** Resource in JSON or CSV format to map entities to patterns (Default: `null`).
    *
    * @group param
    */
  val patternsResource = new ExternalResourceParam(
    this,
    "patternsResource",
    "Resource in JSON or CSV format to map entities to patterns")

  /** Enables regex pattern match (Default: `false`).
    *
    * @group param
    */
  val enablePatternRegex =
    new BooleanParam(this, "enablePatternRegex", "Enables regex pattern match")

  val sentenceMatch = new BooleanParam(
    this,
    "sentenceMatch",
    "Whether to find match at sentence level. True: sentence level. False: token level")

  /** Whether to use RocksDB storage to serialize patterns (Default: `true`).
    *
    * @group param
    */
  val useStorage =
    new BooleanParam(this, "useStorage", "Whether to use RocksDB storage to serialize patterns")

  /** @group setParam */
  def setEnablePatternRegex(value: Boolean): this.type = set(enablePatternRegex, value)

  /** @group setParam */
  def setPatternsResource(
      path: String,
      readAs: ReadAs.Format,
      options: Map[String, String] = Map("format" -> "JSON")): this.type =
    set(patternsResource, ExternalResource(path, readAs, options))

  def setSentenceMatch(value: Boolean): this.type = set(sentenceMatch, value)

  /** @group setParam */
  def setUseStorage(value: Boolean): this.type = set(useStorage, value)

  setDefault(
    storagePath -> ExternalResource("", ReadAs.TEXT, Map()),
    patternsResource -> null,
    enablePatternRegex -> false,
    useStorage -> true,
    sentenceMatch -> false)

  private val AVAILABLE_FORMATS = Array("JSON", "JSONL", "CSV")

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): EntityRulerModel = {

    val entityRuler = new EntityRulerModel()

    if ($(useStorage)) {
      entityRuler
        .setStorageRef($(storageRef))
        .setEnablePatternRegex($(enablePatternRegex))
        .setUseStorage($(useStorage))
        .setSentenceMatch($(sentenceMatch))

    } else {
      validateParameters()
      resourceFormats match {
        case "JSON&TEXT" => storePatternsFromJson(None)
        case "JSONL&TEXT" => storePatternsFromJsonl(None)
        case "JSON&SPARK" => storePatternsFromJSONDataFrame(None, "JSON")
        case "JSONL&SPARK" => storePatternsFromJSONDataFrame(None, "JSONL")
        case "CSV&TEXT" => computePatternsFromCSV()
        case "CSV&SPARK" => storeEntityPatternsFromCSVDataFrame(None)
        case _ @format => throw new IllegalArgumentException(s"format $format not available")
      }
      val entityRulerFeatures = EntityRulerFeatures(patterns, regexPatterns)
      entityRuler
        .setUseStorage($(useStorage))
        .setEntityRulerFeatures(entityRulerFeatures)
    }

    if ($(enablePatternRegex) || $(sentenceMatch)) {
      entityRuler.setRegexEntities(entities)
    }
    entityRuler

  }

  /** Input annotator types: DOCUMENT, TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN)

  /** Output annotator types: CHUNK
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = CHUNK
  override protected val databases: Array[Name] = EntityRulerModel.databases

  protected def index(
      fitDataset: Dataset[_],
      storageSourcePath: Option[String],
      readAs: Option[ReadAs.Value],
      writers: Map[Name, StorageWriter[_]],
      readOptions: Option[Map[String, String]]): Unit = {

    if ($(useStorage)) {
      validateParameters()

      var storageWriter: StorageReadWriter[_] = null

      if ($(enablePatternRegex) || $(sentenceMatch)) {
        storageWriter = writers(Database.ENTITY_REGEX_PATTERNS)
          .asInstanceOf[RegexPatternsReadWriter]
      } else {
        storageWriter = writers(Database.ENTITY_PATTERNS).asInstanceOf[PatternsReadWriter]
      }

      resourceFormats match {
        case "JSON&TEXT" => storePatternsFromJson(Some(storageWriter))
        case "JSONL&TEXT" => storePatternsFromJsonl(Some(storageWriter))
        case "JSON&SPARK" => storePatternsFromJSONDataFrame(Some(storageWriter), "JSON")
        case "JSONL&SPARK" => storePatternsFromJSONDataFrame(Some(storageWriter), "JSONL")
        case "CSV&TEXT" => storePatternsFromCSV(storageWriter)
        case "CSV&SPARK" => storeEntityPatternsFromCSVDataFrame(Some(storageWriter))
        case _ @format => throw new IllegalArgumentException(s"format $format not available")
      }
    }

  }

  private def validateParameters(): Unit = {
    require($(patternsResource) != null, "patternsResource parameter required")
    require($(patternsResource).path != "", "path for a patternsResource file is required")
    require(
      AVAILABLE_FORMATS.contains(
        $(patternsResource).options.getOrElse("format", "").toUpperCase()),
      "format option parameter required with either JSON or CSV values")
    if ($(patternsResource).options("format").toUpperCase() == "CSV") {
      require(
        $(patternsResource).options.getOrElse("delimiter", "") != "",
        "delimiter option parameter required")
    }
    require($(patternsResource).readAs != null, "readAs parameter required")
  }

  private lazy val resourceFormats: String = $(patternsResource)
    .options("format")
    .toUpperCase() + "&" + $(patternsResource).readAs

  private def storePatternsFromJson(storageReadWriter: Option[StorageReadWriter[_]]): Unit = {

    val entityPatterns: Array[EntityPattern] = parseJSON()

    entityPatterns.foreach { entityPattern =>
      val entity =
        if (entityPattern.id.isDefined) s"${entityPattern.label},${entityPattern.id.get}"
        else entityPattern.label
      storageReadWriter.getOrElse(None) match {
        case patternsWriter: PatternsReadWriter =>
          storePatterns(entityPattern.patterns.toIterator, entity, patternsWriter)
        case regexPatternsWriter: RegexPatternsReadWriter =>
          storeRegexPattern(entityPattern.patterns, entity, regexPatternsWriter)
        case None => computePatterns(entityPattern.patterns, entity)
      }
    }
  }

  private def parseJSON(): Array[EntityPattern] = {
    val stream = ResourceHelper.getResourceStream($(patternsResource).path)
    val jsonContent = Source.fromInputStream(stream).mkString
    val entityPatterns: Array[EntityPattern] = JsonParser.parseArray[EntityPattern](jsonContent)

    if ($(sentenceMatch)) {

      val processedEntityPatterns: Array[EntityPattern] =
        entityPatterns
          .groupBy(_.label)
          .map { entityPattern =>
            val patterns: Seq[String] = entityPattern._2.flatMap(ep => ep.patterns).distinct
            EntityPattern(entityPattern._1, patterns)
          }
          .toArray

      processedEntityPatterns

    } else entityPatterns

  }

  private def storePatternsFromJsonl(storageReadWriter: Option[StorageReadWriter[_]]): Unit = {

    val sourceStream = ResourceHelper.SourceStream($(patternsResource).path)
    sourceStream.content.foreach(content =>
      content.foreach { line =>
        val entityPattern: EntityPattern = JsonParser.parseObject[EntityPattern](line)
        val entity =
          if (entityPattern.id.isDefined) s"${entityPattern.label},${entityPattern.id.get}"
          else entityPattern.label
        storageReadWriter.getOrElse(None) match {
          case patternsWriter: PatternsReadWriter =>
            storePatterns(entityPattern.patterns.toIterator, entity, patternsWriter)
          case regexPatternsWriter: RegexPatternsReadWriter =>
            storeRegexPattern(entityPattern.patterns, entity, regexPatternsWriter)
          case None => computePatterns(entityPattern.patterns, entity)
        }
      })
  }

  private def storePatternsFromCSV(storageReadWriter: StorageReadWriter[_]): Unit = {
    storageReadWriter match {
      case patternsWriter: PatternsReadWriter =>
        val entityPatterns: Map[String, String] =
          ResourceHelper.parseKeyValueText($(patternsResource))
        entityPatterns.foreach(entityPattern =>
          storePattern(entityPattern._2, entityPattern._1, patternsWriter))
      case regexPatternsWriter: RegexPatternsReadWriter =>
        val entityPatterns: Map[String, List[String]] =
          ResourceHelper.parseKeyListValues($(patternsResource))
        entityPatterns.foreach(entityPattern =>
          storeRegexPattern(entityPattern._2, entityPattern._1, regexPatternsWriter))
    }
  }

  private def storeEntityPatternsFromCSVDataFrame(
      storageReadWriter: Option[StorageReadWriter[_]]): Unit = {

    val patternOptions = $(patternsResource).options
    var patternsDataFrame = spark.read
      .options(patternOptions)
      .format(patternOptions("format"))
      .options(patternOptions)
      .option("delimiter", patternOptions("delimiter"))
      .load($(patternsResource).path)
      .toDF("label", "pattern")

    patternsDataFrame = patternsDataFrame
      .groupBy("label")
      .agg(collect_set("pattern").alias("patterns"))

    storeFromDataFrame(patternsDataFrame, storageReadWriter)

  }

  private def storePatternsFromJSONDataFrame(
      storageReadWriter: Option[StorageReadWriter[_]],
      format: String): Unit = {

    val path = $(patternsResource).path
    var patternsDataFrame = spark.read.option("multiline", "true").json(path)

    if (format.equals("JSONL")) {
      patternsDataFrame = spark.read.json(path)
    }

    storeFromDataFrame(patternsDataFrame, storageReadWriter)
  }

  private def storeFromDataFrame(
      patternsDataFrame: DataFrame,
      storageReadWriter: Option[StorageReadWriter[_]]): Unit = {

    val fieldId: Array[StructField] =
      patternsDataFrame.schema.fields.filter(field => field.name == "id")
    val cleanedPatternsDataFrame = cleanPatternsDataFrame(patternsDataFrame, fieldId)

    cleanedPatternsDataFrame.rdd.toLocalIterator.foreach { row =>
      val patterns = row.getAs[Seq[String]]("patterns")
      val entity =
        if (fieldId.nonEmpty) row.getAs[String]("label_id") else row.getAs[String]("label")
      storageReadWriter.getOrElse(None) match {
        case patternsWriter: PatternsReadWriter =>
          storePatterns(patterns.toIterator, entity, patternsWriter)
        case regexPatternsWriter: RegexPatternsReadWriter =>
          storeRegexPattern(patterns, entity, regexPatternsWriter)
        case None => computePatterns(patterns, entity)
      }
    }
  }

  private def cleanPatternsDataFrame(
      patternsDataFrame: DataFrame,
      fieldId: Array[StructField]): DataFrame = {

    val spark = patternsDataFrame.sparkSession
    val sparkVersion = Version.parse(spark.version).toFloat

    if (fieldId.nonEmpty) {
      val patternsWithIdDataFrame =
        patternsDataFrame.withColumn("label_id", concat(col("label"), lit(","), col("id")))
      patternsWithIdDataFrame.createOrReplaceTempView("patterns_view")
      val sqlText =
        "SELECT label_id, flatten(collect_set(patterns)) AS patterns FROM patterns_view GROUP BY label_id"
      spark.sql(sqlText)
    } else {
      patternsDataFrame.createOrReplaceTempView("patterns_view")
      val sqlText =
        "SELECT label, flatten(collect_set(patterns)) AS patterns FROM patterns_view GROUP BY label"
      spark.sql(sqlText)
    }

  }

  private def storePatterns(
      patterns: Iterator[String],
      entity: String,
      patternsReaderWriter: PatternsReadWriter): Unit = {
    patterns.foreach(pattern => storePattern(pattern, entity, patternsReaderWriter))
  }

  private def storePattern(
      pattern: String,
      entity: String,
      patternsReaderWriter: PatternsReadWriter): Unit = {
    patternsReaderWriter.lookup(pattern).getOrElse(patternsReaderWriter.add(pattern, entity))
  }

  private def storeRegexPattern(
      pattern: Seq[String],
      entity: String,
      regexPatternsReaderWriter: RegexPatternsReadWriter): Unit = {

    if (!entities.contains(entity)) {
      entities = entities ++ Array(entity)
    }
    regexPatternsReaderWriter
      .lookup(entity)
      .getOrElse(regexPatternsReaderWriter.add(entity, pattern))
  }

  protected def createWriter(database: Name, connection: RocksDBConnection): StorageWriter[_] = {
    database match {
      case Database.ENTITY_PATTERNS => new PatternsReadWriter(connection)
      case Database.ENTITY_REGEX_PATTERNS => new RegexPatternsReadWriter(connection)
    }
  }

  override def indexStorage(fitDataset: Dataset[_], resource: Option[ExternalResource]): Unit = {
    if ($(useStorage)) {
      super.indexStorage(fitDataset, resource)
    }
  }

  private def computePatternsFromCSV(): Unit = {
    if ($(enablePatternRegex) || $(sentenceMatch)) {
      regexPatterns = ResourceHelper.parseKeyListValues($(patternsResource))
      entities = regexPatterns.keys.toArray
    } else {
      patterns = ResourceHelper.parseKeyValueText($(patternsResource)).flatMap {
        case (key, value) => Map(value -> key)
      }
    }
  }

  private def computePatterns(patterns: Seq[String], entity: String): Unit = {
    if ($(enablePatternRegex) || $(sentenceMatch)) {
      regexPatterns = regexPatterns ++ Map(entity -> patterns)
      if (!entities.contains(entity)) {
        entities = entities ++ Array(entity)
      }
    } else {
      patterns.foreach(pattern => this.patterns = this.patterns ++ Map(pattern -> entity))
    }
  }

}
