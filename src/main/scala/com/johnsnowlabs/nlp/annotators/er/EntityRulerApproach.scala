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
import com.johnsnowlabs.util.JsonParser
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.{col, collect_set, concat, flatten, lit}
import org.apache.spark.sql.types.{BooleanType, StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
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
  // TODO: add predefined rules for entities like email, phone, etc
  def this() = this(Identifiable.randomUID("ENTITY_RULER"))

  override val description: String = "Entity Ruler matches entities based on text patterns"

  private var entitiesForRegex: Array[String] = Array()
  private val keywordsPatterns: ArrayBuffer[EntityPattern] = ArrayBuffer()
  private var regexPatterns: Map[String, Seq[String]] = Map()

  /** Resource in JSON or CSV format to map entities to patterns (Default: `null`).
    *
    * @group param
    */
  val patternsResource: ExternalResourceParam = new ExternalResourceParam(
    this,
    "patternsResource",
    "Resource in JSON or CSV format to map entities to patterns")

  val sentenceMatch = new BooleanParam(
    this,
    "sentenceMatch",
    "Whether to find match at sentence level (regex only). True: sentence level. False: token level")

  /** Whether to use RocksDB storage to serialize patterns (Default: `true`).
    *
    * @group param
    */
  val useStorage =
    new BooleanParam(this, "useStorage", "Whether to use RocksDB storage to serialize patterns")

  val alphabet = new ExternalResourceParam(
    this,
    "alphabet",
    "Alphabet resource path to plain text file with all characters in a given alphabet")

  /** @group setParam */
  def setPatternsResource(
      path: String,
      readAs: ReadAs.Format,
      options: Map[String, String] = Map("format" -> "JSON")): this.type =
    set(patternsResource, ExternalResource(path, readAs, options))

  def setSentenceMatch(value: Boolean): this.type = set(sentenceMatch, value)

  /** @group setParam */
  def setUseStorage(value: Boolean): this.type = set(useStorage, value)

  /** @group setParam */
  def setAlphabetResource(path: String): this.type = {
    set(alphabet, ExternalResource(path, ReadAs.TEXT, Map()))
  }

  setDefault(
    storagePath -> ExternalResource("", ReadAs.TEXT, Map()),
    patternsResource -> null,
    useStorage -> false,
    sentenceMatch -> false,
    caseSensitive -> true,
    alphabet -> ExternalResource("english", ReadAs.TEXT, Map()))

  private val AVAILABLE_FORMATS = Array("JSON", "JSONL", "CSV")

  override def beforeTraining(spark: SparkSession): Unit = {
    validateParameters()
  }

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): EntityRulerModel = {

    val entityRuler = new EntityRulerModel()

    if ($(useStorage)) {
      entityRuler
        .setStorageRef($(storageRef))
        .setUseStorage($(useStorage))

    } else {
      storePatterns(None)
      val entityRulerFeatures = EntityRulerFeatures(Map(), regexPatterns)
      entityRuler
        .setUseStorage($(useStorage))
        .setEntityRulerFeatures(entityRulerFeatures)
    }

    var automaton: Option[AhoCorasickAutomaton] = None
    if (keywordsPatterns.nonEmpty) {
      val alphabet = EntityRulerUtil.loadAlphabet($(this.alphabet).path)
      automaton = Some(
        new AhoCorasickAutomaton(alphabet, keywordsPatterns.toArray, $(caseSensitive)))
    }

    entityRuler
      .setRegexEntities(entitiesForRegex)
      .setAhoCorasickAutomaton(automaton)

  }

  protected def index(
      fitDataset: Dataset[_],
      storageSourcePath: Option[String],
      readAs: Option[ReadAs.Value],
      writers: Map[Name, StorageWriter[_]],
      readOptions: Option[Map[String, String]]): Unit = {

    validateParameters()

    if ($(useStorage)) {
      val storageWriter =
        writers(Database.ENTITY_REGEX_PATTERNS).asInstanceOf[RegexPatternsReadWriter]
      storePatterns(Some(storageWriter))
    }

  }

  private def storePatterns(storageWriter: Option[RegexPatternsReadWriter]): Unit = {

    resourceFormats match {
      case "JSON&TEXT" => storePatternsFromJson(storageWriter)
      case "JSONL&TEXT" => storePatternsFromJsonl(storageWriter)
      case "JSON&SPARK" => storePatternsFromJSONDataFrame(storageWriter, "JSON")
      case "JSONL&SPARK" => storePatternsFromJSONDataFrame(storageWriter, "JSONL")
      case "CSV&TEXT" => storePatternsFromCSV(storageWriter)
      case "CSV&SPARK" => storeEntityPatternsFromCSVDataFrame(storageWriter)
      case _ @format => throw new IllegalArgumentException(s"format $format not available")
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

  private def storePatternsFromJson(storageReadWriter: Option[RegexPatternsReadWriter]): Unit = {

    val entityPatterns: Array[EntityPattern] = parseJSON()

    entityPatterns.foreach { entityPattern =>
      if (entityPattern.regex.getOrElse(false)) {
        storeEntityPattern(entityPattern, storageReadWriter)
      } else {
        keywordsPatterns.append(entityPattern)
      }
    }
  }

  private def storeEntityPattern(
      entityPattern: EntityPattern,
      storageReadWriter: Option[RegexPatternsReadWriter]): Unit = {
    val entity =
      if (entityPattern.id.isDefined) s"${entityPattern.label},${entityPattern.id.get}"
      else entityPattern.label
    storageReadWriter.getOrElse(None) match {
      case patternsWriter: PatternsReadWriter =>
        storePatterns(entityPattern.patterns.toIterator, entity, patternsWriter)
      case regexPatternsWriter: RegexPatternsReadWriter =>
        storeRegexPattern(entityPattern.patterns, entity, regexPatternsWriter)
      case None => {
        val isRegex = entityPattern.regex.getOrElse(false)
        computePatterns(entityPattern.patterns, isRegex, entity)
      }
    }
  }

  private def parseJSON(): Array[EntityPattern] = {
    val stream = ResourceHelper.getResourceStream($(patternsResource).path)
    val jsonContent = Source.fromInputStream(stream).mkString
    val entityPatterns: Array[EntityPattern] = JsonParser.parseArray[EntityPattern](jsonContent)

    entityPatterns
  }

  private def storePatternsFromJsonl(storageReadWriter: Option[RegexPatternsReadWriter]): Unit = {

    val sourceStream = ResourceHelper.SourceStream($(patternsResource).path)

    sourceStream.content.foreach(content =>
      content.foreach { line =>
        val entityPattern: EntityPattern = JsonParser.parseObject[EntityPattern](line)
        if (entityPattern.regex.getOrElse(false)) {
          storeEntityPattern(entityPattern, storageReadWriter)
        } else keywordsPatterns.append(entityPattern)
      })
  }

  private def storePatternsFromCSV(regexPatternsWriter: Option[RegexPatternsReadWriter]): Unit = {

    val delimiter: String = $(patternsResource).options("delimiter")
    val patternsLines = ResourceHelper.parseLines($(patternsResource))
    val regexList: ArrayBuffer[String] = ArrayBuffer()
    val keywords: mutable.Map[String, Seq[String]] = mutable.Map()
    val regexPatterns: mutable.Map[String, Seq[String]] = mutable.Map()
    var patternsHasRegex = false

    val groupByLabel =
      patternsLines.groupBy(pattern => EntityRulerUtil.splitString(pattern, delimiter)(0))
    groupByLabel.foreach { case (label, lines) =>
      lines.foreach { line =>
        val columns: Array[String] = EntityRulerUtil.splitString(line, delimiter)
        val pattern = columns(1)
        val isRegex = if (columns.length == 2) false else EntityRulerUtil.toBoolean(columns(2))

        if (isRegex) {
          regexList.append(pattern)
          patternsHasRegex = true
        } else {
          val patterns = keywords.getOrElse(label, Seq())
          keywords(label) = patterns ++ Seq(pattern)
        }
      }

      if (regexPatternsWriter.isEmpty) {
        regexPatterns(label) = regexList
      }

      if (patternsHasRegex && regexPatternsWriter.nonEmpty) {
        storeRegexPattern(regexList, label, regexPatternsWriter.get)
      }

      keywords.foreach { case (label, patterns) =>
        keywordsPatterns.append(EntityPattern(label, patterns))
      }
      keywords.clear()
    }

    if (regexPatternsWriter.isEmpty) {
      this.regexPatterns = regexPatterns.toMap
      if (patternsHasRegex) entitiesForRegex = regexPatterns.keys.toArray
    }

  }

  private def storeEntityPatternsFromCSVDataFrame(
      storageReadWriter: Option[RegexPatternsReadWriter]): Unit = {

    val patternOptions = $(patternsResource).options
    val patternsSchema = StructType(
      Array(
        StructField("label", StringType, nullable = false),
        StructField("pattern", StringType, nullable = false),
        StructField("regex", BooleanType, nullable = true)))

    val patternsDataFrame = spark.read
      .format(patternOptions("format"))
      .options(patternOptions)
      .option("delimiter", patternOptions("delimiter"))
      .schema(patternsSchema)
      .load($(patternsResource).path)
      .na
      .fill(value = false, Array("regex"))

    val groupedByPatternsDataFrame = patternsDataFrame
      .groupBy("label", "regex")
      .agg(collect_set("pattern").alias("patterns"))

    storeFromDataFrame(
      groupedByPatternsDataFrame,
      idFieldExist = false,
      regexFieldExist = true,
      storageReadWriter)

  }

  private def storePatternsFromJSONDataFrame(
      storageReadWriter: Option[RegexPatternsReadWriter],
      format: String): Unit = {

    val path = $(patternsResource).path

    val dataFrameReader = spark.read
    if (format.equals("JSON")) {
      dataFrameReader.option("multiline", "true")
    }

    var patternsDataFrame = dataFrameReader
      .json(path)

    val idField: Array[StructField] =
      patternsDataFrame.schema.fields.filter(field => field.name == "id")
    val regexField: Array[StructField] =
      patternsDataFrame.schema.fields.filter(field => field.name == "regex")

    if (regexField.isEmpty) {
      patternsDataFrame = patternsDataFrame.withColumn("regex", lit(false))
    } else {
      patternsDataFrame = patternsDataFrame.na.fill(value = false, Array("regex"))
    }
    if (idField.nonEmpty) patternsDataFrame.na.drop()

    storeFromDataFrame(
      patternsDataFrame,
      idField.nonEmpty,
      regexField.nonEmpty,
      storageReadWriter)
  }

  private def storeFromDataFrame(
      patternsDataFrame: DataFrame,
      idFieldExist: Boolean,
      regexFieldExist: Boolean,
      storageReadWriter: Option[RegexPatternsReadWriter]): Unit = {

    val regexPatternsDataFrame = patternsDataFrame.filter(col("regex") === true)
    val cleanedRegexPatternsDataFrame =
      cleanPatternsDataFrame(regexPatternsDataFrame, idFieldExist)

    cleanedRegexPatternsDataFrame.rdd.toLocalIterator.foreach { row =>
      val patterns = row.getAs[Seq[String]]("flatten_patterns")
      val entity =
        if (idFieldExist) row.getAs[String]("label_id") else row.getAs[String]("label")
      storageReadWriter.getOrElse(None) match {
        case patternsWriter: PatternsReadWriter =>
          storePatterns(patterns.toIterator, entity, patternsWriter)
        case regexPatternsWriter: RegexPatternsReadWriter =>
          storeRegexPattern(patterns, entity, regexPatternsWriter)
        case None => computePatterns(patterns, isRegex = true, entity)
      }
    }

    val keywordsDataFrame = patternsDataFrame.filter(col("regex") === false)
    val cleanedKeywordsDataFrame = cleanPatternsDataFrame(keywordsDataFrame, idFieldExist)

    cleanedKeywordsDataFrame.rdd.toLocalIterator.foreach { row =>
      val patterns = row.getAs[Seq[String]]("flatten_patterns")
      if (idFieldExist) {
        val labelId = row.getAs[String]("label_id")
        val label = labelId.split(",")(0)
        val id = labelId.split(",")(1)
        keywordsPatterns.append(EntityPattern(label, patterns, Some(id), Some(true)))
      } else {
        val label = row.getAs[String]("label")
        keywordsPatterns.append(EntityPattern(label, patterns, None, Some(true)))
      }

    }

  }

  private def cleanPatternsDataFrame(
      patternsDataFrame: DataFrame,
      idFieldExist: Boolean): DataFrame = {

    if (idFieldExist) {
      val patternsWithIdDataFrame =
        patternsDataFrame.withColumn("label_id", concat(col("label"), lit(","), col("id")))

      patternsWithIdDataFrame
        .groupBy("label_id")
        .agg(flatten(collect_set("patterns")).as("flatten_patterns"))
    } else {
      patternsDataFrame
        .groupBy("label")
        .agg(flatten(collect_set("patterns")).as("flatten_patterns"))
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

    if (!entitiesForRegex.contains(entity)) {
      entitiesForRegex = entitiesForRegex ++ Array(entity)
    }
    regexPatternsReaderWriter
      .lookup(entity)
      .getOrElse(regexPatternsReaderWriter.add(entity, pattern))
  }

  protected def createWriter(database: Name, connection: RocksDBConnection): StorageWriter[_] = {
    new RegexPatternsReadWriter(connection)
  }

  override def indexStorage(fitDataset: Dataset[_], resource: Option[ExternalResource]): Unit = {
    if ($(useStorage)) {
      super.indexStorage(fitDataset, resource)
    }
  }

  private def computePatterns(patterns: Seq[String], isRegex: Boolean, entity: String): Unit = {
    if (isRegex) {
      regexPatterns = regexPatterns ++ Map(entity -> patterns)
      if (!entitiesForRegex.contains(entity)) {
        entitiesForRegex = entitiesForRegex ++ Array(entity)
      }
    }
  }

  /** Input annotator types: DOCUMENT, TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT)
  override val optionalInputAnnotatorTypes: Array[String] = Array(TOKEN)

  /** Output annotator types: CHUNK
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = CHUNK

  override protected val databases: Array[Name] = EntityRulerModel.databases

}
