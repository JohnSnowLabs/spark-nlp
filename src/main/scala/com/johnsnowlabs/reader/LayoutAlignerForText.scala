/*
 * Copyright 2017-2026 John Snow Labs
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
package com.johnsnowlabs.reader

import com.johnsnowlabs.nlp._
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{Metadata, MetadataBuilder, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

import scala.collection.Map

/** LayoutAlignerForText rebuilds final text by combining document chunks and generated image
  * captions produced by multimodal models.
  *
  * It is designed to consume `aligned_doc` + `image_caption` pairs (for example, after
  * `LayoutAlignerForVision` + `AutoGGUFVisionModel`) and emit a coherent document where captions
  * are inserted around their best matching text blocks.
  *
  * DISCLAIMER: By default, input columns (`aligned_doc`, `image_caption`) are not preserved. This
  * stage groups and can explode rows while rebuilding text, so input row cardinality does not
  * reliably match output row cardinality. Keeping passthrough inputs in that mode can produce
  * misleading repeated values. Set `preserveColumns = true` to try to keep grouped input
  * annotations when you explicitly need them for inspection.
  */
class LayoutAlignerForText(override val uid: String)
    extends Transformer
    with HasInputAnnotationCols
    with HasOutputAnnotationCol
    with DefaultParamsWritable {

  /** Constructor used by Spark ML during deserialization. */
  def this() = this(Identifiable.randomUID("LAYOUT_ALIGNER_TEXT"))

  /** This stage consumes two DOCUMENT inputs: aligned text chunks and generated captions. */
  override val inputAnnotatorTypes: Array[String] =
    Array(AnnotatorType.DOCUMENT, AnnotatorType.DOCUMENT)

  /** Delimiter used when joining text/caption pieces into one rebuilt element string. */
  val joinDelimiter: Param[String] =
    new Param[String](this, "joinDelimiter", "Delimiter used to join rebuilt text segments")

  /** Threshold used to decide if inline captions should be prefixed before paragraph text. */
  val inlinePrefixThreshold: IntParam = new IntParam(
    this,
    "inlinePrefixThreshold",
    "Inline images with x <= threshold are inserted before the paragraph text")

  /** Controls whether rebuilt elements are emitted as one output row per element. */
  val explodeElements: BooleanParam = new BooleanParam(
    this,
    "explodeElements",
    "Whether to emit one output row per aligned text element")

  /** Controls whether input annotation columns are preserved in the final output.
    *
    * When `true`, input columns are preserved by aggregating all original input annotation rows
    * that contributed to a rebuilt group. When `false`, input annotation columns are dropped so
    * `outputCol` is the authoritative text output and stale passthrough values are avoided.
    */
  val preserveColumns: BooleanParam = new BooleanParam(
    this,
    "preserveColumns",
    "Whether to preserve input annotation columns using grouped aggregation")

  /** Sets the delimiter used between rebuilt pieces. */
  def setJoinDelimiter(value: String): this.type = set(joinDelimiter, value)

  /** Sets the inline X threshold for prefix insertion. */
  def setInlinePrefixThreshold(value: Int): this.type = set(inlinePrefixThreshold, value)

  /** Enables/disables one-row-per-element output mode. */
  def setExplodeElements(value: Boolean): this.type = set(explodeElements, value)

  /** Enables/disables preservation of input annotation columns. */
  def setPreserveColumns(value: Boolean): this.type = set(preserveColumns, value)

  setDefault(
    joinDelimiter -> "\n",
    inlinePrefixThreshold -> 10,
    explodeElements -> true,
    preserveColumns -> false)

  /** Output metadata required by Spark NLP to mark outputCol as DOCUMENT annotations. */
  private val outputDocTypeMetadata: Metadata =
    new MetadataBuilder().putString("annotatorType", AnnotatorType.DOCUMENT).build()
  private val fileNameColumn = "fileName"

  /** Aggregation state per group while rebuilding aligned text. */
  private case class GroupAccumulator(
      baseValues: Map[String, Any],
      pairs: Vector[(Annotation, Annotation)],
      preservedDocs: Vector[Row],
      preservedCaptions: Vector[Row])

  /** Internal, normalized representation of one `(doc, caption)` pair used by heuristics. */
  private case class PairRecord(
      doc: Annotation,
      caption: Annotation,
      elementId: String,
      paragraphIndex: Int,
      slideIndex: Int,
      docBegin: Int,
      imageType: String,
      coordX: Int,
      orderImageIndex: Int,
      distance: Int,
      confidence: Double,
      imageIdentity: Option[String],
      captionText: String,
      docText: String)

  /** Final block representation produced for each logical element/paragraph. */
  private case class ElementBlock(
      elementId: String,
      slideIndex: Int,
      paragraphIndex: Int,
      docBegin: Int,
      text: String,
      beforeCaptions: Seq[String],
      afterCaptions: Seq[String])

  /** Executes the transformation: 1) validate schema and resolve input columns 2) aggregate
    * `(aligned_doc, image_caption)` pairs 3) rebuild coherent text blocks 4) optionally preserve
    * original input columns 5) re-attach Spark NLP annotation metadata
    */
  override def transform(dataset: Dataset[_]): DataFrame = {
    require(
      validateSchema(dataset.schema),
      s"Wrong or missing inputCols annotators in $uid.\n" +
        msgHelper(dataset.schema) +
        s"\nMake sure such annotators exist with types: ${inputAnnotatorTypes.mkString(", ")}")

    val inputDataFrame = dataset.toDF()
    val outputSchema = transformSchema(inputDataFrame.schema)
    val (docInputCol, captionInputCol) = resolveInputCols(inputDataFrame.schema)
    val fileNameColumnIndex = resolveGroupColumnIndex(inputDataFrame.schema)
    val passthroughColumns =
      resolvePassthroughColumns(inputDataFrame.schema, docInputCol, captionInputCol)

    val outputColumn = getOutputCol
    val outputBaseColumns = outputSchema.fields.map(_.name).filterNot(_ == outputColumn)
    val docInputIndex = inputDataFrame.schema.fieldIndex(docInputCol)
    val captionInputIndex = inputDataFrame.schema.fieldIndex(captionInputCol)

    val groupedAccumulators = inputDataFrame.rdd
      .map { row =>
        val key = buildGroupKey(row, fileNameColumnIndex)
        key -> buildGroupAccumulator(row, passthroughColumns, docInputIndex, captionInputIndex)
      }
      .reduceByKey(mergeAccumulators)
      .values

    val rowsRdd = groupedAccumulators.flatMap { acc =>
      val rebuilt = rebuildTextAnnotations(acc.pairs)
      val baseValues =
        buildOutputBaseValues(acc, outputBaseColumns, docInputCol, captionInputCol)

      if ($(explodeElements)) {
        rebuilt.iterator.map(annotation =>
          Row.fromSeq(baseValues ++ Seq(Seq(annotationToRow(annotation)))))
      } else {
        Iterator.single(Row.fromSeq(baseValues ++ Seq(rebuilt.map(annotationToRow))))
      }
    }

    val outputDataFrame = dataset.sparkSession.createDataFrame(rowsRdd, outputSchema)

    val withInputMetadata = inputDataFrame.schema.fields
      .filter(field => outputDataFrame.columns.contains(field.name))
      .foldLeft(outputDataFrame) { (dataFrame, field) =>
        dataFrame.withColumn(field.name, dataFrame.col(field.name).as(field.name, field.metadata))
      }

    withInputMetadata.withColumn(
      outputColumn,
      col(outputColumn).as(outputColumn, outputDocTypeMetadata))
  }

  /** Computes the output schema:
    *   - always adds/overwrites `outputCol` as DOCUMENT annotations
    *   - drops input annotation columns unless `preserveColumns=true`
    */
  override def transformSchema(schema: StructType): StructType = {
    require(
      validateSchema(schema),
      s"Wrong or missing inputCols annotators in $uid.\n" +
        msgHelper(schema) +
        s"\nMake sure such annotators exist with types: ${inputAnnotatorTypes.mkString(", ")}")

    val (docInputCol, captionInputCol) = resolveInputCols(schema)
    val outputField =
      StructField(getOutputCol, Annotation.arrayType, nullable = false, outputDocTypeMetadata)

    val baseFields = schema.fields
      .filterNot(_.name == getOutputCol)
      .filter { field =>
        $(preserveColumns) || (field.name != docInputCol && field.name != captionInputCol)
      }

    StructType(baseFields ++ Array(outputField))
  }

  /** Standard Spark ML copy implementation. */
  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  /** Verifies exactly two DOCUMENT input columns are provided and present in schema. */
  private def validateSchema(schema: StructType): Boolean = {
    val inputCols = getInputCols
    inputCols.length == 2 &&
    inputCols.forall(schema.fieldNames.contains) &&
    inputCols.forall { colName =>
      val field = schema(colName)
      field.metadata.contains("annotatorType") &&
      field.metadata.getString("annotatorType") == AnnotatorType.DOCUMENT
    }
  }

  /** Resolves the two declared input column names in declaration order. */
  private def resolveInputCols(schema: StructType): (String, String) = {
    val DOC_INDEX = 0
    val CAPTION_INDEX = 1
    val inputCols = getInputCols
    require(inputCols.length == 2, s"$uid expects exactly two input columns")
    require(
      schema.fieldNames.contains(inputCols(DOC_INDEX)),
      s"Missing input column: ${inputCols(DOC_INDEX)}")
    require(
      schema.fieldNames.contains(inputCols(CAPTION_INDEX)),
      s"Missing input column: ${inputCols(CAPTION_INDEX)}")
    (inputCols(DOC_INDEX), inputCols(CAPTION_INDEX))
  }

  /** Resolves the required grouping column used to aggregate rows before rebuilding text. */
  private def resolveGroupColumnIndex(schema: StructType): Int = {
    require(
      schema.fieldNames.contains(fileNameColumn),
      s"$uid requires '$fileNameColumn' column to group rows before rebuilding text")
    schema.fieldIndex(fileNameColumn)
  }

  /** Resolves passthrough columns carried from input into output rows.
    *
    * Input annotation columns are excluded because they are either dropped
    * (`preserveColumns=false`) or rebuilt via grouped preservation (`preserveColumns=true`).
    */
  private def resolvePassthroughColumns(
      schema: StructType,
      docInputCol: String,
      captionInputCol: String): Seq[String] = {
    schema.fieldNames
      .filterNot(_ == getOutputCol)
      .filterNot(name => name == docInputCol || name == captionInputCol)
      .toSeq
  }

  /** Builds a grouping key from the required `fileName` column index. */
  private def buildGroupKey(row: Row, keyIndex: Int): String =
    Option(row.get(keyIndex)).map(_.toString).getOrElse("__NULL__")

  /** Creates one accumulator from one input row.
    *
    * The accumulator stores:
    *   - passthrough column values
    *   - zipped `(doc, caption)` pairs used for rebuilding
    *   - full raw input annotation arrays for optional preservation
    */
  private def buildGroupAccumulator(
      row: Row,
      passthroughColumns: Seq[String],
      docInputIndex: Int,
      captionInputIndex: Int): GroupAccumulator = {
    val docs = extractAnnotationRows(row, docInputIndex).toVector
    val captions = extractAnnotationRows(row, captionInputIndex).toVector
    val pairs = docs.map(Annotation(_)).zip(captions.map(Annotation(_))).toVector
    val baseValues = passthroughColumns.map(name => name -> row.getAs[Any](name)).toMap

    GroupAccumulator(
      baseValues = baseValues,
      pairs = pairs,
      preservedDocs = docs,
      preservedCaptions = captions)
  }

  /** Merges two group accumulators during reduction.
    *
    * Passthrough columns keep the first non-null value, while pair/preserved collections are
    * appended.
    */
  private def mergeAccumulators(
      left: GroupAccumulator,
      right: GroupAccumulator): GroupAccumulator = {
    val mergedBase = (left.baseValues.keySet ++ right.baseValues.keySet).map { key =>
      val leftValue = left.baseValues.getOrElse(key, null)
      val rightValue = right.baseValues.getOrElse(key, null)
      key -> (if (leftValue != null) leftValue else rightValue)
    }.toMap

    GroupAccumulator(
      baseValues = mergedBase,
      pairs = left.pairs ++ right.pairs,
      preservedDocs = left.preservedDocs ++ right.preservedDocs,
      preservedCaptions = left.preservedCaptions ++ right.preservedCaptions)
  }

  /** Extracts raw annotation rows from a row/column index, returning empty on null. */
  private def extractAnnotationRows(row: Row, columnIndex: Int): Seq[Row] =
    Option(row.getAs[Seq[Row]](columnIndex)).getOrElse(Seq.empty)

  /** Builds output base values in schema order, injecting preserved input columns when enabled.
    */
  private def buildOutputBaseValues(
      accumulator: GroupAccumulator,
      outputBaseColumns: Seq[String],
      docInputCol: String,
      captionInputCol: String): Seq[Any] = {
    outputBaseColumns.map {
      case name if $(preserveColumns) && name == docInputCol =>
        accumulator.preservedDocs
      case name if $(preserveColumns) && name == captionInputCol =>
        accumulator.preservedCaptions
      case name =>
        accumulator.baseValues.getOrElse(name, null)
    }
  }

  /** Converts aligned `(doc, caption)` pairs into rebuilt aligned text annotations. */
  private def rebuildTextAnnotations(pairs: Seq[(Annotation, Annotation)]): Seq[Annotation] = {
    if (pairs.isEmpty) {
      return Seq.empty
    }

    val records = pairs.map { case (doc, caption) => toPairRecord(doc, caption) }
    val bestImageAssignments = deduplicateByImageIdentity(records)
    val elementBlocks = buildElementBlocks(records, bestImageAssignments)
    val sortedBlocks =
      elementBlocks.sortBy(block =>
        (block.slideIndex, block.paragraphIndex, block.docBegin, block.elementId))

    sortedBlocks.zipWithIndex.map { case (block, index) =>
      val orderedPieces =
        (block.beforeCaptions ++ Seq(block.text) ++ block.afterCaptions).filter(_.nonEmpty)
      val deduplicatedPieces = collapseConsecutiveDuplicates(orderedPieces)
      val text = deduplicatedPieces.mkString($(joinDelimiter))

      val paragraphMetadata =
        if (block.paragraphIndex == Int.MaxValue) Map.empty[String, String]
        else Map("paragraph_index" -> block.paragraphIndex.toString)
      val slideMetadata =
        if (block.slideIndex == Int.MaxValue) Map.empty[String, String]
        else Map("slide_index" -> block.slideIndex.toString)

      Annotation(
        annotatorType = AnnotatorType.DOCUMENT,
        begin = 0,
        end = if (text.isEmpty) 0 else text.length - 1,
        result = text,
        metadata = Map(
          "sentence" -> index.toString,
          "layout_aligner" -> "LayoutAlignerForText",
          "piece_count" -> deduplicatedPieces.size.toString,
          "element_id" -> block.elementId) ++ paragraphMetadata ++ slideMetadata)
    }
  }

  /** Deduplicates captions that point to the same image identity, keeping best match quality. */
  private def deduplicateByImageIdentity(records: Seq[PairRecord]): Seq[PairRecord] = {
    val captionRecords = records.filter(_.captionText.nonEmpty)
    val (identified, notIdentified) = captionRecords.partition(_.imageIdentity.nonEmpty)
    val bestByImage = identified
      .groupBy(_.imageIdentity.get)
      .values
      .map { group =>
        group.minBy(r => (r.distance, -r.confidence, r.paragraphIndex, r.docBegin))
      }
      .toSeq

    bestByImage ++ notIdentified
  }

  /** Builds one text block per element and splits matched captions into before/after buckets. */
  private def buildElementBlocks(
      allRecords: Seq[PairRecord],
      captionAssignments: Seq[PairRecord]): Seq[ElementBlock] = {
    val captionsByElement = captionAssignments.groupBy(_.elementId)

    allRecords
      .groupBy(_.elementId)
      .values
      .toSeq
      .map { group =>
        val representative = group.minBy(r => (r.slideIndex, r.paragraphIndex, r.docBegin))
        val sortedCaptions =
          captionsByElement
            .getOrElse(representative.elementId, Seq.empty)
            .sortBy(r => (r.orderImageIndex, r.distance, -r.confidence, r.captionText))

        val distinctCaptions = sortedCaptions
          .foldLeft(Vector.empty[PairRecord], Set.empty[String]) { case ((acc, seen), record) =>
            val normalized = normalize(record.captionText)
            if (normalized.isEmpty || seen.contains(normalized)) (acc, seen)
            else (acc :+ record, seen + normalized)
          }
          ._1

        val (before, after) = distinctCaptions.partition(shouldInsertBeforeText)

        ElementBlock(
          elementId = representative.elementId,
          slideIndex = representative.slideIndex,
          paragraphIndex = representative.paragraphIndex,
          docBegin = representative.docBegin,
          text = representative.docText,
          beforeCaptions = before.map(_.captionText),
          afterCaptions = after.map(_.captionText))
      }
  }

  /** Inline images anchored near the left margin are interpreted as prefix content. */
  private def shouldInsertBeforeText(record: PairRecord): Boolean =
    record.imageType.equalsIgnoreCase("inline") && record.coordX <= $(inlinePrefixThreshold)

  /** Removes consecutive duplicates while preserving original order. */
  private def collapseConsecutiveDuplicates(pieces: Seq[String]): Seq[String] =
    pieces.foldLeft(Vector.empty[String]) { (acc, current) =>
      if (acc.lastOption.contains(current)) acc else acc :+ current
    }

  /** Converts raw doc/caption annotations into a normalized `PairRecord` for ordering rules. */
  private def toPairRecord(doc: Annotation, caption: Annotation): PairRecord = {
    val docMetadata: Map[String, String] = Option(doc.metadata).getOrElse(Map.empty)
    val captionMetadata: Map[String, String] = Option(caption.metadata).getOrElse(Map.empty)

    val paragraphIndex = getInt(docMetadata, "paragraph_index").getOrElse(Int.MaxValue)
    val slideIndex = getInt(docMetadata, "slide_index")
      .orElse(getInt(captionMetadata, "slide_index"))
      .getOrElse(Int.MaxValue)

    val orderImageIndex = getInt(captionMetadata, "orderImageIndex").getOrElse(Int.MaxValue)
    val distance = getInt(captionMetadata, "distance").getOrElse(Int.MaxValue)
    val confidence = getDouble(captionMetadata, "confidence").getOrElse(0.0)
    val coord = captionMetadata.getOrElse("coord", "")
    val coordX = extractCoordX(coord).getOrElse(Int.MaxValue)
    val imageType = captionMetadata.getOrElse("image_type", "")
    val elementId =
      docMetadata.getOrElse("element_id", s"${doc.begin}:${doc.end}:${normalize(doc.result)}")
    val captionText = Option(caption.result).map(_.trim).getOrElse("")
    val docText = Option(doc.result).map(_.trim).getOrElse("")

    val sourceFile = captionMetadata.getOrElse("source_file", "")
    val imageIdentityRaw =
      Seq(
        sourceFile,
        captionMetadata.getOrElse("slide_index", ""),
        captionMetadata.getOrElse("orderImageIndex", ""),
        coord).mkString("|")
    val imageIdentity =
      if (imageIdentityRaw.replace("|", "").trim.nonEmpty) Some(imageIdentityRaw) else None

    PairRecord(
      doc = doc,
      caption = caption,
      elementId = elementId,
      paragraphIndex = paragraphIndex,
      slideIndex = slideIndex,
      docBegin = doc.begin,
      imageType = imageType,
      coordX = coordX,
      orderImageIndex = orderImageIndex,
      distance = distance,
      confidence = confidence,
      imageIdentity = imageIdentity,
      captionText = captionText,
      docText = docText)
  }

  /** Safely extracts an integer metadata value. */
  private def getInt(metadata: Map[String, String], key: String): Option[Int] =
    metadata.get(key).flatMap(parseInt)

  /** Safely extracts a double metadata value. */
  private def getDouble(metadata: Map[String, String], key: String): Option[Double] =
    metadata.get(key).flatMap(parseDouble)

  /** Parses an integer value without throwing on malformed input. */
  private def parseInt(value: String): Option[Int] = {
    if (value == null || value.isEmpty) None
    else {
      try Some(value.toInt)
      catch { case _: NumberFormatException => None }
    }
  }

  /** Parses a double value without throwing on malformed input. */
  private def parseDouble(value: String): Option[Double] = {
    if (value == null || value.isEmpty) None
    else {
      try Some(value.toDouble)
      catch { case _: NumberFormatException => None }
    }
  }

  /** Extracts image x-coordinate from serialized coord metadata like `{x:12,y:30}`. */
  private def extractCoordX(coord: String): Option[Int] = {
    val pattern = """x\s*:\s*(-?[0-9]+)""".r
    pattern.findFirstMatchIn(coord).map(_.group(1)).flatMap(parseInt)
  }

  /** Produces a lowercase trimmed representation used for robust duplicate checks. */
  private def normalize(value: String): String =
    Option(value).map(_.trim.toLowerCase).getOrElse("")

  /** Converts an Annotation to Spark SQL Row representation expected by Annotation.arrayType. */
  private def annotationToRow(annotation: Annotation): Row =
    Row(
      annotation.annotatorType,
      annotation.begin,
      annotation.end,
      annotation.result,
      annotation.metadata,
      annotation.embeddings)
}

object LayoutAlignerForText extends DefaultParamsReadable[LayoutAlignerForText]
