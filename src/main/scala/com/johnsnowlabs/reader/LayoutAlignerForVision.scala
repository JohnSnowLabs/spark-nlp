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
import com.johnsnowlabs.nlp.util.SparkNlpConfig
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{Metadata, MetadataBuilder, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.json4s.NoTypeHints
import org.json4s.jackson.Serialization
import scala.collection.Map

/** LayoutAlignerForVision aligns document chunks with nearby images and emits paired outputs: one
  * document annotation column, one image annotation column, and one prompt annotation column.
  */
class LayoutAlignerForVision(override val uid: String)
    extends Transformer
    with HasInputAnnotationCols
    with HasOutputAnnotationCol
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("LAYOUT_ALIGNER_VISION"))

  override val inputAnnotatorTypes: Array[String] =
    Array(AnnotatorType.DOCUMENT, AnnotatorType.IMAGE)

  val maxDistance: IntParam = new IntParam(
    this,
    "maxDistance",
    "Maximum vertical distance (px) to align image with paragraph")

  val paragraphSpacingY: IntParam =
    new IntParam(this, "paragraphSpacingY", "Vertical spacing heuristic used during parsing")

  val includeContextWindow: BooleanParam = new BooleanParam(
    this,
    "includeContextWindow",
    "Include paragraph +/-1 as context for floating images")

  val confidenceThreshold: DoubleParam =
    new DoubleParam(this, "confidenceThreshold", "Minimum confidence required to emit alignment")

  val explodeDocs: BooleanParam = new BooleanParam(
    this,
    "explodeDocs",
    "Whether to explode aligned doc/image pairs into separate rows")

  val mergeImagesPerChunk: BooleanParam = new BooleanParam(
    this,
    "mergeImagesPerChunk",
    "When true, keep one primary image output per paragraph and store all image matches in doc metadata")

  val addNeighborText: BooleanParam = new BooleanParam(
    this,
    "addNeighborText",
    "When true, prompt output includes the aligned text chunk as additional context")

  def setExplodeDocs(value: Boolean): this.type = set(explodeDocs, value)
  def setMergeImagesPerChunk(value: Boolean): this.type = set(mergeImagesPerChunk, value)
  def setAddNeighborText(value: Boolean): this.type = set(addNeighborText, value)

  setDefault(
    maxDistance -> 40,
    paragraphSpacingY -> 25,
    includeContextWindow -> true,
    confidenceThreshold -> 0.0,
    explodeDocs -> true,
    mergeImagesPerChunk -> false,
    addNeighborText -> false)

  private val outputDocTypeMetadata: Metadata =
    new MetadataBuilder().putString("annotatorType", AnnotatorType.DOCUMENT).build()
  private val outputImageTypeMetadata: Metadata =
    new MetadataBuilder().putString("annotatorType", AnnotatorType.IMAGE).build()
  private val outputPromptTypeMetadata: Metadata =
    new MetadataBuilder().putString("annotatorType", AnnotatorType.DOCUMENT).build()

  private val baseImagePrompt =
    "Describe in a short and easy to understand sentence what you see in the image"

  private case class ParagraphLayout(
      annotation: Annotation,
      y: Int,
      index: Int,
      slideIndex: Option[Int],
      pageNumber: Option[Int])
  private case class ParagraphKey(begin: Int, end: Int, index: Option[Int])
  private case class ImageKey(imageId: String, coord: String, imageType: String)
  private case class ImageLayout(
      annotation: AnnotationImage,
      y: Int,
      imageType: String,
      sourceFile: String,
      coord: String,
      slideIndex: Option[Int],
      pageNumber: Option[Int])
  private case class AlignedPair(
      doc: Annotation,
      image: AnnotationImage,
      paragraphKey: ParagraphKey,
      imageKey: ImageKey,
      distance: Int,
      confidence: Double,
      coord: String,
      imageType: String,
      sourceFile: String,
      matchStrategy: String,
      slideIndex: Option[Int])

  private val sameSlideFallbackConfidence = 0.25

  private def getOutputDocCol: String = s"${getOutputCol}_doc"
  private def getOutputImageCol: String = s"${getOutputCol}_image"
  private def getOutputPromptCol: String = s"${getOutputCol}_prompt"

  override def transform(dataset: Dataset[_]): DataFrame = {
    require(
      validateSchema(dataset.schema),
      s"Wrong or missing inputCols annotators in $uid.\n" +
        msgHelper(dataset.schema) +
        s"\nMake sure such annotators exist with types: ${inputAnnotatorTypes.mkString(", ")}")

    val inputDataFrame = dataset.toDF()
    val outputSchema = transformSchema(inputDataFrame.schema)
    val (docInputCol, imageInputCol) = resolveInputCols(inputDataFrame.schema)

    val outputDocCol = getOutputDocCol
    val outputImageCol = getOutputImageCol
    val outputPromptCol = getOutputPromptCol
    val baseColumnNames =
      outputSchema.fields
        .map(_.name)
        .filterNot(name =>
          name == outputDocCol || name == outputImageCol || name == outputPromptCol)
    val baseColumnIndexes = baseColumnNames.map(inputDataFrame.schema.fieldIndex)
    val docInputIndex = inputDataFrame.schema.fieldIndex(docInputCol)
    val imageInputIndex = inputDataFrame.schema.fieldIndex(imageInputCol)

    implicit val encoder: ExpressionEncoder[Row] =
      SparkNlpConfig.getEncoder(inputDataFrame, outputSchema)

    val mappedDataFrame = inputDataFrame.mapPartitions { rows =>
      rows.flatMap { row =>
        val textRows = Option(row.getAs[Seq[Row]](docInputIndex)).getOrElse(Seq.empty)
        val imageRows = Option(row.getAs[Seq[Row]](imageInputIndex)).getOrElse(Seq.empty)
        val pairs = alignPairs(toTextAnnotations(textRows), toImageAnnotations(imageRows))
        val baseValues = baseColumnIndexes.map(row.get)

        if ($(explodeDocs)) {
          pairs.iterator.map { case (doc, image) =>
            val prompt = buildPromptAnnotation(doc)
            Row.fromSeq(
              baseValues ++ Seq(
                Seq(annotationToRow(doc)),
                Seq(annotationImageToRow(image)),
                Seq(annotationToRow(prompt))))
          }
        } else {
          val docValues = pairs.map { case (doc, _) => annotationToRow(doc) }
          val imageValues = pairs.map { case (_, image) => annotationImageToRow(image) }
          val promptValues = pairs.map { case (doc, _) =>
            annotationToRow(buildPromptAnnotation(doc))
          }
          Iterator.single(Row.fromSeq(baseValues ++ Seq(docValues, imageValues, promptValues)))
        }
      }
    }

    val withInputMetadata = inputDataFrame.schema.fields
      .filter(field => mappedDataFrame.columns.contains(field.name))
      .foldLeft(mappedDataFrame) { (dataFrame, field) =>
        dataFrame.withColumn(field.name, dataFrame.col(field.name).as(field.name, field.metadata))
      }

    withInputMetadata
      .withColumn(outputDocCol, col(outputDocCol).as(outputDocCol, outputDocTypeMetadata))
      .withColumn(outputImageCol, col(outputImageCol).as(outputImageCol, outputImageTypeMetadata))
      .withColumn(
        outputPromptCol,
        col(outputPromptCol).as(outputPromptCol, outputPromptTypeMetadata))
  }

  override def transformSchema(schema: StructType): StructType = {
    require(
      validateSchema(schema),
      s"Wrong or missing inputCols annotators in $uid.\n" +
        msgHelper(schema) +
        s"\nMake sure such annotators exist with types: ${inputAnnotatorTypes.mkString(", ")}")

    val outDoc =
      StructField(getOutputDocCol, Annotation.arrayType, nullable = false, outputDocTypeMetadata)
    val outImage =
      StructField(
        getOutputImageCol,
        AnnotationImage.arrayType,
        nullable = false,
        outputImageTypeMetadata)
    val outPrompt =
      StructField(
        getOutputPromptCol,
        Annotation.arrayType,
        nullable = false,
        outputPromptTypeMetadata)

    val baseFields =
      schema.fields
        .filterNot(f =>
          f.name == getOutputDocCol || f.name == getOutputImageCol || f.name == getOutputPromptCol)
    StructType(baseFields ++ Array(outDoc, outImage, outPrompt))
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  private def validateSchema(schema: StructType): Boolean =
    inputAnnotatorTypes.forall(checkSchema(schema, _))

  private def resolveInputCols(schema: StructType): (String, String) = {
    val colsByType =
      getInputCols.map(name => name -> schema(name).metadata.getString("annotatorType")).toMap
    val docCol = colsByType.collectFirst {
      case (name, annotatorType) if annotatorType == AnnotatorType.DOCUMENT =>
        name
    }.get
    val imageCol =
      colsByType.collectFirst {
        case (name, annotatorType) if annotatorType == AnnotatorType.IMAGE =>
          name
      }.get
    (docCol, imageCol)
  }

  private def alignPairs(
      textAnnotations: Seq[Annotation],
      imageAnnotations: Seq[AnnotationImage]): Seq[(Annotation, AnnotationImage)] = {
    if (textAnnotations.isEmpty || imageAnnotations.isEmpty) return Seq.empty

    val paragraphLayout = textAnnotations.flatMap(extractParagraphLayout)
    if (paragraphLayout.isEmpty) return Seq.empty

    val imageLayout = imageAnnotations.flatMap(extractImageLayout)
    if (imageLayout.isEmpty) return Seq.empty

    val candidateAlignments = imageLayout.flatMap { image =>
      val sameSlideParagraphs = image.slideIndex
        .map(slideIndex => paragraphLayout.filter(_.slideIndex.contains(slideIndex)))
        .getOrElse(Seq.empty)
      val samePageParagraphs = if (sameSlideParagraphs.nonEmpty) {
        Seq.empty
      } else {
        image.pageNumber
          .map(pageNumber => paragraphLayout.filter(_.pageNumber.contains(pageNumber)))
          .getOrElse(Seq.empty)
      }
      val candidateParagraphs =
        if (sameSlideParagraphs.nonEmpty) sameSlideParagraphs
        else if (samePageParagraphs.nonEmpty) samePageParagraphs
        else paragraphLayout
      val paragraphByIndex = candidateParagraphs.map(p => p.index -> p).toMap
      val primaryMatchStrategy =
        if (sameSlideParagraphs.nonEmpty) "same_slide_distance"
        else if (samePageParagraphs.nonEmpty) "same_page_distance"
        else "distance"
      val hasScopedCandidates = sameSlideParagraphs.nonEmpty || samePageParagraphs.nonEmpty

      val closest = findClosestParagraph(image.y, candidateParagraphs)
      closest.toSeq.flatMap { closestLayout =>
        val candidates = buildCandidates(closestLayout, image.imageType, paragraphByIndex)
        val strictAlignments = candidates.flatMap { paragraph =>
          buildAlignedPair(
            image = image,
            paragraph = paragraph,
            matchStrategy = primaryMatchStrategy)
        }
        if (strictAlignments.nonEmpty) {
          Seq(selectBestAlignmentForImage(strictAlignments))
        } else if (hasScopedCandidates && candidates.nonEmpty) {
          val closestScoped =
            findClosestParagraph(image.y, candidates).getOrElse(candidates.head)
          val fallbackStrategy =
            if (sameSlideParagraphs.nonEmpty) "same_slide_fallback" else "same_page_fallback"
          buildAlignedPair(
            image = image,
            paragraph = closestScoped,
            matchStrategy = fallbackStrategy,
            skipMaxDistance = true,
            forcedConfidence = Some(sameSlideFallbackConfidence)).toSeq
        } else {
          Seq.empty
        }
      }
    }

    val uniqueAlignments = candidateAlignments
      .groupBy(alignmentIdentity)
      .values
      .map(group => group.minBy(alignmentSortKey))
      .toSeq
      .sortBy(alignmentSortKey)

    val outputAlignments =
      if (uniqueAlignments.isEmpty) Seq.empty
      else if (! $(mergeImagesPerChunk)) uniqueAlignments
      else {
        uniqueAlignments
          .groupBy(_.paragraphKey)
          .values
          .map(mergeParagraphAlignments)
          .toSeq
          .sortBy(alignmentSortKey)
      }

    outputAlignments.map(toOutputPair)
  }

  private def toOutputPair(alignment: AlignedPair): (Annotation, AnnotationImage) =
    (alignment.doc, alignment.image)

  private def selectBestAlignmentForImage(candidates: Seq[AlignedPair]): AlignedPair =
    candidates.minBy(imageAssignmentSortKey)

  private def alignmentIdentity(alignment: AlignedPair): (ParagraphKey, ImageKey, Int, Double) =
    (alignment.paragraphKey, alignment.imageKey, alignment.distance, alignment.confidence)

  private def alignmentSortKey(alignment: AlignedPair): (Int, Int, Int, Double, String) =
    (
      alignment.paragraphKey.begin,
      alignment.paragraphKey.end,
      alignment.distance,
      -alignment.confidence,
      alignment.coord)

  private def imageAssignmentSortKey(alignment: AlignedPair): (Int, Double, Int, Int, String) =
    (
      alignment.distance,
      -alignment.confidence,
      alignment.paragraphKey.begin,
      alignment.paragraphKey.end,
      alignment.coord)

  private def mergeParagraphAlignments(alignments: Seq[AlignedPair]): AlignedPair = {
    val sorted = alignments.sortBy(alignmentSortKey)
    val primary = sorted.head

    val uniqueByImage =
      sorted.groupBy(_.imageKey).values.map(_.head).toSeq.sortBy(alignmentSortKey)
    val imageMatches = uniqueByImage.map { alignment =>
      val slideIndex = alignment.slideIndex.map(_.toString).getOrElse("")
      Map(
        "image_id" -> alignment.sourceFile,
        "source_file" -> alignment.sourceFile,
        "coord" -> alignment.coord,
        "image_type" -> alignment.imageType,
        "distance" -> alignment.distance.toString,
        "confidence" -> alignment.confidence.toString,
        "match_strategy" -> alignment.matchStrategy,
        "slide_index" -> slideIndex)
    }

    implicit val formats = Serialization.formats(NoTypeHints)
    val imageMatchesJson = Serialization.write(imageMatches)
    val imageMetadataKeys =
      Set(
        "image_id",
        "source_file",
        "coord",
        "image_type",
        "distance",
        "confidence",
        "match_strategy",
        "slide_index")
    val baseMetadata = primary.doc.metadata.filterNot { case (key, _) =>
      imageMetadataKeys.contains(key)
    }
    val mergedDoc =
      primary.doc.copy(metadata = baseMetadata + ("image_matches" -> imageMatchesJson))
    primary.copy(doc = mergedDoc)
  }

  private def extractParagraphLayout(annotation: Annotation): Option[ParagraphLayout] = {
    val metadata: Map[String, String] = Option(annotation.metadata).getOrElse(Map.empty)
    val paragraphY = metadata
      .get("page_y")
      .flatMap(parseInt)
      .orElse(metadata.get("paragraph_y").flatMap(parseInt))
    for {
      y <- paragraphY
      idx <- metadata.get("paragraph_index").flatMap(parseInt)
    } yield ParagraphLayout(
      annotation,
      y,
      idx,
      extractSlideIndex(metadata),
      extractPageNumber(metadata))
  }

  private def extractImageLayout(annotation: AnnotationImage): Option[ImageLayout] = {
    val metadata: Map[String, String] = Option(annotation.metadata).getOrElse(Map.empty)
    val coord = metadata.getOrElse("coord", "")
    val sourceFile = Option(annotation.origin)
      .filter(_.nonEmpty)
      .orElse(metadata.get("source_file"))
      .orElse(metadata.get("origin"))
      .getOrElse("")
    val imageType = metadata.getOrElse("image_type", "unknown")
    val slideIndex = extractSlideIndex(metadata)
    val pageNumber = extractPageNumber(metadata)

    extractY(coord).map(y =>
      ImageLayout(annotation, y, imageType, sourceFile, coord, slideIndex, pageNumber))
  }

  private def parseInt(value: String): Option[Int] = {
    if (value == null || value.isEmpty) None
    else {
      try Some(value.toInt)
      catch { case _: NumberFormatException => None }
    }
  }

  private def extractY(coord: String): Option[Int] = {
    val pattern = """y\s*:\s*([0-9]+)""".r
    pattern.findFirstMatchIn(coord).map(_.group(1)).flatMap(parseInt)
  }

  private def extractSlideIndex(metadata: Map[String, String]): Option[Int] = {
    metadata
      .get("slide_index")
      .flatMap(parseInt)
      .orElse(metadata.get("domPath").flatMap(extractSlideIndexFromDomPath))
  }

  private def extractPageNumber(metadata: Map[String, String]): Option[Int] = {
    metadata
      .get("pageNumber")
      .flatMap(parseInt)
      .orElse(metadata.get("page_number").flatMap(parseInt))
  }

  private def extractSlideIndexFromDomPath(domPath: String): Option[Int] = {
    val pattern = """slide\[(\d+)\]""".r
    pattern.findFirstMatchIn(domPath).map(_.group(1)).flatMap(parseInt)
  }

  private def findClosestParagraph(
      imageY: Int,
      paragraphs: Seq[ParagraphLayout]): Option[ParagraphLayout] = {
    paragraphs.reduceOption { (left, right) =>
      val leftDistance = math.abs(imageY - left.y)
      val rightDistance = math.abs(imageY - right.y)
      if (leftDistance <= rightDistance) left else right
    }
  }

  private def buildCandidates(
      closest: ParagraphLayout,
      imageType: String,
      paragraphByIndex: Map[Int, ParagraphLayout]): Seq[ParagraphLayout] = {
    if (! $(includeContextWindow) || imageType != "floating") Seq(closest)
    else {
      val indices = Seq(closest.index - 1, closest.index, closest.index + 1)
      indices.distinct.flatMap(paragraphByIndex.get)
    }
  }

  private def buildAlignedPair(
      image: ImageLayout,
      paragraph: ParagraphLayout,
      matchStrategy: String,
      skipMaxDistance: Boolean = false,
      forcedConfidence: Option[Double] = None): Option[AlignedPair] = {
    val distance = math.abs(image.y - paragraph.y)
    if (!skipMaxDistance && distance > $(maxDistance)) return None

    val confidence = forcedConfidence.getOrElse(computeConfidence(distance))
    if (confidence < $(confidenceThreshold)) return None

    val slideIndex = paragraph.slideIndex.orElse(image.slideIndex)
    val slideMetadata =
      slideIndex
        .map(idx => Map("slide_index" -> idx.toString))
        .getOrElse(Map.empty[String, String])

    val docMetadata = paragraph.annotation.metadata ++ slideMetadata
    val doc = Annotation(
      annotatorType = AnnotatorType.DOCUMENT,
      begin = paragraph.annotation.begin,
      end = paragraph.annotation.end,
      result = paragraph.annotation.result,
      metadata = docMetadata)

    val imageMetadata = Option(image.annotation.metadata).getOrElse(Map.empty) ++ Map(
      "source_file" -> image.sourceFile,
      "distance" -> distance.toString,
      "confidence" -> confidence.toString,
      "match_strategy" -> matchStrategy,
      "paragraph_index" -> paragraph.index.toString,
      "paragraph_y" -> docMetadata.getOrElse(
        "paragraph_y",
        paragraph.y.toString)) ++ slideMetadata
    val pairedImage = image.annotation.copy(metadata = imageMetadata)

    val paragraphKey = ParagraphKey(doc.begin, doc.end, Some(paragraph.index))
    val imageKey = ImageKey(image.sourceFile, image.coord, image.imageType)

    Some(
      AlignedPair(
        doc = doc,
        image = pairedImage,
        paragraphKey = paragraphKey,
        imageKey = imageKey,
        distance = distance,
        confidence = confidence,
        coord = image.coord,
        imageType = image.imageType,
        sourceFile = image.sourceFile,
        matchStrategy = matchStrategy,
        slideIndex = slideIndex))
  }

  private def computeConfidence(distance: Int): Double = {
    if (distance <= 10) 0.95
    else if (distance <= $(paragraphSpacingY)) 0.75
    else 0.4
  }

  private def buildPromptAnnotation(doc: Annotation): Annotation = {
    val neighborText = Option(doc.result).getOrElse("")
    val promptText =
      if ($(addNeighborText) && neighborText.nonEmpty) {
        s"$baseImagePrompt and then summarize it along with this text: $neighborText"
      } else {
        baseImagePrompt
      }

    Annotation(
      annotatorType = AnnotatorType.DOCUMENT,
      begin = 0,
      end = promptText.length - 1,
      result = promptText,
      metadata = Map("prompt_source" -> "LayoutAlignerForVision"))
  }

  private def toTextAnnotations(rows: Seq[Row]): Seq[Annotation] =
    Option(rows).getOrElse(Seq.empty).map(Annotation(_))

  private def toImageAnnotations(rows: Seq[Row]): Seq[AnnotationImage] =
    Option(rows).getOrElse(Seq.empty).map(AnnotationImage(_))

  private def annotationToRow(annotation: Annotation): Row =
    Row(
      annotation.annotatorType,
      annotation.begin,
      annotation.end,
      annotation.result,
      annotation.metadata,
      annotation.embeddings)

  private def annotationImageToRow(annotation: AnnotationImage): Row =
    Row(
      annotation.annotatorType,
      annotation.origin,
      annotation.height,
      annotation.width,
      annotation.nChannels,
      annotation.mode,
      annotation.result,
      annotation.metadata,
      annotation.text)
}

object LayoutAlignerForVision extends DefaultParamsReadable[LayoutAlignerForVision]
