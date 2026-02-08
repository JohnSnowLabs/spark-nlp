package com.johnsnowlabs.reader

import com.johnsnowlabs.nlp._
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{array, col, explode, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.json4s.NoTypeHints
import org.json4s.jackson.Serialization

/**
* LayoutAligner infers relationships between document elements (text, images, tables) using layout metadata produced by Spark NLP readers.
* */
class LayoutAligner(override val uid: String)
  extends AnnotatorModel[LayoutAligner]
    with HasSimpleAnnotate[LayoutAligner] {

  def this() = this(Identifiable.randomUID("LAYOUT_ALIGNER"))

  override val inputAnnotatorTypes: Array[AnnotatorType] =
    Array(AnnotatorType.DOCUMENT, AnnotatorType.IMAGE)

  override val outputAnnotatorType: AnnotatorType = AnnotatorType.CHUNK

  val maxDistance: IntParam = new IntParam(
    this,
    "maxDistance",
    "Maximum vertical distance (px) to align image with paragraph"
  )

  val paragraphSpacingY: IntParam = new IntParam(
    this,
    "paragraphSpacingY",
    "Vertical spacing heuristic used during parsing"
  )

  val includeContextWindow: BooleanParam = new BooleanParam(
    this,
    "includeContextWindow",
    "Include paragraph ±1 as context for floating images"
  )

  val confidenceThreshold: DoubleParam = new DoubleParam(
    this,
    "confidenceThreshold",
    "Minimum confidence required to emit alignment"
  )

  val explodeDocs: BooleanParam =
    new BooleanParam(this, "explodeDocs", "Whether to explode aligned chunks into separate rows")

  val mergeImagesPerChunk: BooleanParam = new BooleanParam(
    this,
    "mergeImagesPerChunk",
    "When true, emit one chunk per paragraph and store images as a JSON array in metadata"
  )

  def setExplodeDocs(value: Boolean): this.type = set(explodeDocs, value)
  def setMergeImagesPerChunk(value: Boolean): this.type = set(mergeImagesPerChunk, value)

  setDefault(
    maxDistance -> 40,
    paragraphSpacingY -> 25,
    includeContextWindow -> true,
    confidenceThreshold -> 0.0,
    explodeDocs -> false,
    mergeImagesPerChunk -> true
  )

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val paragraphs = annotations.filter(_.annotatorType == AnnotatorType.DOCUMENT)
    val images = annotations.filter(_.annotatorType == AnnotatorType.IMAGE)

    if (paragraphs.isEmpty || images.isEmpty) return Seq.empty

    val paragraphLayout = paragraphs.flatMap(extractParagraphLayout)
    if (paragraphLayout.isEmpty) return Seq.empty

    val paragraphByIndex = paragraphLayout.map(p => p.index -> p).toMap
    val candidateAlignments = images.flatMap { img =>
      val imageYOpt = img.metadata.get("coord").flatMap(extractY)
      imageYOpt.toSeq.flatMap { imageY =>
        val closest = findClosestParagraph(imageY, paragraphLayout)
        closest.toSeq.flatMap { closestLayout =>
          val imageType = img.metadata.getOrElse("image_type", "unknown")
          val candidates = buildCandidates(closestLayout, imageType, paragraphByIndex)
          candidates.flatMap { candidate =>
            buildAlignment(img, candidate, imageY, imageType)
          }
        }
      }
    }

    if (candidateAlignments.isEmpty) Seq.empty
    else if (!$(mergeImagesPerChunk)) candidateAlignments
    else {
      candidateAlignments
        .groupBy(keyForParagraph)
        .values
        .map(mergeAlignments)
        .toSeq
    }
  }

  override protected def _transform(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): DataFrame = {
    require(
      validate(dataset.schema),
      s"Wrong or missing inputCols annotators in $uid.\n" +
        msgHelper(dataset.schema) +
        s"\nMake sure such annotators exist in your pipeline, " +
        s"with the right output names and that they have following annotator types: " +
        s"${inputAnnotatorTypes.mkString(", ")}")

    val inputCols = getInputCols
    require(
      inputCols.length == 2,
      s"LayoutAligner expects 2 input cols (document and image), got ${inputCols.length}")

    val inputDataset = beforeAnnotate(dataset)
    val processedDataset = inputDataset.withColumn(
      getOutputCol,
      wrapColumnMetadata(
        dfAnnotateDual(inputDataset.col(inputCols(0)), inputDataset.col(inputCols(1)))))

    afterAnnotate(processedDataset)
  }

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    if ($(explodeDocs)) {
      dataset
        .select(dataset.columns.filterNot(_ == getOutputCol).map(col) :+ explode(
          col(getOutputCol)).as("_tmp"): _*)
        .withColumn(
          getOutputCol,
          array(col("_tmp"))
            .as(getOutputCol, dataset.schema.fields.find(_.name == getOutputCol).get.metadata))
        .drop("_tmp")
    } else dataset
  }

  private def dfAnnotateDual: UserDefinedFunction =
    udf { (textRows: Seq[Row], imageRows: Seq[Row]) =>
      val textAnnotations = toTextAnnotations(textRows)
      val imageAnnotations = toImageAnnotations(imageRows)
      annotate(textAnnotations ++ imageAnnotations)
    }

  private def toTextAnnotations(rows: Seq[Row]): Seq[Annotation] = {
    if (rows == null) Seq.empty
    else rows.map(Annotation(_))
  }

  private def toImageAnnotations(rows: Seq[Row]): Seq[Annotation] = {
    if (rows == null) Seq.empty
    else {
      rows.map { row =>
        val image = AnnotationImage(row)
        val baseMetadata = Option(image.metadata).getOrElse(Map.empty)
        val metadata =
          if (image.origin == null || image.origin.isEmpty) baseMetadata
          else baseMetadata + ("origin" -> image.origin)
        Annotation(
          annotatorType = AnnotatorType.IMAGE,
          begin = 0,
          end = 0,
          result = Option(image.text).getOrElse(""),
          metadata = metadata)
      }
    }
  }

  private case class ParagraphLayout(annotation: Annotation, y: Int, index: Int)

  private def extractParagraphLayout(annotation: Annotation): Option[ParagraphLayout] = {
    for {
      y <- annotation.metadata.get("paragraph_y").flatMap(parseInt)
      idx <- annotation.metadata.get("paragraph_index").flatMap(parseInt)
    } yield ParagraphLayout(annotation, y, idx)
  }

  private def parseInt(value: String): Option[Int] = {
    if (value == null || value.isEmpty) None
    else {
      try {
        Some(value.toInt)
      } catch {
        case _: NumberFormatException => None
      }
    }
  }

  private def parseDouble(value: String): Option[Double] = {
    if (value == null || value.isEmpty) None
    else {
      try {
        Some(value.toDouble)
      } catch {
        case _: NumberFormatException => None
      }
    }
  }

  private def extractY(coord: String): Option[Int] = {
    val pattern = """y\s*:\s*([0-9]+)""".r
    pattern.findFirstMatchIn(coord).map(_.group(1)).flatMap(parseInt)
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
    if (!$(includeContextWindow) || imageType != "floating") Seq(closest)
    else {
      val indices = Seq(closest.index - 1, closest.index, closest.index + 1)
      indices.distinct.flatMap(paragraphByIndex.get)
    }
  }

  private def buildAlignment(
      image: Annotation,
      paragraph: ParagraphLayout,
      imageY: Int,
      imageType: String): Option[Annotation] = {
    val distance = math.abs(imageY - paragraph.y)
    if (distance > $(maxDistance)) return None

    val confidence = computeConfidence(distance)
    if (confidence < $(confidenceThreshold)) return None

    val coordValue = image.metadata.getOrElse("coord", "")
    val sourceFile = image.metadata.getOrElse("origin", "")
    Some(
      Annotation(
        annotatorType = AnnotatorType.CHUNK,
        begin = paragraph.annotation.begin,
        end = paragraph.annotation.end,
        result = paragraph.annotation.result,
        metadata = Map(
          "image_id" -> sourceFile,
          "source_file" -> sourceFile,
          "coord" -> coordValue,
          "paragraph_index" -> paragraph.index.toString,
          "paragraph_y" -> paragraph.y.toString,
          "distance" -> distance.toString,
          "confidence" -> confidence.toString,
          "image_type" -> imageType
        )
      )
    )
  }

  private def computeConfidence(distance: Int): Double = {
    if (distance <= 10) 0.95
    else if (distance <= $(paragraphSpacingY)) 0.75
    else 0.4
  }

  private case class ParagraphKey(begin: Int, end: Int, index: Option[Int])

  private def keyForParagraph(annotation: Annotation): ParagraphKey = {
    val index = annotation.metadata.get("paragraph_index").flatMap(parseInt)
    ParagraphKey(annotation.begin, annotation.end, index)
  }

  private case class ImageKey(
      imageId: String,
      coord: String,
      imageType: String)

  private def keyForImage(annotation: Annotation): ImageKey = {
    ImageKey(
      annotation.metadata.getOrElse("image_id", ""),
      annotation.metadata.getOrElse("coord", ""),
      annotation.metadata.getOrElse("image_type", "unknown")
    )
  }

  private def selectBestAlignment(alignments: Seq[Annotation]): Annotation = {
    alignments.minBy { alignment =>
      val distance = alignment.metadata.get("distance").flatMap(parseInt).getOrElse(Int.MaxValue)
      val confidence = alignment.metadata.get("confidence").flatMap(parseDouble).getOrElse(0.0)
      (distance, -confidence)
    }
  }

  private def mergeAlignments(alignments: Seq[Annotation]): Annotation = {
    val best = selectBestAlignment(alignments)
    val unique = alignments.groupBy(keyForImage).values.map(_.head).toSeq

    implicit val formats = Serialization.formats(NoTypeHints)
    val images = unique.map { alignment =>
      Map(
        "image_id" -> alignment.metadata.getOrElse("image_id", ""),
        "source_file" -> alignment.metadata.getOrElse("source_file", ""),
        "coord" -> alignment.metadata.getOrElse("coord", ""),
        "image_type" -> alignment.metadata.getOrElse("image_type", "unknown"),
        "distance" -> alignment.metadata.getOrElse("distance", ""),
        "confidence" -> alignment.metadata.getOrElse("confidence", "")
      )
    }
    val json = Serialization.write(images)
    val imageKeys = Set(
      "image_id",
      "source_file",
      "coord",
      "image_type",
      "distance",
      "confidence"
    )
    val baseMetadata = best.metadata.filterNot { case (key, _) => imageKeys.contains(key) }

    Annotation(
      annotatorType = AnnotatorType.CHUNK,
      begin = best.begin,
      end = best.end,
      result = best.result,
      metadata = baseMetadata + ("image_matches" -> json)
    )
  }

}

object LayoutAligner extends DefaultParamsReadable[LayoutAligner]
