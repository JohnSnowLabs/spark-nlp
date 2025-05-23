package com.johnsnowlabs.ml.ai.util

import scala.util.matching.Regex
import java.awt.image.BufferedImage
import java.awt.{Color, Graphics2D, BasicStroke, Font}
import java.io.ByteArrayOutputStream
import javax.imageio.ImageIO
import java.util.Base64

object Florence2Utils {

  // Task prompts without additional inputs
  val taskPromptsWithoutInputs: Map[String, String] = Map(
    "<OCR>" -> "What is the text in the image?",
    "<OCR_WITH_REGION>" -> "What is the text in the image, with regions?",
    "<CAPTION>" -> "What does the image describe?",
    "<DETAILED_CAPTION>" -> "Describe in detail what is shown in the image.",
    "<MORE_DETAILED_CAPTION>" -> "Describe with a paragraph what is shown in the image.",
    "<OD>" -> "Locate the objects with category name in the image.",
    "<DENSE_REGION_CAPTION>" -> "Locate the objects in the image, with their descriptions.",
    "<REGION_PROPOSAL>" -> "Locate the region proposals in the image.")

  // Task prompts with additional input
  val taskPromptsWithInput: Map[String, String] = Map(
    "<CAPTION_TO_PHRASE_GROUNDING>" -> "Locate the phrases in the caption: {input}",
    "<REFERRING_EXPRESSION_SEGMENTATION>" -> "Locate {input} in the image with mask",
    "<REGION_TO_SEGMENTATION>" -> "What is the polygon mask of region {input}",
    "<OPEN_VOCABULARY_DETECTION>" -> "Locate {input} in the image.",
    "<REGION_TO_CATEGORY>" -> "What is the region {input}?",
    "<REGION_TO_DESCRIPTION>" -> "What does the region {input} describe?",
    "<REGION_TO_OCR>" -> "What text is in the region {input}?")

  // Preprocessing: construct prompts from task tokens
  def constructPrompts(texts: Seq[String]): Seq[String] = {
    texts.map { text =>
      // 1. fixed task prompts without additional inputs
      taskPromptsWithoutInputs
        .collectFirst {
          case (taskToken, taskPrompt) if text == taskToken => taskPrompt
        }
        .orElse {
          // 2. task prompts with additional inputs
          taskPromptsWithInput.collectFirst {
            case (taskToken, taskPrompt) if text.contains(taskToken) =>
              taskPrompt.replace("{input}", text.replace(taskToken, "").trim)
          }
        }
        .getOrElse(text)
    }
  }

  // Case classes for post-processing results
  case class BBox(bbox: Seq[Double], catName: String)
  case class OCRInstance(quadBox: Seq[Double], text: String)
  case class PhraseGroundingInstance(bbox: Seq[Seq[Double]], catName: String)
  case class PolygonInstance(
      polygons: Seq[Seq[Double]],
      catName: String,
      bbox: Option[Seq[Double]] = None)

  sealed trait Florence2Result
  case class PureTextResult(text: String) extends Florence2Result
  case class BBoxesResult(bboxes: Seq[BBox]) extends Florence2Result
  case class OCRResult(instances: Seq[OCRInstance]) extends Florence2Result
  case class PhraseGroundingResult(instances: Seq[PhraseGroundingInstance])
      extends Florence2Result
  case class PolygonsResult(instances: Seq[PolygonInstance]) extends Florence2Result
  case class MixedResult(
      bboxes: Seq[BBox],
      bboxesLabels: Seq[String],
      polygons: Seq[Seq[Double]],
      polygonsLabels: Seq[String])
      extends Florence2Result

  // Post-processing: parse model output
  def postProcessGeneration(
      text: String,
      task: String,
      imageSize: (Int, Int)): Florence2Result = {
    val taskType = taskAnswerPostProcessingType.getOrElse(task, "pure_text")
    taskType match {
      case "pure_text" =>
        PureTextResult(text.replace("<s>", "").replace("</s>", ""))
      case t if Set("od", "description_with_bboxes", "bboxes").contains(t) =>
        val instances = parseOD(text, imageSize)
        BBoxesResult(instances)
      case "ocr" =>
        val instances = parseOCR(text, imageSize)
        OCRResult(instances)
      case "phrase_grounding" =>
        val instances = parsePhraseGrounding(text, imageSize)
        PhraseGroundingResult(instances)
      case t if Set("description_with_polygons", "polygons").contains(t) =>
        val instances = parsePolygons(text, imageSize)
        PolygonsResult(instances)
      case "description_with_bboxes_or_polygons" =>
        if (text.contains("<poly>")) {
          val instances = parsePolygons(text, imageSize)
          PolygonsResult(instances)
        } else {
          val instances = parseOD(text, imageSize)
          BBoxesResult(instances)
        }
      case _ =>
        throw new IllegalArgumentException(s"Unknown task answer post processing type: $taskType")
    }
  }

  // Mapping from task to post-processing type
  val taskAnswerPostProcessingType: Map[String, String] = Map(
    "<OCR>" -> "pure_text",
    "<OCR_WITH_REGION>" -> "ocr",
    "<CAPTION>" -> "pure_text",
    "<DETAILED_CAPTION>" -> "pure_text",
    "<MORE_DETAILED_CAPTION>" -> "pure_text",
    "<OD>" -> "description_with_bboxes",
    "<DENSE_REGION_CAPTION>" -> "description_with_bboxes",
    "<CAPTION_TO_PHRASE_GROUNDING>" -> "phrase_grounding",
    "<REFERRING_EXPRESSION_SEGMENTATION>" -> "polygons",
    "<REGION_TO_SEGMENTATION>" -> "polygons",
    "<OPEN_VOCABULARY_DETECTION>" -> "description_with_bboxes_or_polygons",
    "<REGION_TO_CATEGORY>" -> "pure_text",
    "<REGION_TO_DESCRIPTION>" -> "pure_text",
    "<REGION_TO_OCR>" -> "pure_text",
    "<REGION_PROPOSAL>" -> "bboxes")

  // --- Parsing helpers ---

  // Parse object detection (bboxes)
  def parseOD(text: String, imageSize: (Int, Int)): Seq[BBox] = {
    val pattern = new Regex(
      "([a-zA-Z0-9 ]+)<loc_([0-9]+)><loc_([0-9]+)><loc_([0-9]+)><loc_([0-9]+)>")
    pattern
      .findAllMatchIn(text)
      .map { m =>
        val cat = m.group(1).trim.toLowerCase
        val bins = (2 to 5).map(i => m.group(i).toInt)
        val bbox = dequantizeBox(bins, imageSize)
        BBox(bbox, cat)
      }
      .toSeq
  }

  // Parse OCR
  def parseOCR(text: String, imageSize: (Int, Int)): Seq[OCRInstance] = {
    val pattern = new Regex(
      "(.+?)<loc_([0-9]+)><loc_([0-9]+)><loc_([0-9]+)><loc_([0-9]+)><loc_([0-9]+)><loc_([0-9]+)><loc_([0-9]+)><loc_([0-9]+)>")
    pattern
      .findAllMatchIn(text.replace("<s>", ""))
      .map { m =>
        val ocrText = m.group(1).trim
        val quadBoxBins = (2 to 9).map(i => m.group(i).toInt)
        val quadBox = dequantizeCoordinates(quadBoxBins, imageSize)
        OCRInstance(quadBox, ocrText)
      }
      .toSeq
  }

  // Parse phrase grounding
  def parsePhraseGrounding(text: String, imageSize: (Int, Int)): Seq[PhraseGroundingInstance] = {
    val phrasePattern = new Regex("([^<]+(?:<loc_\\d+>){4,})")
    val boxPattern = new Regex("<loc_([0-9]+)><loc_([0-9]+)><loc_([0-9]+)><loc_([0-9]+)>")
    phrasePattern
      .findAllMatchIn(text.replace("<s>", "").replace("</s>", "").replace("<pad>", ""))
      .flatMap { m =>
        val phraseText = m.group(1)
        val phrase = phraseText.takeWhile(_ != '<').trim
        val bboxes = boxPattern
          .findAllMatchIn(phraseText)
          .map { b =>
            val bins = (1 to 4).map(i => b.group(i).toInt)
            dequantizeBox(bins, imageSize)
          }
          .toSeq
        if (phrase.nonEmpty && bboxes.nonEmpty) Some(PhraseGroundingInstance(bboxes, phrase))
        else None
      }
      .toSeq
  }

  // Parse polygons
  def parsePolygons(text: String, imageSize: (Int, Int)): Seq[PolygonInstance] = {
    val polygonStart = "<poly>"
    val polygonEnd = "</poly>"
    val polygonSep = "<sep>"
    val phrasePattern = new Regex(
      s"([^<]+(?:<loc_\\d+>|$polygonSep|$polygonStart|$polygonEnd){4,})")
    val boxPattern = new Regex("<loc_([0-9]+)><loc_([0-9]+)><loc_([0-9]+)><loc_([0-9]+)>")
    phrasePattern
      .findAllMatchIn(text.replace("<s>", "").replace("</s>", "").replace("<pad>", ""))
      .flatMap { m =>
        val phraseText = m.group(1)
        val phrase = phraseText.takeWhile(_ != '<').trim
        val polygons = boxPattern
          .findAllMatchIn(phraseText)
          .map { b =>
            val bins = (1 to 4).map(i => b.group(i).toInt)
            dequantizeBox(bins, imageSize)
          }
          .toSeq
        if (phrase.nonEmpty && polygons.nonEmpty) Some(PolygonInstance(polygons, phrase))
        else None
      }
      .toSeq
  }

  // --- Quantization helpers ---
  // These are simplified versions; you may want to match the Python logic more closely for edge cases.
  def dequantizeBox(bins: Seq[Int], imageSize: (Int, Int)): Seq[Double] = {
    // bins: xmin, ymin, xmax, ymax
    val (w, h) = imageSize
    val binsW = 1000.0
    val binsH = 1000.0
    val sizePerBinW = w / binsW
    val sizePerBinH = h / binsH
    val Array(xmin, ymin, xmax, ymax) = bins.toArray
    Seq(
      (xmin + 0.5) * sizePerBinW,
      (ymin + 0.5) * sizePerBinH,
      (xmax + 0.5) * sizePerBinW,
      (ymax + 0.5) * sizePerBinH)
  }

  def dequantizeCoordinates(bins: Seq[Int], imageSize: (Int, Int)): Seq[Double] = {
    // bins: x1, y1, x2, y2, ...
    val (w, h) = imageSize
    val binsW = 1000.0
    val binsH = 1000.0
    val sizePerBinW = w / binsW
    val sizePerBinH = h / binsH
    bins
      .grouped(2)
      .flatMap {
        case Seq(x, y) =>
          Seq((x + 0.5) * sizePerBinW, (y + 0.5) * sizePerBinH)
        case _ => Seq.empty[Double]
      }
      .toSeq
  }

  val colorMap: Array[Color] = Array(
    Color.BLUE,
    Color.ORANGE,
    Color.GREEN,
    Color.MAGENTA,
    Color.PINK,
    Color.GRAY,
    Color.YELLOW,
    Color.CYAN,
    Color.RED,
    Color.LIGHT_GRAY,
    new Color(128, 0, 128), // purple
    new Color(255, 192, 203), // pink
    new Color(128, 128, 0), // olive
    new Color(0, 255, 255), // aqua
    new Color(255, 0, 255), // magenta
    new Color(255, 127, 80), // coral
    new Color(255, 215, 0), // gold
    new Color(210, 180, 140), // tan
    new Color(135, 206, 235) // skyblue
  )

  def plotBBox(
      image: BufferedImage,
      bboxes: Seq[Seq[Double]],
      labels: Seq[String]): BufferedImage = {
    val out = new BufferedImage(image.getWidth, image.getHeight, BufferedImage.TYPE_INT_ARGB)
    val g = out.createGraphics()
    g.drawImage(image, 0, 0, null)
    val maxFontSize = 18
    val minFontSize = 8
    for (((bbox, label), idx) <- bboxes.zip(labels).zipWithIndex) {
      val color = colorMap(idx % colorMap.length)
      g.setColor(color)
      val x1 = bbox(0).toInt
      val y1 = bbox(1).toInt
      val x2 = bbox(2).toInt
      val y2 = bbox(3).toInt
      g.setStroke(new BasicStroke(2))
      g.drawRect(x1, y1, x2 - x1, y2 - y1)
      // Dynamically adjust font size
      var fontSize = maxFontSize
      var labelWidth = 0
      var labelHeight = 0
      var font: Font = null
      var metrics: java.awt.FontMetrics = null
      do {
        font = new Font("Arial", Font.BOLD, fontSize)
        g.setFont(font)
        metrics = g.getFontMetrics(font)
        labelWidth = metrics.stringWidth(label) + 8
        labelHeight = metrics.getHeight
        fontSize -= 1
      } while (labelWidth > (x2 - x1).max(out.getWidth - 2 * 4) && fontSize >= minFontSize)
      // Clamp label position
      var labelX = x1
      var labelY = y1 - labelHeight + metrics.getAscent
      if (labelY < 0) {
        // If label would go above image, draw below the box
        labelY = y1 + labelHeight
      }
      if (labelX + labelWidth > out.getWidth) {
        labelX = out.getWidth - labelWidth - 1
      }
      if (labelX < 0) labelX = 0
      // Draw background
      g.setColor(new Color(color.getRed, color.getGreen, color.getBlue, 180))
      g.fillRect(labelX, labelY - metrics.getAscent, labelWidth, labelHeight)
      // Draw text
      g.setColor(Color.WHITE)
      g.drawString(label, labelX + 4, labelY)
    }
    g.dispose()
    out
  }

  def drawPolygons(
      image: BufferedImage,
      polygons: Seq[Seq[Seq[Double]]],
      labels: Seq[String],
      fillMask: Boolean = false): BufferedImage = {
    val out = new BufferedImage(image.getWidth, image.getHeight, BufferedImage.TYPE_INT_ARGB)
    val g = out.createGraphics()
    g.drawImage(image, 0, 0, null)
    val maxFontSize = 18
    val minFontSize = 8
    for (((polyList, label), idx) <- polygons.zip(labels).zipWithIndex) {
      val color = colorMap(idx % colorMap.length)
      g.setColor(color)
      for (polygon <- polyList) {
        val n = polygon.length / 2
        val xPoints = (0 until n).map(i => polygon(2 * i).toInt).toArray
        val yPoints = (0 until n).map(i => polygon(2 * i + 1).toInt).toArray
        if (fillMask) {
          g.setColor(new Color(color.getRed, color.getGreen, color.getBlue, 80))
          g.fillPolygon(xPoints, yPoints, n)
          g.setColor(color)
        }
        g.setStroke(new BasicStroke(2))
        g.drawPolygon(xPoints, yPoints, n)
        // Dynamic font size and clamped label
        var fontSize = maxFontSize
        var labelWidth = 0
        var labelHeight = 0
        var font: Font = null
        var metrics: java.awt.FontMetrics = null
        do {
          font = new Font("Arial", Font.BOLD, fontSize)
          g.setFont(font)
          metrics = g.getFontMetrics(font)
          labelWidth = metrics.stringWidth(label) + 8
          labelHeight = metrics.getHeight
          fontSize -= 1
        } while (labelWidth > out.getWidth - 2 * 4 && fontSize >= minFontSize)
        // Use first polygon point for label anchor
        var labelX = xPoints(0)
        var labelY = yPoints(0) - labelHeight + metrics.getAscent
        if (labelY < 0) {
          labelY = yPoints(0) + labelHeight
        }
        if (labelX + labelWidth > out.getWidth) {
          labelX = out.getWidth - labelWidth - 1
        }
        if (labelX < 0) labelX = 0
        g.setColor(new Color(color.getRed, color.getGreen, color.getBlue, 180))
        g.fillRect(labelX, labelY - metrics.getAscent, labelWidth, labelHeight)
        g.setColor(Color.WHITE)
        g.drawString(label, labelX + 4, labelY)
        g.setColor(color)
      }
    }
    g.dispose()
    out
  }

  def drawOcrBBoxes(
      image: BufferedImage,
      quadBoxes: Seq[Seq[Double]],
      labels: Seq[String]): BufferedImage = {
    val out = new BufferedImage(image.getWidth, image.getHeight, BufferedImage.TYPE_INT_ARGB)
    val g = out.createGraphics()
    g.drawImage(image, 0, 0, null)
    val maxFontSize = 18
    val minFontSize = 8
    for (((box, label), idx) <- quadBoxes.zip(labels).zipWithIndex) {
      val color = colorMap(idx % colorMap.length)
      g.setColor(color)
      val n = box.length / 2
      val xPoints = (0 until n).map(i => box(2 * i).toInt).toArray
      val yPoints = (0 until n).map(i => box(2 * i + 1).toInt).toArray
      g.setStroke(new BasicStroke(3))
      g.drawPolygon(xPoints, yPoints, n)
      // Dynamic font size and clamped label
      var fontSize = maxFontSize
      var labelWidth = 0
      var labelHeight = 0
      var font: Font = null
      var metrics: java.awt.FontMetrics = null
      do {
        font = new Font("Arial", Font.BOLD, fontSize)
        g.setFont(font)
        metrics = g.getFontMetrics(font)
        labelWidth = metrics.stringWidth(label) + 8
        labelHeight = metrics.getHeight
        fontSize -= 1
      } while (labelWidth > out.getWidth - 2 * 4 && fontSize >= minFontSize)
      var labelX = xPoints(0)
      var labelY = yPoints(0) - labelHeight + metrics.getAscent
      if (labelY < 0) {
        labelY = yPoints(0) + labelHeight
      }
      if (labelX + labelWidth > out.getWidth) {
        labelX = out.getWidth - labelWidth - 1
      }
      if (labelX < 0) labelX = 0
      g.setColor(new Color(color.getRed, color.getGreen, color.getBlue, 180))
      g.fillRect(labelX, labelY - metrics.getAscent, labelWidth, labelHeight)
      g.setColor(Color.WHITE)
      g.drawString(label, labelX + 4, labelY)
      g.setColor(color)
    }
    g.dispose()
    out
  }

  def convertToODFormat(mixed: MixedResult): Map[String, Any] = {
    Map("bboxes" -> mixed.bboxes.map(_.bbox), "labels" -> mixed.bboxesLabels)
  }

  def bufferedImageToBase64PNG(image: BufferedImage): String = {
    val baos = new ByteArrayOutputStream()
    ImageIO.write(image, "png", baos)
    Base64.getEncoder.encodeToString(baos.toByteArray)
  }

  /** Post-processes an image according to the task and model output, returning a base64-encoded
    * PNG.
    * @param image
    *   The input BufferedImage
    * @param task
    *   The task string (e.g. "<OD>", "<CAPTION_TO_PHRASE_GROUNDING>", etc.)
    * @param result
    *   The post-processed Florence2Result
    * @param textInput
    *   Optional text input (for phrase grounding, etc.)
    * @return
    *   Option[String] with base64 PNG, or None if no visualization is needed
    */
  def postProcessImage(
      image: BufferedImage,
      task: String,
      result: Florence2Result,
      textInput: Option[String] = None): Option[String] = {
    task match {
      case "<OD>" | "<DENSE_REGION_CAPTION>" | "<REGION_PROPOSAL>" =>
        result match {
          case BBoxesResult(bboxes) if bboxes.nonEmpty =>
            val img = plotBBox(image, bboxes.map(_.bbox), bboxes.map(_.catName))
            Some(bufferedImageToBase64PNG(img))
          case _ => None
        }
      case "<CAPTION_TO_PHRASE_GROUNDING>" =>
        result match {
          case PhraseGroundingResult(instances) if instances.nonEmpty =>
            val allBBoxes = instances.flatMap(_.bbox)
            val allLabels = instances.flatMap(inst => List.fill(inst.bbox.size)(inst.catName))
            val img = plotBBox(image, allBBoxes, allLabels)
            Some(bufferedImageToBase64PNG(img))
          case _ => None
        }
      case "<REFERRING_EXPRESSION_SEGMENTATION>" | "<REGION_TO_SEGMENTATION>" =>
        result match {
          case PolygonsResult(instances) if instances.nonEmpty =>
            val img = drawPolygons(
              image,
              instances.map(_.polygons),
              instances.map(_.catName),
              fillMask = true)
            Some(bufferedImageToBase64PNG(img))
          case _ => None
        }
      case "<OPEN_VOCABULARY_DETECTION>" =>
        result match {
          case MixedResult(bboxes, bboxesLabels, _, _) if bboxes.nonEmpty =>
            val img = plotBBox(image, bboxes.map(_.bbox), bboxesLabels)
            Some(bufferedImageToBase64PNG(img))
          case _ => None
        }
      case "<OCR_WITH_REGION>" =>
        result match {
          case OCRResult(instances) if instances.nonEmpty =>
            val img = drawOcrBBoxes(image, instances.map(_.quadBox), instances.map(_.text))
            Some(bufferedImageToBase64PNG(img))
          case _ => None
        }
      case _ => None
    }
  }
}
