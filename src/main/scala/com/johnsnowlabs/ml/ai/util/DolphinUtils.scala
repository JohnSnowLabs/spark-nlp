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

package com.johnsnowlabs.ml.ai.util

import com.johnsnowlabs.nlp.annotators.cv.util.transform.DonutImageUtils.ImageDims
import com.johnsnowlabs.reader.ElementType

import java.awt.image.BufferedImage
import java.util.regex.Pattern
import scala.collection.mutable
import scala.util.matching.Regex

/** Stage-1 layout parsing, box geometry and markdown assembly for Dolphin.
  *
  * Geometry ports the upstream demo code of each release (bytedance/Dolphin, branches `v1.0` and
  * `v1.5`); `scripts/dolphin/gen_parity.py` runs that code to produce the test references.
  */
private[johnsnowlabs] object DolphinUtils {

  /** Dolphin 1.0 and 1.5 ship identical configs and tokenizers and differ only in weights, so the
    * release is recognised from the stage-1 grammar it produces.
    */
  sealed trait Release

  object Release {

    /** `[x1,y1,x2,y2] label`, coordinates normalized to the square-padded page. */
    case object V1_0 extends Release

    /** `[x1,y1,x2,y2][label]`, coordinates in the 896x896 model input. */
    case object V1_5 extends Release
  }

  /** One region emitted by stage 1.
    *
    * @param bbox
    *   `[x1, y1, x2, y2]` normalized to 0..1 against the square-padded page
    * @param readingGroup
    *   index of the `[RELATION_SEP]`-delimited group, e.g. the column on a two-column page
    */
  case class LayoutElement(
      bbox: Array[Double],
      label: String,
      readingOrder: Int,
      readingGroup: Int)

  case class Layout(release: Release, elements: Seq[LayoutElement])

  private val V1_0Element: Regex =
    raw"\[(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+),\s*(\d*\.?\d+)\]\s*(\w+)".r
  private val V1_5Coordinates: Regex =
    raw"\[(\d*\.?\d+),(\d*\.?\d+),(\d*\.?\d+),(\d*\.?\d+)\]".r
  private val V1_5Label: Regex = raw"\]\[([^\]]+)\]".r
  private val V1_5Marker: Regex = raw"\[\d*\.?\d+,\d*\.?\d+,\d*\.?\d+,\d*\.?\d+\]\[".r

  private val PairSeparator = "[PAIR_SEP]"
  private val RelationSeparator = "[RELATION_SEP]"

  /** Side of the square model input that 1.5 coordinates are expressed in. */
  private val InputSide = 896.0

  def parseLayout(raw: String): Layout =
    if (V1_5Marker.findFirstIn(raw).isDefined) Layout(Release.V1_5, parseV1_5(raw))
    else Layout(Release.V1_0, parseV1_0(raw))

  /** `parse_layout_string` from `v1.0`: one regex over the whole string. */
  private def parseV1_0(raw: String): Seq[LayoutElement] = {
    val separators = mutable.ArrayBuffer.empty[Int]
    var from = raw.indexOf(RelationSeparator)
    while (from >= 0) {
      separators += from
      from = raw.indexOf(RelationSeparator, from + RelationSeparator.length)
    }

    V1_0Element
      .findAllMatchIn(raw)
      .zipWithIndex
      .map { case (m, order) =>
        LayoutElement(
          bbox = Array.tabulate(4)(i => m.group(i + 1).toDouble),
          label = m.group(5).trim,
          readingOrder = order,
          readingGroup = separators.count(_ < m.start))
      }
      .toSeq
  }

  /** `parse_layout_string` from `v1.5`: split on both separators, then take the first coordinates
    * and the first bracketed label of each segment.
    */
  private def parseV1_5(raw: String): Seq[LayoutElement] = {
    val elements = mutable.ArrayBuffer.empty[LayoutElement]
    var group = 0
    raw.split(Pattern.quote(PairSeparator), -1).foreach { pair =>
      pair.split(Pattern.quote(RelationSeparator), -1).zipWithIndex.foreach {
        case (segment, index) =>
          if (index > 0) group += 1
          for {
            coordinates <- V1_5Coordinates.findFirstMatchIn(segment)
            label <- V1_5Label.findFirstMatchIn(segment)
          } elements += LayoutElement(
            bbox = Array.tabulate(4)(i => coordinates.group(i + 1).toDouble / InputSide),
            label = label.group(1).trim,
            readingOrder = elements.length,
            readingGroup = group)
      }
    }
    elements
  }

  // ------------------------------------------------------------------ labels

  /** Labels are generated text, not a closed set, so anything unrecognised falls through to
    * `UncategorizedText`.
    */
  def elementTypeFor(label: String): String = label.toLowerCase match {
    case "title" | "sec" | "sub_sec" | "sub_sub_sec" => ElementType.TITLE
    case "sec_0" | "sec_1" | "sec_2" | "sec_3" | "sec_4" | "sec_5" => ElementType.TITLE
    case "para" | "half_para" | "text" => ElementType.NARRATIVE_TEXT
    case "list" => ElementType.LIST_ITEM
    case "tab" => ElementType.TABLE
    case "fig" => ElementType.IMAGE
    case "cap" | "reference" | "author" => ElementType.TEXT
    case "equ" | "formula" | "code" | "alg" => ElementType.TEXT
    case "header" | "head" => ElementType.HEADER
    case "foot" | "fnote" => ElementType.FOOTER
    case _ => ElementType.UNCATEGORIZED_TEXT
  }

  def isFigure(label: String): Boolean = label.equalsIgnoreCase("fig")

  def isTable(label: String): Boolean = label.equalsIgnoreCase("tab")

  // ------------------------------------------------------------------ coordinates

  /** A rectangle in pixels. */
  case class Box(x1: Int, y1: Int, x2: Int, y2: Int)

  /** A located element: its box on the square-padded page and the same box in original pixels. */
  case class MappedBox(padded: Box, original: Box)

  /** Upstream `process_coordinates` for `release`: scale, clamp, edge-adjust (1.0 only),
    * re-clamp, push below an overlapping predecessor, map back to original pixels.
    *
    * @param binary
    *   the page's Otsu mask, or `None` to skip edge adjustment
    */
  def locate(
      bbox: Array[Double],
      release: Release,
      dims: ImageDims,
      binary: Option[Array[Array[Boolean]]],
      previous: Option[MappedBox]): MappedBox = {
    val pw = dims.paddedWidth
    val ph = dims.paddedHeight

    def clamp(box: Box): Box = {
      val x1 = math.max(0, math.min(box.x1, pw - 1))
      val y1 = math.max(0, math.min(box.y1, ph - 1))
      var x2 = math.max(0, math.min(box.x2, pw))
      var y2 = math.max(0, math.min(box.y2, ph))
      if (x2 <= x1) x2 = math.min(x1 + 1, pw)
      if (y2 <= y1) y2 = math.min(y1 + 1, ph)
      Box(x1, y1, x2, y2)
    }

    val scaled = release match {
      case Release.V1_0 =>
        Box(
          (bbox(0) * pw).toInt,
          (bbox(1) * ph).toInt,
          (bbox(2) * pw).toInt,
          (bbox(3) * ph).toInt)
      case Release.V1_5 =>
        // Python's round() is half-to-even, like Math.rint
        def round(v: Double): Int = math.rint(v).toInt
        Box(
          round(bbox(0) * pw),
          round(bbox(1) * ph),
          round(bbox(2) * pw) + 1,
          round(bbox(3) * ph) + 1)
    }

    val initial = clamp(scaled)
    val adjusted = binary.fold(initial)(b => clamp(adjustBoxEdges(b, initial)))

    val settled = previous match {
      case Some(prev)
          if adjusted.x1 < prev.padded.x2 && adjusted.x2 > prev.padded.x1 &&
            adjusted.y1 < prev.padded.y2 && adjusted.y2 > prev.padded.y1 =>
        val y1 = math.min(prev.padded.y2, ph - 1)
        val y2 = if (adjusted.y2 <= y1) math.min(y1 + 1, ph) else adjusted.y2
        Box(adjusted.x1, y1, adjusted.x2, y2)
      case _ => adjusted
    }

    val x1 = math.max(0, settled.x1 - dims.left)
    val y1 = math.max(0, settled.y1 - dims.top)
    var x2 = math.min(dims.originalWidth, settled.x2 - dims.left)
    var y2 = math.min(dims.originalHeight, settled.y2 - dims.top)
    if (x2 <= x1) x2 = math.min(x1 + 1, dims.originalWidth)
    if (y2 <= y1) y2 = math.min(y1 + 1, dims.originalHeight)

    MappedBox(settled, Box(x1, y1, x2, y2))
  }

  /** Crop, or `None` when either edge is `minSize` pixels or fewer. */
  def crop(page: BufferedImage, box: Box, minSize: Int): Option[BufferedImage] = {
    val w = box.x2 - box.x1
    val h = box.y2 - box.y1
    if (w <= minSize || h <= minSize) None
    else if (box.x1 + w > page.getWidth || box.y1 + h > page.getHeight) None
    else Some(page.getSubimage(box.x1, box.y1, w, h))
  }

  // ------------------------------------------------------------------ markdown

  private def headingPrefix(label: String): Option[String] = label.toLowerCase match {
    case "title" | "sec_0" => Some("#")
    case "sec" | "sec_1" => Some("##")
    case "sub_sec" | "sec_2" | "sec_3" | "sec_4" | "sec_5" => Some("###")
    case "sub_sub_sec" => Some("####")
    case _ => None
  }

  private val MathDelimiters = Seq("$$", "$", "\\[", "\\(", "\\begin")

  private def wrapFormula(text: String): String =
    if (MathDelimiters.exists(text.trim.startsWith)) text else s"$$$$${text}$$$$"

  /** Assemble parsed elements into markdown, in reading order. Tables stay HTML: pipe tables
    * cannot express the `rowspan`/`colspan` Dolphin emits.
    */
  def toMarkdown(elements: Seq[(String, String)]): String = {
    val out = new mutable.StringBuilder()
    elements.foreach { case (label, text) =>
      val trimmed = text.trim
      if (trimmed.nonEmpty || isFigure(label)) {
        headingPrefix(label) match {
          case Some(hashes) =>
            out ++= s"$hashes ${trimmed.replace("\n", " ")}\n\n"
          case None =>
            label.toLowerCase match {
              case "list" => out ++= s"- $trimmed\n"
              case "tab" => out ++= s"$trimmed\n\n"
              case "formula" | "equ" | "alg" => out ++= s"${wrapFormula(trimmed)}\n\n"
              case "code" => out ++= s"```\n$trimmed\n```\n\n"
              case "fig" => out ++= "![Figure]()\n\n"
              case _ => out ++= s"$trimmed\n\n"
            }
        }
      }
    }
    out.toString.trim
  }

  // ------------------------------------------------------------------ box edges

  /** Upstream `adjust_box_edges` (1.0): walk each edge outwards, up to `maxPixels`, while the
    * line under it crosses ink.
    *
    * Behaviours kept from the reference because they decide the crop:
    *   - the score is `np.abs(np.diff(line))` over a uint8 0/255 line, so a 0->255 step scores
    *     255 and a 255->0 step wraps to 1;
    *   - a line shorter than two pixels scores NaN (0/0), which fails every comparison, so that
    *     edge walks the full `maxPixels` without becoming the best box;
    *   - `bestBox` starts as the unclamped input and only changes on a strict improvement, while
    *     the walk itself runs on a copy clamped to the last valid index.
    *
    * The reference recomputes Otsu at every step; the mask cannot change, so it is passed in.
    */
  def adjustBoxEdges(
      binary: Array[Array[Boolean]],
      box: Box,
      maxPixels: Int = 15,
      threshold: Double = 0.2): Box = {
    val height = binary.length
    if (height == 0) return box
    val width = binary(0).length

    def limit(v: Int, max: Int): Int = math.max(0, math.min(v, max))

    // edges: 0 = x1, 1 = y1, 2 = x2, 3 = y2
    def edgeScore(b: Box, edge: Int): Double = {
      val vertical = edge == 0 || edge == 2
      val at = edge match {
        case 0 => b.x1
        case 1 => b.y1
        case 2 => b.x2
        case _ => b.y2
      }
      val (from, to) =
        if (vertical) (b.y1, math.min(b.y2, height - 1)) else (b.x1, math.min(b.x2, width - 1))
      if (to - from + 1 < 2) return Double.NaN
      def value(i: Int): Int = if (if (vertical) binary(i)(at) else binary(at)(i)) 255 else 0
      var sum = 0L
      var i = from + 1
      while (i <= to) {
        sum += (value(i) - value(i - 1)) & 0xff
        i += 1
      }
      sum.toDouble / (to - from)
    }

    def step(b: Box, edge: Int, direction: Int): Box = edge match {
      case 0 => b.copy(x1 = limit(b.x1 + direction, width - 1))
      case 1 => b.copy(y1 = limit(b.y1 + direction, height - 1))
      case 2 => b.copy(x2 = limit(b.x2 + direction, width - 1))
      case _ => b.copy(y2 = limit(b.y2 + direction, height - 1))
    }

    var best = box
    var cur = Box(
      limit(box.x1, width - 1),
      limit(box.y1, height - 1),
      limit(box.x2, width - 1),
      limit(box.y2, height - 1))

    Seq((0, -1), (2, 1), (1, -1), (3, 1)).foreach { case (edge, direction) =>
      var bestScore = edgeScore(cur, edge)
      if (!(bestScore <= threshold)) {
        var steps = 0
        var done = false
        while (steps < maxPixels && !done) {
          cur = step(cur, edge, direction)
          val score = edgeScore(cur, edge)
          if (score < bestScore) {
            bestScore = score
            best = cur
          }
          done = score <= threshold
          steps += 1
        }
      }
    }
    best
  }

  /** OpenCV `BGR2GRAY` + Otsu + `THRESH_BINARY_INV`: true where the pixel is ink. */
  def binarizeOtsu(image: BufferedImage): Array[Array[Boolean]] = {
    val w = image.getWidth
    val h = image.getHeight
    val gray = Array.ofDim[Int](h, w)
    val histogram = new Array[Long](256)

    val pixels = image.getRGB(0, 0, w, h, null, 0, w)
    var y = 0
    while (y < h) {
      var x = 0
      while (x < w) {
        val p = pixels(y * w + x)
        val r = (p >> 16) & 0xff
        val g = (p >> 8) & 0xff
        val b = p & 0xff
        // cv2's fixed-point luma
        val v = (r * 4899 + g * 9617 + b * 1868 + 8192) >> 14
        gray(y)(x) = v
        histogram(v) += 1
        x += 1
      }
      y += 1
    }

    val total = (w.toLong * h).toDouble
    var sumAll = 0.0
    var i = 0
    while (i < 256) { sumAll += i.toDouble * histogram(i); i += 1 }

    var weightBackground = 0.0
    var sumBackground = 0.0
    var bestVariance = -1.0
    var threshold = 0
    i = 0
    while (i < 256) {
      weightBackground += histogram(i)
      if (weightBackground > 0) {
        val weightForeground = total - weightBackground
        if (weightForeground > 0) {
          sumBackground += i.toDouble * histogram(i)
          val meanBackground = sumBackground / weightBackground
          val meanForeground = (sumAll - sumBackground) / weightForeground
          val variance =
            weightBackground * weightForeground * math.pow(meanBackground - meanForeground, 2)
          if (variance > bestVariance) {
            bestVariance = variance
            threshold = i
          }
        } else sumBackground += i.toDouble * histogram(i)
      }
      i += 1
    }

    Array.tabulate(h, w)((row, col) => gray(row)(col) <= threshold)
  }
}
