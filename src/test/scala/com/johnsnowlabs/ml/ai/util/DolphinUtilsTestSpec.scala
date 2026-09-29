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

import com.johnsnowlabs.ml.ai.util.DolphinUtils.{Box, MappedBox, Release}
import com.johnsnowlabs.nlp.annotators.cv.DolphinReference
import com.johnsnowlabs.nlp.annotators.cv.util.transform.DonutImageUtils
import com.johnsnowlabs.nlp.annotators.cv.util.transform.DonutImageUtils.ImageDims
import com.johnsnowlabs.reader.ElementType
import com.johnsnowlabs.tags.FastTest
import org.scalatest.flatspec.AnyFlatSpec

import java.awt.image.BufferedImage
import java.awt.{Color, Graphics2D}
import java.io.File
import javax.imageio.ImageIO

class DolphinUtilsTestSpec extends AnyFlatSpec {

  /** A white page with one black rectangle, inclusive of both corners. */
  private def inkPage(
      width: Int,
      height: Int,
      inkX: (Int, Int),
      inkY: (Int, Int)): BufferedImage = {
    val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val g: Graphics2D = image.createGraphics()
    try {
      g.setColor(Color.WHITE)
      g.fillRect(0, 0, width, height)
      g.setColor(Color.BLACK)
      g.fillRect(inkX._1, inkY._1, inkX._2 - inkX._1 + 1, inkY._2 - inkY._1 + 1)
    } finally g.dispose()
    image
  }

  // ---------------------------------------------------------------- parseLayout

  "parseLayout" should "read the space-separated 1.0 grammar" taggedAs FastTest in {
    val layout =
      DolphinUtils.parseLayout("[0.1,0.2,0.3,0.4] title [0.5,0.6,0.7,0.8] para")
    assert(layout.release == Release.V1_0)
    assert(layout.elements.map(_.label) == Seq("title", "para"))
    assert(layout.elements.map(_.readingOrder) == Seq(0, 1))
    assert(layout.elements.map(_.readingGroup) == Seq(0, 0))
    assert(layout.elements.head.bbox.toSeq == Seq(0.1, 0.2, 0.3, 0.4))
  }

  it should "read the 1.0 separator grammar and number the reading groups" taggedAs FastTest in {
    val raw = "[0.1, 0.2, 0.3, 0.4] title[PAIR_SEP][0.5,0.6,0.7,0.8] para" +
      "[RELATION_SEP][0.1,0.5,0.4,0.9] tab"
    val layout = DolphinUtils.parseLayout(raw)
    assert(layout.release == Release.V1_0)
    assert(layout.elements.map(_.label) == Seq("title", "para", "tab"))
    assert(layout.elements.map(_.readingGroup) == Seq(0, 0, 1))
  }

  it should "recognise the 1.5 grammar and normalize its 896px coordinates" taggedAs FastTest in {
    val raw = "[112,224,448,896][sec_0][PAIR_SEP][0,0,896,448][text]" +
      "[RELATION_SEP][10,20,30,40][tab][PAIR_SEP]"
    val layout = DolphinUtils.parseLayout(raw)
    assert(layout.release == Release.V1_5)
    assert(layout.elements.map(_.label) == Seq("sec_0", "text", "tab"))
    assert(layout.elements.map(_.readingOrder) == Seq(0, 1, 2))
    assert(layout.elements.map(_.readingGroup) == Seq(0, 0, 1))
    assert(layout.elements.head.bbox.toSeq == Seq(0.125, 0.25, 0.5, 1.0))
  }

  it should "accept integer coordinates" taggedAs FastTest in {
    val layout = DolphinUtils.parseLayout("[0,0,1,1] para")
    assert(layout.elements.map(_.bbox.toSeq) == Seq(Seq(0.0, 0.0, 1.0, 1.0)))
  }

  it should "return nothing rather than throw on text with no boxes" taggedAs FastTest in {
    assert(
      DolphinUtils.parseLayout("the model rambled instead of emitting boxes").elements.isEmpty)
    assert(DolphinUtils.parseLayout("").elements.isEmpty)
  }

  // ---------------------------------------------------------------- labels

  "elementTypeFor" should "map every label the reference dispatches on" taggedAs FastTest in {
    assert(DolphinUtils.elementTypeFor("title") == ElementType.TITLE)
    assert(DolphinUtils.elementTypeFor("sec") == ElementType.TITLE)
    assert(DolphinUtils.elementTypeFor("sub_sec") == ElementType.TITLE)
    assert(DolphinUtils.elementTypeFor("sub_sub_sec") == ElementType.TITLE)
    assert(DolphinUtils.elementTypeFor("para") == ElementType.NARRATIVE_TEXT)
    assert(DolphinUtils.elementTypeFor("list") == ElementType.LIST_ITEM)
    assert(DolphinUtils.elementTypeFor("tab") == ElementType.TABLE)
    assert(DolphinUtils.elementTypeFor("fig") == ElementType.IMAGE)
    assert(DolphinUtils.elementTypeFor("cap") == ElementType.TEXT)
    assert(DolphinUtils.elementTypeFor("author") == ElementType.TEXT)
    assert(DolphinUtils.elementTypeFor("header") == ElementType.HEADER)
    assert(DolphinUtils.elementTypeFor("fnote") == ElementType.FOOTER)
  }

  it should "map the 1.5 labels" taggedAs FastTest in {
    (0 to 5).foreach(level =>
      assert(DolphinUtils.elementTypeFor(s"sec_$level") == ElementType.TITLE))
    assert(DolphinUtils.elementTypeFor("text") == ElementType.NARRATIVE_TEXT)
    assert(DolphinUtils.elementTypeFor("half_para") == ElementType.NARRATIVE_TEXT)
    assert(DolphinUtils.elementTypeFor("equ") == ElementType.TEXT)
    assert(DolphinUtils.elementTypeFor("code") == ElementType.TEXT)
  }

  // The label is generated text, not a closed enum -- the fallback is load-bearing.
  it should "fall back to UncategorizedText for a label nobody has documented" taggedAs FastTest in {
    assert(DolphinUtils.elementTypeFor("marginalia") == ElementType.UNCATEGORIZED_TEXT)
    assert(DolphinUtils.elementTypeFor("") == ElementType.UNCATEGORIZED_TEXT)
  }

  it should "be case insensitive" taggedAs FastTest in {
    assert(DolphinUtils.elementTypeFor("TAB") == ElementType.TABLE)
    assert(DolphinUtils.isTable("TAB"))
    assert(DolphinUtils.isFigure("Fig"))
    assert(!DolphinUtils.isTable("table"))
  }

  // ---------------------------------------------------------------- markdown

  "toMarkdown" should "render each heading level" taggedAs FastTest in {
    val markdown = DolphinUtils.toMarkdown(
      Seq(("title", "T"), ("sec", "S"), ("sub_sec", "SS"), ("sub_sub_sec", "SSS")))
    assert(markdown == "# T\n\n## S\n\n### SS\n\n#### SSS")
    val v15 = DolphinUtils.toMarkdown(Seq(("sec_0", "A"), ("sec_1", "B"), ("sec_4", "C")))
    assert(v15 == "# A\n\n## B\n\n### C")
  }

  it should "fence code and wrap a 1.5 formula" taggedAs FastTest in {
    assert(DolphinUtils.toMarkdown(Seq(("code", "x = 1"))) == "```\nx = 1\n```")
    assert(DolphinUtils.toMarkdown(Seq(("equ", "x = 1"))) == "$$x = 1$$")
  }

  it should "flatten newlines inside a heading" taggedAs FastTest in {
    assert(DolphinUtils.toMarkdown(Seq(("title", "two\nlines"))) == "# two lines")
  }

  // Markdown pipe tables cannot express rowspan/colspan, so the HTML passes through.
  it should "pass table HTML through untouched" taggedAs FastTest in {
    val html = "<table><tr><td rowspan=\"2\">a</td></tr></table>"
    assert(DolphinUtils.toMarkdown(Seq(("tab", html))) == html)
  }

  it should "wrap a bare formula in display math but leave a delimited one alone" taggedAs FastTest in {
    assert(DolphinUtils.toMarkdown(Seq(("formula", "x = 1"))) == "$$x = 1$$")
    assert(
      DolphinUtils.toMarkdown(Seq(("formula", "\\begin{aligned}x\\end{aligned}"))) ==
        "\\begin{aligned}x\\end{aligned}")
  }

  it should "emit a figure placeholder even with no text, and skip empty text otherwise" taggedAs FastTest in {
    assert(DolphinUtils.toMarkdown(Seq(("fig", ""))) == "![Figure]()")
    assert(DolphinUtils.toMarkdown(Seq(("para", "   "))).isEmpty)
  }

  it should "prefix list items with a dash" taggedAs FastTest in {
    assert(DolphinUtils.toMarkdown(Seq(("list", "one"), ("list", "two"))) == "- one\n- two")
  }

  it should "produce an empty document for no elements" taggedAs FastTest in {
    assert(DolphinUtils.toMarkdown(Seq.empty).isEmpty)
  }

  // ---------------------------------------------------------------- otsu

  "binarizeOtsu" should "mark ink true and paper false" taggedAs FastTest in {
    val page = inkPage(20, 20, inkX = (5, 10), inkY = (5, 10))
    val binary = DolphinUtils.binarizeOtsu(page)
    assert(binary.length == 20 && binary(0).length == 20)
    assert(binary(7)(7), "the black rectangle must be ink")
    assert(!binary(1)(1), "the white margin must not be ink")
    assert(binary(5)(5) && binary(10)(10), "the rectangle is inclusive of its corners")
    assert(!binary(11)(11))
  }

  // ---------------------------------------------------------------- locate

  private val Square = ImageDims(100, 100, 100, 100, 0, 0)
  private val Page200 = ImageDims(200, 200, 200, 200, 0, 0)
  private val Quarter = Array(0.25, 0.25, 0.75, 0.75)

  private def locate10(
      bbox: Array[Double],
      dims: ImageDims,
      binary: Option[Array[Array[Boolean]]] = None,
      previous: Option[MappedBox] = None): MappedBox =
    DolphinUtils.locate(bbox, Release.V1_0, dims, binary, previous)

  "locate" should "denormalize against the padded page" taggedAs FastTest in {
    assert(locate10(Quarter, Square).padded == Box(25, 25, 75, 75))
  }

  it should "map padded pixels back through the pad offsets" taggedAs FastTest in {
    // a 200x160 page pads to 200x200, so 20px bars top and bottom
    val box = locate10(Quarter, ImageDims(200, 160, 200, 200, 0, 20))
    assert(box.padded == Box(50, 50, 150, 150))
    assert(box.original == Box(50, 30, 150, 130))
  }

  it should "keep a box that lies in the padding one original pixel wide" taggedAs FastTest in {
    // a 100x200 page pads to 200x200, so 50px bars left and right
    val box = locate10(Array(0.0, 0.25, 0.2, 0.75), ImageDims(100, 200, 200, 200, 50, 0))
    assert(box.padded == Box(0, 50, 40, 150))
    assert(box.original == Box(0, 50, 1, 150))
  }

  it should "clamp coordinates that fall outside 0..1" taggedAs FastTest in {
    assert(locate10(Array(-0.5, -0.5, 1.5, 1.5), Square).padded == Box(0, 0, 100, 100))
  }

  it should "widen a degenerate box to one pixel" taggedAs FastTest in {
    assert(locate10(Array(0.5, 0.5, 0.5, 0.5), Square).padded == Box(50, 50, 51, 51))
  }

  it should "push an overlapping box below its predecessor" taggedAs FastTest in {
    val previous = MappedBox(Box(0, 0, 60, 40), Box(0, 0, 60, 40))
    val box = locate10(Array(0.1, 0.1, 0.5, 0.9), Square, previous = Some(previous))
    assert(box.padded.y1 == 40, "y1 snaps to the predecessor's bottom")
    assert(box.padded.x1 == 10 && box.padded.x2 == 50, "x is untouched")
  }

  it should "round 1.5 coordinates half-to-even, as Python's round() does" taggedAs FastTest in {
    val layout = DolphinUtils.parseLayout("[448,448,448,448][text]")
    val box = DolphinUtils.locate(
      layout.elements.head.bbox,
      Release.V1_5,
      ImageDims(5, 5, 5, 5, 0, 0),
      None,
      None)
    // 448 / 896 * 5 = 2.5 rounds to 2, and x2/y2 gain a pixel
    assert(box.padded == Box(2, 2, 3, 3))
  }

  // Ink spans x 45..100, y 55..75. Column 50 crosses it, so the left edge walks left until
  // column 44 misses it.
  it should "pull a box edge off the ink" taggedAs FastTest in {
    val binary = DolphinUtils.binarizeOtsu(inkPage(200, 200, inkX = (45, 100), inkY = (55, 75)))
    assert(locate10(Quarter, Page200).padded == Box(50, 50, 150, 150))
    assert(locate10(Quarter, Page200, Some(binary)).padded == Box(44, 50, 150, 150))
  }

  it should "derive original pixels from the adjusted box" taggedAs FastTest in {
    val binary = DolphinUtils.binarizeOtsu(inkPage(200, 200, inkX = (45, 100), inkY = (55, 75)))
    assert(locate10(Quarter, Page200, Some(binary)).original == Box(44, 50, 150, 150))
  }

  it should "adjust box edges before de-overlapping" taggedAs FastTest in {
    val binary = DolphinUtils.binarizeOtsu(inkPage(200, 200, inkX = (45, 100), inkY = (55, 75)))
    // overlaps the adjusted box [44,50,150,150] but not the raw one [50,50,150,150]
    val previous = MappedBox(Box(0, 0, 48, 80), Box(0, 0, 48, 80))
    val box = locate10(Quarter, Page200, Some(binary), Some(previous))
    assert(box.padded.x1 == 44)
    assert(box.padded.y1 == 80, "de-overlap must see the adjusted box")
  }

  // Upstream check_edge scores the edge being moved (`edge = current_box[i]`); scoring x1 for
  // every vertical edge left right edges cutting through text.
  it should "walk the right edge off ink under that edge" taggedAs FastTest in {
    val binary = DolphinUtils.binarizeOtsu(inkPage(200, 200, inkX = (140, 160), inkY = (55, 75)))
    assert(locate10(Quarter, Page200, Some(binary)).padded == Box(50, 50, 161, 150))
  }

  // The references are ByteDance's own boxes for every stage-1 element of every fixture page.
  Seq("1.0" -> Release.V1_0, "1.5" -> Release.V1_5).foreach { case (version, release) =>
    it should s"place every $version fixture box where upstream does" taggedAs FastTest in {
      val pages = DolphinReference.load(version).cases.filter(_.mode == "layout")
      assert(pages.nonEmpty)
      pages.foreach { page =>
        val layout = DolphinUtils.parseLayout(page.output)
        assert(layout.release == release, page.image)
        val image = ImageIO.read(new File(s"${DolphinReference.Fixtures}/${page.image}"))
        val (padded, dims) = DonutImageUtils.padToSquare(image)
        val binary =
          if (layout.release == Release.V1_0) Some(DolphinUtils.binarizeOtsu(padded)) else None

        var previous: Option[MappedBox] = None
        val actual = layout.elements.map { element =>
          val box = DolphinUtils.locate(element.bbox, layout.release, dims, binary, previous)
          previous = Some(box)
          (element.label, box.padded, box.original)
        }
        val expected = page.boxes.map { b =>
          def box(v: Seq[Int]) = Box(v.head, v(1), v(2), v(3))
          (b.label, box(b.padded), box(b.original))
        }
        assert(actual == expected, page.image)
      }
    }
  }

  // ---------------------------------------------------------------- crop

  // The reference keeps a crop when both edges exceed 3 pixels (demo_page_hf.py:
  // `cropped.shape[0] > 3 and cropped.shape[1] > 3`), so minSize is 3, not 4.
  "crop" should "keep a 4-pixel edge and drop a 3-pixel one at minSize 3" taggedAs FastTest in {
    val page = new BufferedImage(50, 50, BufferedImage.TYPE_INT_RGB)
    assert(DolphinUtils.crop(page, DolphinUtils.Box(0, 0, 4, 4), 3).isDefined)
    assert(DolphinUtils.crop(page, DolphinUtils.Box(0, 0, 3, 4), 3).isEmpty)
    assert(DolphinUtils.crop(page, DolphinUtils.Box(0, 0, 4, 3), 3).isEmpty)
  }

  it should "return a subimage of exactly the box's size" taggedAs FastTest in {
    val page = new BufferedImage(50, 50, BufferedImage.TYPE_INT_RGB)
    val cropped = DolphinUtils.crop(page, DolphinUtils.Box(5, 6, 25, 36), 3)
    assert(cropped.exists(c => c.getWidth == 20 && c.getHeight == 30))
  }

  it should "refuse a box that runs past the page" taggedAs FastTest in {
    val page = new BufferedImage(50, 50, BufferedImage.TYPE_INT_RGB)
    assert(DolphinUtils.crop(page, DolphinUtils.Box(40, 0, 60, 20), 3).isEmpty)
  }
}
