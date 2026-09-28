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

package com.johnsnowlabs.nlp.annotators.cv.util

import com.johnsnowlabs.nlp.annotators.cv.util.transform.DonutImageUtils
import com.johnsnowlabs.tags.FastTest
import org.json4s.{DefaultFormats, JObject}
import org.json4s.jackson.JsonMethods.parse
import org.scalatest.flatspec.AnyFlatSpec

import java.awt.image.BufferedImage
import java.awt.{Color, Graphics2D}
import java.io.File
import javax.imageio.ImageIO
import scala.io.Source

class DonutImageUtilsTestSpec extends AnyFlatSpec {

  private val Size = 896
  private val Mean = Array(0.485, 0.456, 0.406)
  private val Std = Array(0.229, 0.224, 0.225)
  private val Resample = 2 // bilinear, per Dolphin's preprocessor_config.json
  private val fixtures = "src/test/resources/dolphin"

  // Black padding, after rescale + normalize, is exactly (0 - mean) / std per channel.
  private val PadR = ((0.0 - Mean(0)) / Std(0)).toFloat
  private val PadG = ((0.0 - Mean(1)) / Std(1)).toFloat
  private val PadB = ((0.0 - Mean(2)) / Std(2)).toFloat

  // Target sizes were cross-checked against transformers 4.47.1 DonutImageProcessor.
  "fitTarget" should "match DonutImageProcessor for portrait pages" taggedAs FastTest in {
    assert(DonutImageUtils.fitTarget(2096, 1466, Size) == ((896, 626)))
    assert(DonutImageUtils.fitTarget(2006, 1444, Size) == ((896, 645)))
    assert(DonutImageUtils.fitTarget(1599, 1165, Size) == ((896, 653)))
    assert(DonutImageUtils.fitTarget(500, 403, Size) == ((896, 722)))
  }

  it should "match DonutImageProcessor for landscape and extreme aspect ratios" taggedAs FastTest in {
    assert(DonutImageUtils.fitTarget(670, 1044, Size) == ((575, 896)))
    assert(DonutImageUtils.fitTarget(348, 1302, Size) == ((239, 896)))
    // a single text line -- the shape stage-2 element parsing mostly produces
    assert(DonutImageUtils.fitTarget(51, 816, Size) == ((56, 896)))
    assert(DonutImageUtils.fitTarget(162, 1862, Size) == ((77, 896)))
  }

  it should "leave an exactly square image untouched" taggedAs FastTest in {
    assert(DonutImageUtils.fitTarget(896, 896, Size) == ((896, 896)))
  }

  "pixelValues" should "return a CHW tensor with exact black padding" taggedAs FastTest in {
    val src = new BufferedImage(1044, 670, BufferedImage.TYPE_INT_RGB)
    val g = src.createGraphics()
    g.setColor(Color.WHITE); g.fillRect(0, 0, 1044, 670); g.dispose()

    val px = DonutImageUtils.pixelValues(src, Size, Resample, Mean, Std, true, true, 1.0 / 255.0)
    assert(px.length == 3)
    assert(px.head.length == Size)
    assert(px.head.head.length == Size)

    val padTop = (Size - 575) / 2
    // top padding row: exactly (0 - mean) / std
    assert(math.abs(px(0)(padTop / 2)(Size / 2) - PadR) < 1e-5)
    assert(math.abs(px(1)(padTop / 2)(Size / 2) - PadG) < 1e-5)
    assert(math.abs(px(2)(padTop / 2)(Size / 2) - PadB) < 1e-5)

    // interior is white: (1 - mean) / std
    assert(math.abs(px(0)(Size / 2)(Size / 2) - ((1.0 - Mean(0)) / Std(0)).toFloat) < 1e-3)
  }

  // Phase 1 asserted only loose distributional agreement, because Java2D bilinear could not do
  // better. With PIL's exact path (bilinear resize, bicubic thumbnail, fixed-point coefficients,
  // uint8 between passes) the tensors are bit-identical, so this asserts every sampled element.
  "pixelValues" should "match DonutImageProcessor exactly, element for element" taggedAs FastTest in {
    val fixture = new File("src/test/resources/image_preprocessor/dolphin_pixel_values.json")
    assume(fixture.exists(), s"missing ${fixture.getPath}")
    val source = Source.fromFile(fixture)
    val json =
      try parse(source.mkString)
      finally source.close()
    implicit val formats: DefaultFormats.type = DefaultFormats

    var checkedImages = 0
    json.asInstanceOf[JObject].obj.foreach { case (name, entry) =>
      val file = new File(s"$fixtures/$name")
      if (file.exists()) {
        checkedImages += 1
        val px =
          DonutImageUtils.pixelValues(
            ImageIO.read(file),
            Size,
            Resample,
            Mean,
            Std,
            doNormalize = true,
            doRescale = true,
            rescaleFactor = 1.0 / 255.0)

        val samples = (entry \ "samples").extract[List[List[Double]]]
        assert(samples.nonEmpty, s"no samples for $name")
        val mismatches = samples.flatMap { s =>
          val (c, y, x, expected) = (s.head.toInt, s(1).toInt, s(2).toInt, s(3))
          val actual = px(c)(y)(x)
          if (math.abs(actual - expected) <= 1e-6) None
          else Some(s"$name[$c][$y][$x] expected $expected but was $actual")
        }
        assert(
          mismatches.isEmpty,
          s"${mismatches.length}/${samples.length} sampled elements differ for $name:\n" +
            mismatches.take(5).mkString("\n"))

        val flat = px.flatten.flatten
        val mean = flat.map(_.toDouble).sum / flat.length
        assert(
          math.abs(mean - (entry \ "mean").extract[Double]) < 1e-5,
          s"$name tensor mean drifted")
      }
    }
    assert(checkedImages >= 2, s"only $checkedImages fixture images were available to check")
  }

  "padToSquare" should "centre the image in a black max(h, w) square" taggedAs FastTest in {
    val src = new BufferedImage(1466, 2096, BufferedImage.TYPE_INT_RGB)
    val g = src.createGraphics()
    g.setColor(Color.WHITE); g.fillRect(0, 0, 1466, 2096); g.dispose()

    val (padded, dims) = DonutImageUtils.padToSquare(src)
    assert(padded.getWidth == 2096 && padded.getHeight == 2096)
    assert(dims.originalWidth == 1466 && dims.originalHeight == 2096)
    assert(dims.paddedWidth == 2096 && dims.paddedHeight == 2096)
    assert(dims.left == (2096 - 1466) / 2 && dims.top == 0)
    assert(new Color(padded.getRGB(5, 1000)) == Color.BLACK)
    assert(new Color(padded.getRGB(dims.left + 10, 1000)) == Color.WHITE)
  }

  it should "be a no-op for an already-square image" taggedAs FastTest in {
    val src = new BufferedImage(500, 500, BufferedImage.TYPE_INT_RGB)
    val (padded, dims) = DonutImageUtils.padToSquare(src)
    assert(padded.getWidth == 500 && padded.getHeight == 500)
    assert(dims.left == 0 && dims.top == 0)
  }

  "toRgbBuffered" should "read Spark's BGRA bytes and drop alpha, as PIL's convert(\"RGB\") does" taggedAs FastTest in {
    // two pixels, row-wise BGRA: an opaque one and a fully transparent one
    val bytes = Array[Byte](10, 20, 30, -1, 40, 50, 60, 0)
    val rgb = DonutImageUtils.toRgbBuffered(bytes, 2, 1, 4)
    assert(rgb.getType == BufferedImage.TYPE_INT_RGB)
    assert(new Color(rgb.getRGB(0, 0)) == new Color(30, 20, 10))
    assert(new Color(rgb.getRGB(1, 0)) == new Color(60, 50, 40))
  }

  it should "pass through an opaque 3-channel image unchanged" taggedAs FastTest in {
    val w = 2
    val h = 2
    // row-wise BGR
    val bytes = Array[Byte](0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1) // pure red pixels
    val rgb = DonutImageUtils.toRgbBuffered(bytes, w, h, 3)
    assert(new Color(rgb.getRGB(0, 0)) == Color.RED)
  }

  "squareDims" should "agree with padToSquare without allocating the canvas" taggedAs FastTest in {
    Seq((1466, 2096), (2096, 1466), (500, 500), (1, 7)).foreach { case (w, h) =>
      val src = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
      val (_, padded) = DonutImageUtils.padToSquare(src)
      assert(DonutImageUtils.squareDims(src) == padded, s"disagreed for ${w}x$h")
    }
  }
}
