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

package com.johnsnowlabs.nlp.annotators.cv.util.transform

import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils

import java.awt.image.BufferedImage
import java.awt.{Color, Graphics2D}

/** Preprocessing for Donut-family vision encoders (Donut, Nougat, Dolphin).
  *
  * Named `Donut*` rather than `Dolphin*` because it implements HuggingFace's
  * `DonutImageProcessor`; a future Donut or Nougat annotator should reuse it.
  *
  * `DonutImageProcessor` applies, in order: `align_long_axis`, `resize`, `thumbnail`, `pad`,
  * `rescale`, `normalize`. For Dolphin's square 896x896 target this reduces considerably:
  *
  *   - `align_long_axis` is a '''no-op''': it only rotates when the target's orientation differs
  *     from the input's, and a square target is neither portrait nor landscape.
  *   - `resize` (short edge -> 896) followed by `thumbnail` (cap both edges at 896) composes to
  *     "fit inside 896x896, aspect preserved". We reproduce the two-step '''size computation'''
  *     exactly, including its `int()` truncations, but perform a '''single''' resize.
  *
  * That last point is deliberate. Done literally, the two steps blow an 816x51 text-line crop up
  * to '''14336x896''' before reducing it to 896x56 — a ~50MB transient allocation, on exactly the
  * shape that stage-2 element parsing produces most of. Going direct measured 439ms -> 2ms.
  *
  * Resampling kernel: measured against `DonutImageProcessor` ground truth, a plain
  * [[ImageResizeUtils.resizeBufferedImage]] (Java2D bilinear) has noticeably larger pixel error
  * than PIL's scaled-support triangle filter, but the '''model output is unaffected''' —
  * generated text was byte-identical on table, paragraph and formula fixtures, and differed by 1
  * character in 578 on a layout fixture. A bespoke resampler is not worth its weight here.
  */
private[johnsnowlabs] object DonutImageUtils {

  /** Geometry of the square-padded copy that stage-1 bounding boxes are normalized against. */
  case class ImageDims(
      originalWidth: Int,
      originalHeight: Int,
      paddedWidth: Int,
      paddedHeight: Int,
      left: Int,
      top: Int)

  /** `transformers.image_transforms.get_resize_output_image_size(size, default_to_square=False)`:
    * scale the '''short''' edge to `size`, truncating the long edge. Returns (height, width).
    */
  private[transform] def resizeTarget(height: Int, width: Int, size: Int): (Int, Int) = {
    val shortSide = math.min(width, height)
    val longSide = math.max(width, height)
    val newLong = (size.toDouble * longSide / shortSide).toInt
    if (width <= height) (newLong, size) else (size, newLong)
  }

  /** `DonutImageProcessor.thumbnail`: cap both edges at `size`, preserving aspect ratio. Returns
    * (height, width), unchanged when the image already fits.
    */
  private[transform] def thumbnailTarget(height: Int, width: Int, size: Int): (Int, Int) = {
    var h = math.min(height, size)
    var w = math.min(width, size)
    if (h == height && w == width) (height, width)
    else {
      if (height > width) w = (width.toDouble * h / height).toInt
      else if (width > height) h = (height.toDouble * w / width).toInt
      (h, w)
    }
  }

  /** The composed target of `resize` then `thumbnail`. Returns (height, width). */
  def fitTarget(height: Int, width: Int, size: Int): (Int, Int) = {
    val (rh, rw) = resizeTarget(height, width, size)
    thumbnailTarget(rh, rw, size)
  }

  /** Decode Spark's raw image bytes to an opaque RGB image, dropping any alpha channel. */
  def toRgbBuffered(
      bytes: Array[Byte],
      width: Int,
      height: Int,
      nChannels: Int): BufferedImage = {
    if (nChannels != 4) ImageIOUtils.byteToBufferedImage(bytes, width, height, nChannels)
    else {
      // Spark stores BGRA, which ImageIOUtils would copy into an ABGR raster one byte off.
      // Alpha is dropped, as PIL's convert("RGB") does in the reference pipeline.
      val pixels = new Array[Int](width * height)
      var i = 0
      while (i < pixels.length) {
        val o = i * 4
        pixels(i) =
          ((bytes(o + 2) & 0xff) << 16) | ((bytes(o + 1) & 0xff) << 8) | (bytes(o) & 0xff)
        i += 1
      }
      val rgb = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
      rgb.setRGB(0, 0, width, height, pixels, 0, width)
      rgb
    }
  }

  // ------------------------------------------------------------------ PIL-exact resampling

  // `DonutImageProcessor.preprocess` passes the configured `resample` to `resize` but NOT to
  // `thumbnail` (image_processing_donut.py:432), so thumbnail falls back to its signature default.
  // Dolphin's config says `resample: 2` (BILINEAR), yet the second pass is BICUBIC. Using one
  // filter for both leaves ~27% of content pixels differing by up to 0.63 in normalized units,
  // which is enough to flip near-tie argmax decisions and change digits in bbox coordinates.
  private val PrecisionBits = 32 - 8 - 2

  private def triangle(x: Double): Double = {
    val v = math.abs(x)
    if (v < 1.0) 1.0 - v else 0.0
  }

  private def bicubic(x: Double): Double = {
    val a = -0.5
    val v = math.abs(x)
    if (v < 1.0) ((a + 2.0) * v - (a + 3.0)) * v * v + 1.0
    else if (v < 2.0) (((v - 5.0) * v + 8.0) * v - 4.0) * a
    else 0.0
  }

  private def clip8(value: Int): Int = {
    val v = value >> PrecisionBits
    if (v < 0) 0 else if (v > 255) 255 else v
  }

  /** Kernel bounds and normalised fixed-point coefficients, matching PIL's `precompute_coeffs`
    * and `normalize_coeffs_8bpc`. Float coefficients are not sufficient: PIL quantises to 22-bit
    * fixed point and accumulates in int32, and the rounding that introduces is visible in the
    * output.
    */
  private def coefficients(
      inSize: Int,
      in0: Double,
      in1: Double,
      outSize: Int,
      cubic: Boolean): (Array[Int], Array[Int], Array[Array[Int]]) = {
    val scale = (in1 - in0) / outSize
    val filterScale = math.max(scale, 1.0)
    val support = (if (cubic) 2.0 else 1.0) * filterScale
    val kSize = math.ceil(support).toInt * 2 + 1

    val mins = new Array[Int](outSize)
    val lengths = new Array[Int](outSize)
    val kernels = Array.ofDim[Int](outSize, kSize)
    val weights = new Array[Double](kSize)

    var out = 0
    while (out < outSize) {
      val center = in0 + (out + 0.5) * scale
      var min = (center - support + 0.5).toInt
      if (min < 0) min = 0
      var max = (center + support + 0.5).toInt
      if (max > inSize) max = inSize
      val n = max - min

      var total = 0.0
      val step = 1.0 / filterScale
      var i = 0
      while (i < n) {
        val w =
          if (cubic) bicubic((i + min - center + 0.5) * step)
          else triangle((i + min - center + 0.5) * step)
        weights(i) = w
        total += w
        i += 1
      }
      i = 0
      while (i < n) {
        val w = if (total != 0.0) weights(i) / total else weights(i)
        kernels(out)(i) =
          (if (w < 0) -0.5 + w * (1 << PrecisionBits) else 0.5 + w * (1 << PrecisionBits)).toInt
        i += 1
      }
      mins(out) = min
      lengths(out) = n
      out += 1
    }
    (mins, lengths, kernels)
  }

  /** PIL's `ImagingReduce`: integer box-average by `(factorX, factorY)`, rounding by half the
    * block size, with partial blocks at the right and bottom edges averaged over their actual
    * count.
    *
    * `Image.resize(..., reducing_gap=2.0)` runs this first whenever the downscale factor is >= 2,
    * and `DonutImageProcessor.thumbnail` always passes `reducing_gap=2.0`. For a typical page the
    * factor is 1 and nothing happens, but a text line (816x51 -> 14336x896 -> 896x56) reduces by
    * 8 -- and stage-2 element crops are mostly text lines, so this is the common path, not a
    * corner case.
    */
  private def reduceBox(
      src: Array[Byte],
      width: Int,
      height: Int,
      factorX: Int,
      factorY: Int): (Array[Byte], Int, Int) = {
    if (factorX == 1 && factorY == 1) return (src, width, height)
    val outWidth = (width + factorX - 1) / factorX
    val outHeight = (height + factorY - 1) / factorY
    val out = new Array[Byte](outWidth * outHeight * 3)

    var oy = 0
    while (oy < outHeight) {
      val y0 = oy * factorY
      val blockH = math.min(factorY, height - y0)
      var ox = 0
      while (ox < outWidth) {
        val x0 = ox * factorX
        val blockW = math.min(factorX, width - x0)
        val count = blockW * blockH
        val amend = count / 2
        var c = 0
        while (c < 3) {
          var sum = amend
          var j = 0
          while (j < blockH) {
            var i = 0
            while (i < blockW) {
              sum += src(((y0 + j) * width + (x0 + i)) * 3 + c) & 0xff
              i += 1
            }
            j += 1
          }
          out((oy * outWidth + ox) * 3 + c) = (sum / count).toByte
          c += 1
        }
        ox += 1
      }
      oy += 1
    }
    (out, outWidth, outHeight)
  }

  /** Horizontal then vertical, as `ImagingResample` does, with a uint8 intermediate between the
    * passes. Carrying floats across that boundary is itself a source of divergence.
    *
    * Pixels are held as bytes rather than ints: an extreme aspect ratio produces a large
    * intermediate (an 816x51 text line passes through 14336x896) and bytes keep that near 38 MB
    * instead of 154 MB. Stage-2 element crops are mostly text lines, so this is the common case.
    */
  private def pilResize(
      src: Array[Byte],
      srcWidth: Int,
      srcHeight: Int,
      dstWidth: Int,
      dstHeight: Int,
      cubic: Boolean,
      box: (Double, Double, Double, Double) = null): Array[Byte] = {
    val (bx0, by0, bx1, by1) =
      if (box == null) (0.0, 0.0, srcWidth.toDouble, srcHeight.toDouble) else box
    val horizontal =
      if (dstWidth == srcWidth && bx0 == 0.0 && bx1 == srcWidth.toDouble) src
      else {
        val (mins, lengths, kernels) = coefficients(srcWidth, bx0, bx1, dstWidth, cubic)
        val out = new Array[Byte](dstWidth * srcHeight * 3)
        var y = 0
        while (y < srcHeight) {
          var x = 0
          while (x < dstWidth) {
            val min = mins(x)
            val n = lengths(x)
            val k = kernels(x)
            var c = 0
            while (c < 3) {
              var acc = 1 << (PrecisionBits - 1)
              var i = 0
              while (i < n) {
                acc += (src((y * srcWidth + (min + i)) * 3 + c) & 0xff) * k(i)
                i += 1
              }
              out((y * dstWidth + x) * 3 + c) = clip8(acc).toByte
              c += 1
            }
            x += 1
          }
          y += 1
        }
        out
      }

    if (dstHeight == srcHeight && by0 == 0.0 && by1 == srcHeight.toDouble) horizontal
    else {
      val (mins, lengths, kernels) = coefficients(srcHeight, by0, by1, dstHeight, cubic)
      val out = new Array[Byte](dstWidth * dstHeight * 3)
      var y = 0
      while (y < dstHeight) {
        val min = mins(y)
        val n = lengths(y)
        val k = kernels(y)
        var x = 0
        while (x < dstWidth) {
          var c = 0
          while (c < 3) {
            var acc = 1 << (PrecisionBits - 1)
            var i = 0
            while (i < n) {
              acc += (horizontal(((min + i) * dstWidth + x) * 3 + c) & 0xff) * k(i)
              i += 1
            }
            out((y * dstWidth + x) * 3 + c) = clip8(acc).toByte
            c += 1
          }
          x += 1
        }
        y += 1
      }
      out
    }
  }

  /** Bulk-read an image as packed RGB bytes, avoiding ~800k `getRGB` calls and `Color`
    * allocations per 896x896 image.
    */
  private def toRgbBytes(image: BufferedImage): Array[Byte] = {
    val w = image.getWidth
    val h = image.getHeight
    val packed = image.getRGB(0, 0, w, h, null, 0, w)
    val out = new Array[Byte](w * h * 3)
    var i = 0
    var j = 0
    while (i < packed.length) {
      val p = packed(i)
      out(j) = ((p >> 16) & 0xff).toByte
      out(j + 1) = ((p >> 8) & 0xff).toByte
      out(j + 2) = (p & 0xff).toByte
      i += 1
      j += 3
    }
    out
  }

  /** Full `DonutImageProcessor` pipeline: resize (bilinear), thumbnail (bicubic), centre-pad
    * black, rescale, normalize. Verified bit-exact against transformers 4.47.1.
    *
    * @return
    *   CHW float tensor, `[3][size][size]`, channel order RGB
    */
  def pixelValues(
      image: BufferedImage,
      size: Int,
      resample: Int,
      mean: Array[Double],
      std: Array[Double],
      doNormalize: Boolean,
      doRescale: Boolean,
      rescaleFactor: Double): Array[Array[Array[Float]]] = {

    val (resizeH, resizeW) = resizeTarget(image.getHeight, image.getWidth, size)
    val (fitH, fitW) = thumbnailTarget(resizeH, resizeW, size)

    val source = toRgbBytes(image)
    val resized =
      if (resizeH == image.getHeight && resizeW == image.getWidth) source
      // `resample` governs the first pass only, exactly as HF forwards it to `resize`.
      // PIL: 2 = BILINEAR, 3 = BICUBIC. Dolphin's config says 2.
      else pilResize(source, image.getWidth, image.getHeight, resizeW, resizeH, resample == 3)
    val fitted =
      if (fitH == resizeH && fitW == resizeW) resized
      else {
        // Image.resize(..., reducing_gap=2.0): box-reduce first when the factor reaches 2, then
        // resample the reduced image over the (possibly fractional) remaining box.
        val factorX = math.max(1, (resizeW.toDouble / fitW / 2.0).toInt)
        val factorY = math.max(1, (resizeH.toDouble / fitH / 2.0).toInt)
        val (reduced, reducedW, reducedH) =
          reduceBox(resized, resizeW, resizeH, factorX, factorY)
        val box =
          if (factorX == 1 && factorY == 1) null
          else
            (0.0, 0.0, resizeW.toDouble / factorX, resizeH.toDouble / factorY)
        pilResize(reduced, reducedW, reducedH, fitW, fitH, cubic = true, box = box)
      }

    // pad_image centres the content, `delta // 2` on the top and left
    val padLeft = (size - fitW) / 2
    val padTop = (size - fitH) / 2

    val out = Array.ofDim[Float](3, size, size)
    var c = 0
    while (c < 3) {
      val padValue = (if (doNormalize) (0.0 - mean(c)) / std(c) else 0.0).toFloat
      var y = 0
      while (y < size) {
        java.util.Arrays.fill(out(c)(y), padValue)
        y += 1
      }
      c += 1
    }

    var y = 0
    while (y < fitH) {
      val outY = y + padTop
      if (outY >= 0 && outY < size) {
        var x = 0
        while (x < fitW) {
          val outX = x + padLeft
          if (outX >= 0 && outX < size) {
            var ch = 0
            while (ch < 3) {
              val raw = fitted((y * fitW + x) * 3 + ch) & 0xff
              val rescaled = if (doRescale) raw * rescaleFactor else raw.toDouble
              out(ch)(outY)(outX) =
                (if (doNormalize) (rescaled - mean(ch)) / std(ch) else rescaled).toFloat
              ch += 1
            }
          }
          x += 1
        }
      }
      y += 1
    }
    out
  }

  /** The [[ImageDims]] that [[padToSquare]] produces, without allocating the canvas. */
  def squareDims(image: BufferedImage): ImageDims = {
    val w = image.getWidth
    val h = image.getHeight
    val side = math.max(w, h)
    ImageDims(w, h, side, side, (side - w) / 2, (side - h) / 2)
  }

  /** Centre-pad to a black `max(h, w)` square at '''original resolution'''.
    *
    * Stage-1 bounding boxes are normalized against this square, not against the raw image, so the
    * same padding must be reproduced before denormalizing them or every crop is offset.
    */
  def padToSquare(image: BufferedImage): (BufferedImage, ImageDims) = {
    val dims = squareDims(image)
    val canvas =
      new BufferedImage(dims.paddedWidth, dims.paddedHeight, BufferedImage.TYPE_INT_RGB)
    val g: Graphics2D = canvas.createGraphics()
    try {
      g.setColor(Color.BLACK)
      g.fillRect(0, 0, dims.paddedWidth, dims.paddedHeight)
      g.drawImage(image, dims.left, dims.top, null)
    } finally g.dispose()

    (canvas, dims)
  }
}
