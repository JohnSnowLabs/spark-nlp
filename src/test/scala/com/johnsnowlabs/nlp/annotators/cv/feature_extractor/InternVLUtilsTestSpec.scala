package com.johnsnowlabs.nlp.annotators.cv.feature_extractor

import com.johnsnowlabs.nlp.annotators.cv.util.transform.InternVLUtils
import java.awt.image.BufferedImage
import org.scalatest.flatspec.AnyFlatSpec
import com.johnsnowlabs.tags.{FastTest, SlowTest}

class InternVLUtilsTestSpec extends AnyFlatSpec {

  def createTestImage(width: Int, height: Int, color: Int = 0xffffff): BufferedImage = {
    val img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)
    val g2d = img.createGraphics()
    g2d.setColor(new java.awt.Color(color))
    g2d.fillRect(0, 0, width, height)
    g2d.dispose()
    img
  }

  "findClosestAspectRatio" should "find the closest ratio from a list" taggedAs SlowTest in {
    val aspect = 1.5
    val ratios = Seq((1, 1), (3, 2), (4, 3))
    val result = InternVLUtils.findClosestAspectRatio(aspect, ratios, 300, 200, 224)
    assert(result == (3, 2))
  }

  "resizeImage" should "resize a BufferedImage to the given dimensions" taggedAs SlowTest in {
    val img = createTestImage(100, 100)
    val resized = InternVLUtils.resizeImage(img, 50, 50)
    assert(resized.getWidth == 50)
    assert(resized.getHeight == 50)
  }

  "cropImage" should "crop a BufferedImage to the given rectangle" taggedAs SlowTest in {
    val img = createTestImage(100, 100, 0x123456)
    val cropped = InternVLUtils.cropImage(img, 10, 10, 30, 30)
    assert(cropped.getWidth == 30)
    assert(cropped.getHeight == 30)
  }

  "dynamicPreprocess" should "produce the correct number of crops and thumbnail if requested" taggedAs SlowTest in {
    val img = createTestImage(1600, 1067)
    val crops = InternVLUtils.dynamicPreprocess(
      img,
      minNum = 1,
      maxNum = 12,
      imageSize = 448,
      useThumbnail = false)
    assert(crops.length == 6)
    val cropsWithThumb = InternVLUtils.dynamicPreprocess(
      img,
      minNum = 1,
      maxNum = 12,
      imageSize = 448,
      useThumbnail = true)
    assert(cropsWithThumb.length == 7)
    assert(cropsWithThumb.last.getWidth == 448 && cropsWithThumb.last.getHeight == 448)
  }
}
