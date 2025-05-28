package com.johnsnowlabs.nlp.annotators.cv.util.transform

import java.awt.image.BufferedImage
import java.awt.Color
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import com.johnsnowlabs.tags.{FastTest, SlowTest}

class SmolVLMUtilsTest extends AnyFlatSpec with Matchers {

  "SmolVLMUtils" should "resize image correctly" taggedAs FastTest in {
    // Create a test image
    val originalImage = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB)
    val g = originalImage.createGraphics()
    g.setColor(Color.RED)
    g.fillRect(0, 0, 100, 100)
    g.dispose()

    // Resize to 50x50
    val resizedImage = SmolVLMUtils.resizeImage(originalImage, 50, 50)

    // Check dimensions
    resizedImage.getWidth shouldBe 50
    resizedImage.getHeight shouldBe 50
  }

  it should "crop image correctly" taggedAs FastTest in {
    // Create a test image
    val originalImage = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB)
    val g = originalImage.createGraphics()
    g.setColor(Color.RED)
    g.fillRect(0, 0, 100, 100)
    g.dispose()

    // Crop a 20x20 section from (10,10) to (30,30)
    val croppedImage = SmolVLMUtils._crop(originalImage, 10, 10, 30, 30)

    // Check dimensions
    croppedImage.getWidth shouldBe 20
    croppedImage.getHeight shouldBe 20
  }

  it should "split large image into smaller tiles" taggedAs FastTest in {
    // Create a test image larger than max size
    val imageURL =
      "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"

    // download using java.net.URL
    val originalImage = javax.imageio.ImageIO.read(new java.net.URL(imageURL))

//    val originalImage = new BufferedImage(800, 800, BufferedImage.TYPE_INT_RGB)
//    val g = originalImage.createGraphics()
//    g.setColor(Color.RED)
//    g.fillRect(0, 0, 800, 800)
//    g.dispose()
    val resizedImage = SmolVLMUtils.resizeWithLongestEdge(originalImage, 1536)
    val resizedForEncoder = SmolVLMUtils.resizeForVisionEncoder(resizedImage, 384)
    val result = SmolVLMUtils.splitImage(resizedForEncoder, 384)

    // Check number of splits
    result.numSplitsH shouldBe 3
    result.numSplitsW shouldBe 4
    result.frames.size shouldBe 13 // 12 tiles + 1 global image

    // save the global image to check the content
    val globalImage = result.frames.last
    val outputFile = new java.io.File("global_image.png")
    javax.imageio.ImageIO.write(globalImage, "png", outputFile)

    // save the crops to check the content
    for (i <- 0 until result.frames.size - 1) {
      val tile = result.frames(i)
      val outputFile = new java.io.File(s"tile_$i.png")
      javax.imageio.ImageIO.write(tile, "png", outputFile)
    }

    // Check dimensions of first tile
    result.frames.head.getWidth shouldBe 384
    result.frames.head.getHeight shouldBe 384
    // Check dimensions of global image
    result.frames.last.getWidth shouldBe 384
    result.frames.last.getHeight shouldBe 384

  }

  it should "not split small image" taggedAs FastTest in {
    // Create a test image smaller than max size
    val originalImage = new BufferedImage(200, 200, BufferedImage.TYPE_INT_RGB)
    val g = originalImage.createGraphics()
    g.setColor(Color.RED)
    g.fillRect(0, 0, 200, 200)
    g.dispose()

    // Try to split with max size of 400
    val result = SmolVLMUtils.splitImage(originalImage, 400)

    // Check no splits were made
    result.numSplitsH shouldBe 0
    result.numSplitsW shouldBe 0
    result.frames.size shouldBe 1 // Only original image
    result.frames.head shouldBe originalImage
  }

  it should "calculate correct resize output size" taggedAs FastTest in {
    // Create a test image
    val image = new BufferedImage(1000, 500, BufferedImage.TYPE_INT_RGB)

    // Test with resolution max side of 500
    val size = SmolVLMUtils.getResizeOutputImageSize(image, 500)

    // Check dimensions maintain aspect ratio
    size.width shouldBe 500
    size.height shouldBe 250
  }

  it should "resize image with longest edge correctly" taggedAs FastTest in {
    // Create a test image
    val originalImage = new BufferedImage(1000, 500, BufferedImage.TYPE_INT_RGB)
    val g = originalImage.createGraphics()
    g.setColor(Color.RED)
    g.fillRect(0, 0, 1000, 500)
    g.dispose()

    // Resize with longest edge of 500
    val resizedImage = SmolVLMUtils.resizeWithLongestEdge(originalImage, 500)

    // Check dimensions maintain aspect ratio
    resizedImage.getWidth shouldBe 500
    resizedImage.getHeight shouldBe 250
  }

  it should "resize image for vision encoder correctly" taggedAs FastTest in {
    // Create a test image
    val originalImage = new BufferedImage(1000, 500, BufferedImage.TYPE_INT_RGB)
    val g = originalImage.createGraphics()
    g.setColor(Color.RED)
    g.fillRect(0, 0, 1000, 500)
    g.dispose()

    // Resize for vision encoder with max size of 384
    val resizedImage = SmolVLMUtils.resizeForVisionEncoder(originalImage, 384)

    // Check dimensions are multiples of max size
    resizedImage.getWidth shouldBe 1152 // 384 * 4
    resizedImage.getHeight shouldBe 768 // 384 * 2
  }

  it should "create correct pixel mask" taggedAs FastTest in {
    // Create a test image
    val image = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB)
    val outputSize = SmolVLMUtils.ImageSize(100, 100)

    // Create pixel mask
    val mask = SmolVLMUtils.makePixelMask(image, outputSize)

    // Check mask dimensions and values
    mask.length shouldBe 100
    mask(0).length shouldBe 100
    mask.flatten.sum shouldBe 10000 // All pixels should be valid
  }

  it should "handle non-square image splitting correctly" taggedAs FastTest in {
    // Create a rectangular test image
    val originalImage = new BufferedImage(800, 400, BufferedImage.TYPE_INT_RGB)
    val g = originalImage.createGraphics()
    g.setColor(Color.RED)
    g.fillRect(0, 0, 800, 400)
    g.dispose()

    // Split with max size of 400
    val result = SmolVLMUtils.splitImage(originalImage, 400)

    // Check number of splits
    result.numSplitsH shouldBe 1
    result.numSplitsW shouldBe 2
    result.frames.size shouldBe 3 // 2 tiles + 1 global image

    // Check dimensions of first tile
    result.frames.head.getWidth shouldBe 400
    result.frames.head.getHeight shouldBe 400
  }

  it should "handle edge case of very small image" taggedAs FastTest in {
    // Create a very small test image
    val originalImage = new BufferedImage(10, 10, BufferedImage.TYPE_INT_RGB)
    val g = originalImage.createGraphics()
    g.setColor(Color.RED)
    g.fillRect(0, 0, 10, 10)
    g.dispose()

    // Try to split with max size of 400
    val result = SmolVLMUtils.splitImage(originalImage, 400)

    // Check no splits were made
    result.numSplitsH shouldBe 0
    result.numSplitsW shouldBe 0
    result.frames.size shouldBe 1
    result.frames.head shouldBe originalImage
  }

  it should "handle edge case of very large image" taggedAs FastTest in {
    // Create a very large test image
    val originalImage = new BufferedImage(2000, 2000, BufferedImage.TYPE_INT_RGB)
    val g = originalImage.createGraphics()
    g.setColor(Color.RED)
    g.fillRect(0, 0, 2000, 2000)
    g.dispose()

    // Split with max size of 400
    val result = SmolVLMUtils.splitImage(originalImage, 400)

    // Check number of splits
    result.numSplitsH shouldBe 5
    result.numSplitsW shouldBe 5
    result.frames.size shouldBe 26 // 25 tiles + 1 global image

    // Check dimensions of first tile
    result.frames.head.getWidth shouldBe 400
    result.frames.head.getHeight shouldBe 400
  }

  it should "create correct pixel mask for partial image" taggedAs FastTest in {
    // Create a test image
    val image = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB)
    val outputSize = SmolVLMUtils.ImageSize(200, 200) // Larger than input image

    // Create pixel mask
    val mask = SmolVLMUtils.makePixelMask(image, outputSize)

    // Check mask dimensions and values
    mask.length shouldBe 200
    mask(0).length shouldBe 200
    mask.flatten.sum shouldBe 10000 // Only the original image area should be valid
  }

  it should "handle resize with minimum length constraint" taggedAs FastTest in {
    // Create a test image
    val image = new BufferedImage(100, 50, BufferedImage.TYPE_INT_RGB)

    // Test with resolution max side of 500 and min length of 100
    val size = SmolVLMUtils.getResizeOutputImageSize(image, 500)

    // Check dimensions maintain aspect ratio and respect minimum length
    size.width shouldBe 500
    size.height shouldBe 250
  }

  it should "handle resize with maximum length constraint" taggedAs FastTest in {
    // Create a test image
    val image = new BufferedImage(2000, 1000, BufferedImage.TYPE_INT_RGB)

    // Test with resolution max side of 500
    val size = SmolVLMUtils.getResizeOutputImageSize(image, 500)

    // Check dimensions are scaled down appropriately
    size.width shouldBe 500
    size.height shouldBe 250
  }

  it should "pad single image correctly" taggedAs FastTest in {
    // Create a test image array (2x2x3)
    val image = Array(
      Array(Array(1.0f, 2.0f, 3.0f), Array(4.0f, 5.0f, 6.0f)),
      Array(Array(7.0f, 8.0f, 9.0f), Array(10.0f, 11.0f, 12.0f)))
    val images = Seq(Seq(image))

    // Pad to 3x3x3
    val result = SmolVLMUtils.pad(images, constantValue = 0.0f, returnPixelMask = true)

    // Check padded dimensions
    result.paddedImages.head.head.length shouldBe 3 // height
    result.paddedImages.head.head(0).length shouldBe 3 // width
    result.paddedImages.head.head(0)(0).length shouldBe 3 // channels

    // Check original values preserved
    result.paddedImages.head.head(0)(0) shouldBe Array(1.0f, 2.0f, 3.0f)
    result.paddedImages.head.head(0)(1) shouldBe Array(4.0f, 5.0f, 6.0f)
    result.paddedImages.head.head(1)(0) shouldBe Array(7.0f, 8.0f, 9.0f)
    result.paddedImages.head.head(1)(1) shouldBe Array(10.0f, 11.0f, 12.0f)

    // Check padding values
    result.paddedImages.head.head(0)(2) shouldBe Array(0.0f, 0.0f, 0.0f)
    result.paddedImages.head.head(1)(2) shouldBe Array(0.0f, 0.0f, 0.0f)
    result.paddedImages.head.head(2)(0) shouldBe Array(0.0f, 0.0f, 0.0f)
    result.paddedImages.head.head(2)(1) shouldBe Array(0.0f, 0.0f, 0.0f)
    result.paddedImages.head.head(2)(2) shouldBe Array(0.0f, 0.0f, 0.0f)

    // Check pixel mask
    result.pixelMasks.isDefined shouldBe true
    result.pixelMasks.get.head.head shouldBe Array(Array(1, 1, 0), Array(1, 1, 0), Array(0, 0, 0))
  }

  it should "pad multiple images in batch correctly" taggedAs FastTest in {
    // Create test image arrays with proper dimensions (2x2x3)
    val image1 = Array(
      Array(Array(1.0f, 2.0f, 3.0f), Array(4.0f, 5.0f, 6.0f)),
      Array(Array(7.0f, 8.0f, 9.0f), Array(10.0f, 11.0f, 12.0f)))
    val image2 = Array(
      Array(Array(13.0f, 14.0f, 15.0f), Array(16.0f, 17.0f, 18.0f)),
      Array(Array(19.0f, 20.0f, 21.0f), Array(22.0f, 23.0f, 24.0f)))
    val images = Seq(Seq(image1, image2))

    // Pad to 3x3x3
    val result = SmolVLMUtils.pad(images, constantValue = 0.0f, returnPixelMask = true)

    // Check batch size
    result.paddedImages.length shouldBe 1
    result.paddedImages.head.length shouldBe 2

    // Check first image
    result.paddedImages.head(0)(0)(0) shouldBe Array(1.0f, 2.0f, 3.0f)
    result.paddedImages.head(0)(0)(1) shouldBe Array(4.0f, 5.0f, 6.0f)
    result.paddedImages.head(0)(1)(0) shouldBe Array(7.0f, 8.0f, 9.0f)
    result.paddedImages.head(0)(1)(1) shouldBe Array(10.0f, 11.0f, 12.0f)

    // Check second image
    result.paddedImages.head(1)(0)(0) shouldBe Array(13.0f, 14.0f, 15.0f)
    result.paddedImages.head(1)(0)(1) shouldBe Array(16.0f, 17.0f, 18.0f)
    result.paddedImages.head(1)(1)(0) shouldBe Array(19.0f, 20.0f, 21.0f)
    result.paddedImages.head(1)(1)(1) shouldBe Array(22.0f, 23.0f, 24.0f)

    // Check pixel masks
    result.pixelMasks.get.head(0) shouldBe Array(Array(1, 1, 0), Array(1, 1, 0), Array(0, 0, 0))
    result.pixelMasks.get.head(1) shouldBe Array(Array(1, 1, 0), Array(1, 1, 0), Array(0, 0, 0))
  }

  it should "pad with different constant value" taggedAs FastTest in {
    // Create a test image array with proper dimensions (2x2x3)
    val image = Array(
      Array(Array(1.0f, 2.0f, 3.0f), Array(4.0f, 5.0f, 6.0f)),
      Array(Array(7.0f, 8.0f, 9.0f), Array(10.0f, 11.0f, 12.0f)))
    val images = Seq(Seq(image))

    // Pad with constant value of 5.0
    val result = SmolVLMUtils.pad(images, constantValue = 5.0f, returnPixelMask = true)

    // Check padding values
    result.paddedImages.head.head(0)(2) shouldBe Array(5.0f, 5.0f, 5.0f)
    result.paddedImages.head.head(1)(2) shouldBe Array(5.0f, 5.0f, 5.0f)
    result.paddedImages.head.head(2)(0) shouldBe Array(5.0f, 5.0f, 5.0f)
    result.paddedImages.head.head(2)(1) shouldBe Array(5.0f, 5.0f, 5.0f)
    result.paddedImages.head.head(2)(2) shouldBe Array(5.0f, 5.0f, 5.0f)
  }

  it should "handle padding without pixel mask" taggedAs FastTest in {
    // Create a test image array with proper dimensions (2x2x3)
    val image = Array(
      Array(Array(1.0f, 2.0f, 3.0f), Array(4.0f, 5.0f, 6.0f)),
      Array(Array(7.0f, 8.0f, 9.0f), Array(10.0f, 11.0f, 12.0f)))
    val images = Seq(Seq(image))

    // Pad without pixel mask
    val result = SmolVLMUtils.pad(images, constantValue = 0.0f, returnPixelMask = false)

    // Check pixel mask is None
    result.pixelMasks shouldBe None

    // Check padding is still correct
    result.paddedImages.head.head.length shouldBe 3
    result.paddedImages.head.head(0).length shouldBe 3
    result.paddedImages.head.head(0)(0) shouldBe Array(1.0f, 2.0f, 3.0f)
  }

  it should "handle different sized images in batch" taggedAs FastTest in {
    // Create test image arrays of different sizes
    val image1 = Array(Array(Array(1.0f, 2.0f, 3.0f)))
    val image2 = Array(
      Array(Array(4.0f, 5.0f, 6.0f), Array(7.0f, 8.0f, 9.0f)),
      Array(Array(10.0f, 11.0f, 12.0f), Array(13.0f, 14.0f, 15.0f)))
    val images = Seq(Seq(image1, image2))

    // Pad to max dimensions
    val result = SmolVLMUtils.pad(images, constantValue = 0.0f, returnPixelMask = true)

    // Check dimensions are consistent
    result.paddedImages.head(0).length shouldBe 2 // height
    result.paddedImages.head(0)(0).length shouldBe 2 // width
    result.paddedImages.head(1).length shouldBe 2 // height
    result.paddedImages.head(1)(0).length shouldBe 2 // width

    // Check pixel masks
    result.pixelMasks.get.head(0) shouldBe Array(Array(1, 0), Array(0, 0))
    result.pixelMasks.get.head(1) shouldBe Array(Array(1, 1), Array(1, 1))
  }

  it should "handle empty batch" taggedAs FastTest in {
    // Create empty batch
    val images = Seq(Seq())

    // Pad empty batch
    val result = SmolVLMUtils.pad(images, constantValue = 0.0f, returnPixelMask = true)

    // Check result is empty
    result.paddedImages shouldBe Seq(Seq())
    result.pixelMasks.get shouldBe Seq(Seq())
  }
}
