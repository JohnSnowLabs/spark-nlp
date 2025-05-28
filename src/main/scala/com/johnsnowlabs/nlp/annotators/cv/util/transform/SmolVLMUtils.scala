package com.johnsnowlabs.nlp.annotators.cv.util.transform
import java.awt.image.BufferedImage
import java.awt.{Color, Graphics2D}
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.ArrayBuffer

import ImageResizeUtils.resizeBufferedImage

private[johnsnowlabs] object SmolVLMUtils {
  val MAX_IMAGE_SIZE = 4096

  def resizeImage(image: BufferedImage, width: Int, height: Int): BufferedImage = {
    val resizedImage = new BufferedImage(width, height, image.getType)
    val g = resizedImage.createGraphics()
    g.drawImage(image, 0, 0, width, height, null)
    g.dispose()
    resizedImage
  }

  def _crop(image: BufferedImage, x1: Int, y1: Int, x2: Int, y2: Int): BufferedImage = {
    val width = x2 - x1
    val height = y2 - y1
    val croppedImage = new BufferedImage(width, height, image.getType)
    val g = croppedImage.createGraphics()
    g.drawImage(image, 0, 0, width, height, x1, y1, x2, y2, null)
    g.dispose()
    croppedImage
  }

  case class SplitImageResult(frames: Seq[BufferedImage], numSplitsH: Int, numSplitsW: Int)

  def splitImage(image: BufferedImage, maxImageSize: Int, resample: Int = 1): SplitImageResult = {
    val height = image.getHeight
    val width = image.getWidth
    val maxHeight = maxImageSize
    val maxWidth = maxImageSize

    val frames = new ArrayBuffer[BufferedImage]()

    if (height > maxHeight || width > maxWidth) {
      // Calculate the number of splits
      val numSplitsH = math.ceil(height.toDouble / maxHeight).toInt
      val numSplitsW = math.ceil(width.toDouble / maxWidth).toInt

      // Calculate the optimal width and height for the sub-images
      val optimalHeight = math.ceil(height.toDouble / numSplitsH).toInt
      val optimalWidth = math.ceil(width.toDouble / numSplitsW).toInt

      // Iterate through each row and column
      for (r <- 0 until numSplitsH) {
        for (c <- 0 until numSplitsW) {
          // Calculate the starting point of the crop
          val startX = c * optimalWidth
          val startY = r * optimalHeight

          // Calculate the ending point of the crop
          val endX = math.min(startX + optimalWidth, width)
          val endY = math.min(startY + optimalHeight, height)

          // Crop the image
          val croppedImage = _crop(image, startX, startY, endX, endY)
          frames += croppedImage
        }
      }

      // For the global image at the end, we resize it to match the max_image_size
      val resizedImage = resizeBufferedImage(maxWidth, maxHeight, resample)(image)
      frames += resizedImage

      SplitImageResult(frames.toSeq, numSplitsH, numSplitsW)
    } else {
      // If image is smaller than max size, just add the original image
      frames += image
      SplitImageResult(frames.toSeq, 0, 0)
    }
  }

  case class ImageSize(height: Int, width: Int)

  private def resizeOutputSizeRescaleToMaxLen(
      height: Int,
      width: Int,
      minLen: Int = 1,
      maxLen: Option[Int] = None): ImageSize = {
    val effectiveMaxLen = maxLen.getOrElse(math.max(height, width))
    val aspectRatio = width.toDouble / height

    val (newHeight, newWidth) = if (width >= height) {
      val newWidth = effectiveMaxLen
      val newHeight = (newWidth / aspectRatio).toInt
      (if (newHeight % 2 != 0) newHeight + 1 else newHeight, newWidth)
    } else {
      val newHeight = effectiveMaxLen
      val newWidth = (newHeight * aspectRatio).toInt
      (newHeight, if (newWidth % 2 != 0) newWidth + 1 else newWidth)
    }

    // Avoid resizing to a size smaller than minLen
    ImageSize(height = math.max(newHeight, minLen), width = math.max(newWidth, minLen))
  }

  private def resizeOutputSizeScaleBelowUpperBound(
      height: Int,
      width: Int,
      maxLen: Option[Int] = None): ImageSize = {
    val effectiveMaxLen = maxLen.getOrElse(math.max(height, width))
    val aspectRatio = width.toDouble / height

    val (newHeight, newWidth) = if (width >= height && width > effectiveMaxLen) {
      val newWidth = effectiveMaxLen
      val newHeight = (newWidth / aspectRatio).toInt
      (newHeight, newWidth)
    } else if (height > width && height > effectiveMaxLen) {
      val newHeight = effectiveMaxLen
      val newWidth = (newHeight * aspectRatio).toInt
      (newHeight, newWidth)
    } else {
      (height, width)
    }

    // Avoid resizing to a size smaller than 1
    ImageSize(height = math.max(newHeight, 1), width = math.max(newWidth, 1))
  }

  def getResizeOutputImageSize(
      image: BufferedImage,
      resolutionMaxSide: Int,
      maxImageSize: Int = MAX_IMAGE_SIZE): ImageSize = {
    val height = image.getHeight
    val width = image.getWidth

    // Find the output size, when rescaling the longest edge to max_len and preserving the aspect ratio
    val firstResize =
      resizeOutputSizeRescaleToMaxLen(height, width, maxLen = Some(resolutionMaxSide))

    // Find the output size when scaling the image to be below the MAX_IMAGE_SIZE
    resizeOutputSizeScaleBelowUpperBound(
      firstResize.height,
      firstResize.width,
      maxLen = Some(maxImageSize))
  }

//  def getMaxHeightWidth(imagesList: Seq[Seq[BufferedImage]]): ImageSize = {
//    var maxHeight = Int.MinValue
//    var maxWidth = Int.MinValue
//
//    for (images <- imagesList) {
//      for (image <- images) {
//        val height = image.getHeight
//        val width = image.getWidth
//        maxHeight = math.max(height, maxHeight)
//        maxWidth = math.max(width, maxWidth)
//      }
//    }
//
//    ImageSize(maxHeight, maxWidth)
//  }

  def makePixelMask(image: BufferedImage, outputSize: ImageSize): Array[Array[Int]] = {
    val inputHeight = image.getHeight
    val inputWidth = image.getWidth

    // Create a 2D array filled with zeros
    val mask = Array.ofDim[Int](outputSize.height, outputSize.width)

    // Fill the valid region with ones
    for (y <- 0 until math.min(inputHeight, outputSize.height)) {
      for (x <- 0 until math.min(inputWidth, outputSize.width)) {
        mask(y)(x) = 1
      }
    }

    mask
  }

  def resizeWithLongestEdge(
      image: BufferedImage,
      longestEdge: Int,
      resample: Int = 1): BufferedImage = {
    val outputSize = getResizeOutputImageSize(image, longestEdge)
    // set the output size to be 1536x1152
//    val outputSize = ImageSize(1152, 1536)
    resizeBufferedImage(outputSize.width, outputSize.height, resample)(image)
  }

  def resizeForVisionEncoder(
      image: BufferedImage,
      visionEncoderMaxSize: Int,
      resample: Int = 1): BufferedImage = {
    val height = image.getHeight
    val width = image.getWidth
    val aspectRatio = width.toDouble / height

    val (newHeight, newWidth) = if (width >= height) {
      val newWidth = math.ceil(width.toDouble / visionEncoderMaxSize).toInt * visionEncoderMaxSize
      val newHeight = (newWidth / aspectRatio).toInt
      val finalHeight =
        math.ceil(newHeight.toDouble / visionEncoderMaxSize).toInt * visionEncoderMaxSize
      (finalHeight, newWidth)
    } else {
      val newHeight =
        math.ceil(height.toDouble / visionEncoderMaxSize).toInt * visionEncoderMaxSize
      val newWidth = (newHeight * aspectRatio).toInt
      val finalWidth =
        math.ceil(newWidth.toDouble / visionEncoderMaxSize).toInt * visionEncoderMaxSize
      (newHeight, finalWidth)
    }

    // Use our existing resize method with the calculated dimensions
    resizeBufferedImage(newWidth, newHeight, resample)(image)
  }

  case class BatchFeature(
      paddedImages: Seq[Seq[Array[Array[Array[Float]]]]],
      pixelMasks: Option[Seq[Seq[Array[Array[Int]]]]] = None)

  private def getMaxHeightWidth(imagesList: Seq[Seq[Array[Array[Array[Float]]]]]): ImageSize = {
    var maxHeight = Int.MinValue
    var maxWidth = Int.MinValue

    for (images <- imagesList) {
      for (image <- images) {
        // image shape is now (channels, height, width)
        val height = image(0).length // height is the second dimension
        val width = image(0)(0).length // width is the third dimension
        maxHeight = math.max(height, maxHeight)
        maxWidth = math.max(width, maxWidth)
      }
    }

    ImageSize(maxHeight, maxWidth)
  }

  private def makePixelMask(
      image: Array[Array[Array[Float]]],
      outputSize: ImageSize): Array[Array[Int]] = {
    // image shape is now (channels, height, width)
    val inputHeight = image(0).length // height is the second dimension
    val inputWidth = image(0)(0).length // width is the third dimension

    // Create a 2D array filled with zeros
    val mask = Array.ofDim[Int](outputSize.height, outputSize.width)

    // Fill the valid region with ones
    for (y <- 0 until math.min(inputHeight, outputSize.height)) {
      for (x <- 0 until math.min(inputWidth, outputSize.width)) {
        mask(y)(x) = 1
      }
    }

    mask
  }

  private def padImage(
      image: Array[Array[Array[Float]]],
      outputSize: ImageSize,
      constantValue: Float = 0f): Array[Array[Array[Float]]] = {
    // image shape is now (channels, height, width)
    val inputHeight = image(0).length // height is the second dimension
    val inputWidth = image(0)(0).length // width is the third dimension
    val outputHeight = outputSize.height
    val outputWidth = outputSize.width
    val numChannels = image.length // channels is the first dimension

    // Create a new array with the output size (channels, height, width)
    val paddedImage = Array.ofDim[Float](numChannels, outputHeight, outputWidth)

    // Fill with constant value
    for (c <- 0 until numChannels) {
      for (y <- 0 until outputHeight) {
        for (x <- 0 until outputWidth) {
          paddedImage(c)(y)(x) = constantValue
        }
      }
    }

    // Copy the original image
    for (c <- 0 until numChannels) {
      for (y <- 0 until math.min(inputHeight, outputHeight)) {
        for (x <- 0 until math.min(inputWidth, outputWidth)) {
          paddedImage(c)(y)(x) = image(c)(y)(x)
        }
      }
    }

    paddedImage
  }

  def pad(
      images: Seq[Seq[Array[Array[Array[Float]]]]],
      constantValue: Float = 0f,
      returnPixelMask: Boolean = true): BatchFeature = {
    // Get the maximum dimensions across all images
    val padSize = getMaxHeightWidth(images)

    // Get batch dimensions
    val batchSize = images.size
    val maxNumImages = images.map(_.size).max
    val numChannels = images(0)(0).length // channels is the first dimension

    // Create empty padded images and masks
    val paddedImages = Array.ofDim[Array[Array[Array[Float]]]](batchSize, maxNumImages)
    val pixelMasks =
      if (returnPixelMask) Some(Array.ofDim[Array[Array[Int]]](batchSize, maxNumImages)) else None

    // Process each batch and image
    for (batchIdx <- 0 until batchSize) {
      for (sampleIdx <- 0 until maxNumImages) {
        if (sampleIdx < images(batchIdx).size) {
          // Pad the actual image
          paddedImages(batchIdx)(sampleIdx) =
            padImage(images(batchIdx)(sampleIdx), padSize, constantValue)

          // Create pixel mask if requested
          if (returnPixelMask) {
            pixelMasks.get(batchIdx)(sampleIdx) =
              makePixelMask(images(batchIdx)(sampleIdx), padSize)
          }
        } else {
          // Create empty image for padding
          paddedImages(batchIdx)(sampleIdx) =
            Array.ofDim[Float](numChannels, padSize.height, padSize.width)

          // Create empty mask if requested
          if (returnPixelMask) {
            pixelMasks.get(batchIdx)(sampleIdx) = Array.ofDim[Int](padSize.height, padSize.width)
          }
        }
      }
    }

    BatchFeature(
      paddedImages = paddedImages.map(_.toSeq).toSeq,
      pixelMasks = pixelMasks.map(_.map(_.toSeq).toSeq))
  }

  private def promptSplitImage(
      imageSeqLen: Int,
      imageRows: Int,
      imageCols: Int,
      fakeTokenAroundImage: String,
      imageToken: String,
      globalImageToken: String): String = {
    val textSplitImages = new StringBuilder()

    for (nH <- 0 until imageRows) {
      for (nW <- 0 until imageCols) {
//        textSplitImages.append("\n")
        textSplitImages.append(s"${fakeTokenAroundImage}")
        textSplitImages.append(s"<row_${nH + 1}_col_${nW + 1}>")
        textSplitImages.append(imageToken)
//         repeat imageToken for imageSeqLen times
//        for (_ <- 0 until imageSeqLen) {
//          textSplitImages.append(imageToken)
//        }
      }
      textSplitImages.append("\n")
    }

    textSplitImages.append("\n\n")
    textSplitImages.append(fakeTokenAroundImage)
    textSplitImages.append(globalImageToken)
    // repeat imageToken for imageSeqLen times
//    for (_ <- 0 until imageSeqLen) {
//      textSplitImages.append(imageToken)
//    }
    textSplitImages.append(imageToken)
    textSplitImages.append(fakeTokenAroundImage)

    textSplitImages.toString
  }

  private def promptSingleImage(
      imageSeqLen: Int,
      fakeTokenAroundImage: String,
      imageToken: String,
      globalImageToken: String): String = {
    fakeTokenAroundImage +
      globalImageToken +
      (imageToken * imageSeqLen) +
      fakeTokenAroundImage
  }

  def getImagePromptString(
      imageRows: Int,
      imageCols: Int,
      imageSeqLen: Int,
      fakeTokenAroundImage: String,
      imageToken: String,
      globalImageToken: String): String = {
    if (imageRows == 0 && imageCols == 0) {
      promptSingleImage(
        imageSeqLen = imageSeqLen,
        fakeTokenAroundImage = fakeTokenAroundImage,
        imageToken = imageToken,
        globalImageToken = globalImageToken)
    } else {
      promptSplitImage(
        imageSeqLen = imageSeqLen,
        imageRows = imageRows,
        imageCols = imageCols,
        fakeTokenAroundImage = fakeTokenAroundImage,
        imageToken = imageToken,
        globalImageToken = globalImageToken)
    }
  }

}
