package com.johnsnowlabs.nlp.annotators.cv.util.transform

import scala.collection.mutable.ListBuffer
import java.awt.image.BufferedImage
import scala.collection.mutable.ArrayBuffer
import ImageResizeUtils.resizeBufferedImage
import scala.collection.mutable.ArrayBuffer
import scala.math.max

object MllamaUtils {

  /** Get all supported aspect ratios for a given max number of image tiles
    *
    * @param maxImageTiles
    * @return
    */
  def getAllSupportedAspectRatios(maxImageTiles: Int): List[(Int, Int)] = {
    val aspectRatios = ListBuffer[(Int, Int)]()
    for (width <- 1 to maxImageTiles) {
      for (height <- 1 to maxImageTiles) {
        if (width * height <= maxImageTiles) {
          aspectRatios += ((width, height))
        }
      }
    }

    aspectRatios.toList
  }

  /** Get the size of the image that fits the canvas
    *
    * @param imageHeight
    * @param imageWidth
    * @param canvasHeight
    * @param canvasWidth
    * @param tileSize
    * @return
    */
  def getImageSizeFitToCanvas(
      imageHeight: Int,
      imageWidth: Int,
      canvasHeight: Int,
      canvasWidth: Int,
      tileSize: Int): (Int, Int) = {
    val targetWidth = math.max(math.min(imageWidth, canvasWidth), tileSize)
    val targetHeight = math.max(math.min(imageHeight, canvasHeight), tileSize)

    val scaleH = targetHeight.toDouble / imageHeight.toDouble
    val scaleW = targetWidth.toDouble / imageWidth.toDouble

    if (scaleW < scaleH) {
      (targetWidth, math.min(math.floor(imageHeight * scaleW).toInt, targetHeight))
    } else {
      (math.min(math.floor(imageWidth * scaleH).toInt, targetWidth), targetHeight)
    }
  }

  /** Get the optimal tiled canvas size for the image
    *
    * @param imageHeight
    * @param imageWidth
    * @param maxImageTiles
    * @param tileSize
    * @return
    */
  def getOptimalTiledCanvas(
      imageHeight: Int,
      imageWidth: Int,
      maxImageTiles: Int,
      tileSize: Int): (Int, Int) = {
    val possibleTileArrangements = getAllSupportedAspectRatios(maxImageTiles)
    val possibleCanvasSizes = possibleTileArrangements.map { case (w, h) =>
      (w * tileSize, h * tileSize)
    }

    val targetHeights = possibleCanvasSizes.map(_._1)
    val targetWidths = possibleCanvasSizes.map(_._2)

    val scaleH = targetHeights.map(_.toDouble / imageHeight.toDouble)
    val scaleW = targetWidths.map(_.toDouble / imageWidth.toDouble)

    val scales = scaleH.zip(scaleW).map { case (h, w) => if (w > h) h else w }

    val upScalingOptions = scales.filter(_ >= 1.0)
    val selectedScale = if (upScalingOptions.nonEmpty) {
      upScalingOptions.min
    } else {
      scales.filter(_ < 1.0).max
    }

    val chosenCanvas = possibleCanvasSizes.filter { case (_, h) =>
      (h.toDouble / imageHeight.toDouble == selectedScale) ||
      (h.toDouble / imageWidth.toDouble == selectedScale)
    }

    if (chosenCanvas.size > 1) {
      chosenCanvas.minBy { case (w, h) => w * h }
    } else {
      chosenCanvas.head
    }
  }

  /** Convert a crop of an image to a 3D array
    *
    * @param imgCrop
    * @return
    */
  def imageCropToArray(imgCrop: BufferedImage): Array[Array[Array[Int]]] = {
    val height = imgCrop.getHeight
    val width = imgCrop.getWidth

    // Create a 3D array for RGB channels
    val channels = 3
    val cropArray = Array.ofDim[Int](channels, height, width)

    for (y <- 0 until height; x <- 0 until width) {
      val color = new java.awt.Color(imgCrop.getRGB(x, y))
      cropArray(0)(y)(x) = color.getRed // Red channel
      cropArray(1)(y)(x) = color.getGreen // Green channel
      cropArray(2)(y)(x) = color.getBlue // Blue channel
    }

    cropArray
  }

  /** Split an image into tiles
    *
    * @param image
    * @param numTilesHeight
    * @param numTilesWidth
    * @return
    */
  def splitToTiles(
      image: BufferedImage,
      numTilesHeight: Int,
      numTilesWidth: Int): Array[Array[Array[Array[Float]]]] = {
    val cropHeight = image.getHeight / numTilesHeight
    val cropWidth = image.getWidth / numTilesWidth

    val cropsBuffer = ArrayBuffer[Array[Array[Array[Float]]]]()

    for (i <- 0 until numTilesHeight) {
      for (j <- 0 until numTilesWidth) {
        // Extract a crop of 336x336
        val imgCrop = image.getSubimage(j * cropHeight, i * cropWidth, cropHeight, cropWidth)
        // Convert the crop to a 3D array (3, 336, 336)
        val cropArray = imageCropToArray(imgCrop)

        // Normalize the crop if the option is enabled
        val normalizedCrop = {
          // Convert Int array to Double array if normalization is off
          cropArray.map(_.map(_.map(_.toFloat / 255.0.toFloat)))
        }

        cropsBuffer.append(normalizedCrop)
      }
    }
    cropsBuffer.toArray
  }

  /** Convert a 3D array to a BufferedImage
    *
    * @param imageArray
    * @return
    */
  def arrayToBufferedImage(imageArray: Array[Array[Array[Int]]]): BufferedImage = {
    val height = imageArray(0).length
    val width = imageArray(0)(0).length

    val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)

    for (y <- 0 until height; x <- 0 until width) {
      val rgb = imageArray.map(_(y)(x)).map(_.toByte)
      val color = new java.awt.Color(rgb(0), rgb(1), rgb(2))
      image.setRGB(x, y, color.getRGB)
    }

    image
  }

  /** Convert a 3D array of floats to a BufferedImage
    *
    * @param imageArray
    * @return
    */
  def floatArrayToBufferedImage(
      imageArray: Array[Array[Array[Float]]],
      rescaleFactor: Double): BufferedImage = {
    val height = imageArray(0).length
    val width = imageArray(0)(0).length

    val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB)

    for (y <- 0 until height; x <- 0 until width) {
      val rgb = imageArray.map(_(y)(x)).map { x => (x * (1 / rescaleFactor)).toInt }
      val color = new java.awt.Color(rgb(0), rgb(1), rgb(2))
      image.setRGB(x, y, color.getRGB)
    }

    image
  }

  /** Pack images into a 6D array
    *
    * @param batchImages
    * @param maxImageTiles
    * @return
    */
  def packImages(
      batchImages: Array[Array[Array[Array[Array[Float]]]]],
      maxImageTiles: Int): (Array[Array[Array[Array[Array[Array[Float]]]]]], List[List[Int]]) = {
    val batchSize = batchImages.size
    val maxNumImages = batchImages.map(_.length).max

    val channels = batchImages.head.head.length
    val tileHeight = batchImages.head.head.head.length
    val tileWidth = batchImages.head.head.head.head.length

    // (batch_size, max_num_images, max_image_tiles, channels, tile_height, tile_width).
    val stackedImages = ArrayBuffer[Array[Array[Array[Array[Array[Float]]]]]]()

    val allNumTiles = ListBuffer.empty[List[Int]]

    // go over each sample in the batch
    for ((images, i) <- batchImages.zipWithIndex) {
      val numSampleTiles = ListBuffer.empty[Int]
      val tempStackedImages = ArrayBuffer[Array[Array[Array[Array[Float]]]]]()
      // go over each image in the sample

      for ((image, j) <- images.zipWithIndex) {
        val tempStackedTiles = ArrayBuffer[Array[Array[Array[Float]]]]()
        val numTiles = image.length
        numSampleTiles += numTiles
        for {
          k <- 0 until numTiles
        } {
          tempStackedTiles.append(image)
        }
        // add padded images to the sample
        for (_ <- 0 until maxImageTiles - image.length) {
          tempStackedTiles.append(Array.ofDim[Float](channels, tileHeight, tileWidth))
        }
        tempStackedImages.append(tempStackedTiles.toArray)
      }

      // add padded images to the sample.
      for (_ <- 0 until maxNumImages - images.length) {
        val tempStackedTiles = ArrayBuffer[Array[Array[Array[Float]]]]()
        for (_ <- 0 until maxImageTiles) {
          tempStackedTiles.append(Array.ofDim[Float](channels, tileHeight, tileWidth))
        }
        tempStackedImages.append(tempStackedTiles.toArray)

      }
      stackedImages.append(tempStackedImages.toArray)
      allNumTiles += numSampleTiles.toList
    }

    (stackedImages.toArray, allNumTiles.toList)
  }

  /** build aspect ratio mask
    *
    * @param aspectRatios
    * @param maxImageTiles
    * @return
    */
  def buildAspectRatioMask(
      aspectRatios: List[List[(Int, Int)]],
      maxImageTiles: Int): Array[Array[Array[Int]]] = {
    val batchSize = aspectRatios.size
    val maxNumImages = aspectRatios.map(_.size).max

    val aspectRatioMask = Array.ofDim[Int](batchSize, maxNumImages, maxImageTiles)

    // Set the first tile to 1 for all aspect ratios
    for {
      i <- 0 until batchSize
      j <- 0 until maxNumImages
    } {
      aspectRatioMask(i)(j)(0) = 1
    }

    for ((sampleAspectRatios, i) <- aspectRatios.zipWithIndex) {
      for ((numTilesW, numTilesH) <- sampleAspectRatios) {
        for (k <- 0 until numTilesW * numTilesH) {
          aspectRatioMask(i)(numTilesH)(k) = 1
        }
      }
    }

    aspectRatioMask
  }

  /** Pack aspect ratios into a 3D array
    *
    * @param aspectRatios
    * @param padValue
    * @return
    */
  def packAspectRatios(
      aspectRatios: List[List[(Int, Int)]],
      padValue: Int = 1): Array[Array[Array[Int]]] = {
    val batchSize = aspectRatios.size
    val maxNumImages = aspectRatios.map(_.size).max

    val aspectRatiosStacked = Array.fill(batchSize, maxNumImages, 2)(padValue)

    for ((row, i) <- aspectRatios.zipWithIndex) {
      if (row.nonEmpty) {
        aspectRatiosStacked(i).take(row.size) = row.map(t => Array(t._1, t._2))
      }
    }

    aspectRatiosStacked
  }

  /** Convert aspect ratios to IDs
    *
    * @param aspectRatios
    * @param maxImageTiles
    * @return
    */
  def convertAspectRatiosToIds(
      aspectRatios: List[List[(Int, Int)]],
      maxImageTiles: Int): Array[Array[Int]] = {
    val batchSize = aspectRatios.size
    val maxNumImages = aspectRatios.map(_.size).max
    val supportedAspectRatios = getAllSupportedAspectRatios(maxImageTiles)

    val aspectRatiosIds = Array.fill(batchSize, maxNumImages)(0) // Initialize with 0 for padding

    for ((sampleAspectRatios, i) <- aspectRatios.zipWithIndex) {
      for ((aspectRatio, j) <- sampleAspectRatios.zipWithIndex) {
        aspectRatiosIds(i)(j) = supportedAspectRatios.indexOf(aspectRatio) + 1
      }
    }

    aspectRatiosIds
  }

  /** Resize an image to fit the canvas
    *
    * @param width
    * @param height
    * @param resample
    * @param maxImageTiles
    * @param image
    * @return
    */
  def resizeImage(width: Int, height: Int, resample: Int, maxImageTiles: Int)(
      image: BufferedImage): (BufferedImage, (Int, Int)) = {
    val imageHeight = image.getHeight
    val imageWidth = image.getWidth

    val (canvasWidth, canvasHeight) =
      getOptimalTiledCanvas(imageHeight, imageWidth, maxImageTiles, height)

    val numTilesHeight = canvasHeight / height
    val numTilesWidth = canvasWidth / width
    (
      resizeBufferedImage(canvasWidth, canvasHeight, resample)(image),
      (numTilesHeight, numTilesWidth))
  }

  def padConstant(
      image: Array[Array[Float]],
      padding: Int,
      constantValue: Float): Array[Array[Float]] = {
    val rows = image.length
    val cols = image(0).length

    val paddedRows = rows + 2 * padding
    val paddedCols = cols + 2 * padding

    val paddedImage = Array.ofDim[Float](paddedRows, paddedCols)

    for (i <- 0 until paddedRows) {
      for (j <- 0 until paddedCols) {
        if (i >= padding && i < rows + padding && j >= padding && j < cols + padding) {
          paddedImage(i)(j) = image(i - padding)(j - padding)
        } else {
          paddedImage(i)(j) = constantValue
        }
      }
    }

    paddedImage
  }

  def padBufferedImage(
      image: BufferedImage,
      padding: (Int, Int),
      constantColor: Int): BufferedImage = {
    val originalWidth = image.getWidth
    val originalHeight = image.getHeight

    val paddedWidth = originalWidth + 2 * padding._2
    val paddedHeight = originalHeight + 2 * padding._1

    val paddedImage = new BufferedImage(paddedWidth, paddedHeight, image.getType)

    for (x <- 0 until paddedWidth; y <- 0 until paddedHeight) {
      if (x >= padding._2 && x < originalWidth + padding._2 && y >= padding._1 && y < originalHeight + padding._1) {
        paddedImage.setRGB(x, y, image.getRGB(x - padding._2, y - padding._1))
      } else {
        paddedImage.setRGB(x, y, constantColor)
      }
    }

    paddedImage
  }

  def pad(image: BufferedImage, paddingConstant: Int, aspectRatio: (Int, Int)): BufferedImage = {
    val originalWidth = image.getWidth
    val originalHeight = image.getHeight

    val numTilesHeight = aspectRatio._1
    val numTilesWidth = aspectRatio._2

    val paddedWidth = numTilesWidth * originalWidth
    val paddedHeight = numTilesHeight * originalHeight

    val paddingHeight = paddedHeight - originalHeight
    val paddingWidth = paddedWidth - originalWidth

    val paddedImage = padBufferedImage(image, (paddingHeight, paddingWidth), paddingConstant)
    paddedImage
  }

  def getCrossAttentionTokenMask(inputIds: Array[Int], imageTokenId: Int): Array[Array[Int]] = {
    val imageTokenLocations = inputIds.zipWithIndex.filter(_._1 == imageTokenId).map(_._2)

    if (imageTokenLocations.isEmpty) {
      Array.empty
    } else if (imageTokenLocations.length == 1) {
      Array(Array(imageTokenLocations(0), -1))
    } else {
      val visionMasks =
        imageTokenLocations.sliding(2).map(pair => Array(pair(0), pair(1))).toArray
      visionMasks.init.zip(visionMasks.tail).foreach { case (prev, curr) =>
        if (prev(0) + 1 == curr(0)) {
          prev(1) = curr(1)
        }
      }
      visionMasks.last(0) = visionMasks.last(0)
      visionMasks.last(1) = inputIds.length
      visionMasks
    }
  }

  def convertSparseCrossAttentionMaskToDense(
      crossAttentionTokenMask: Array[Array[Array[Int]]],
      numTiles: Array[Array[Int]],
      maxNumTiles: Int,
      length: Int): Array[Array[Array[Array[Int]]]] = {
    val batchSize = crossAttentionTokenMask.length
    val maxNumImages = crossAttentionTokenMask.map(_.length).max

    val crossAttentionMask = Array.ofDim[Int](batchSize, length, maxNumImages, maxNumTiles)

    for {
      sampleIdx <- crossAttentionTokenMask.indices
      (sampleMasks, sampleNumTiles) <- crossAttentionTokenMask(sampleIdx)
        .zip(numTiles(sampleIdx))
        .zipWithIndex
      (locations, maskNumTiles) <- sampleMasks.zip(sampleNumTiles).zipWithIndex
      if locations.length == 2
    } {
      val (start, end) = (locations(0), locations(1))
      val effectiveEnd = if (end == -1) length else math.min(end, length)
      for {
        i <- start until effectiveEnd
        j <- 0 until maskNumTiles
      } {
        crossAttentionMask(sampleIdx)(i)(maskIdx)(j) = 1
      }
    }

    crossAttentionMask
  }

}
