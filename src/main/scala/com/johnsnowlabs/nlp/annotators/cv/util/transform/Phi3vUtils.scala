package com.johnsnowlabs.nlp.annotators.cv.util.transform
import java.awt.image.BufferedImage
import java.awt.{Color, Graphics2D}
import scala.collection.mutable.ListBuffer
import scala.collection.mutable.ArrayBuffer

import ImageResizeUtils.resizeBufferedImage

private[johnsnowlabs] object Phi3vUtils {
  // padding image

  def padding_336(image: BufferedImage): BufferedImage = {
    // Get the current width and height of the image
    val width = image.getWidth
    val height = image.getHeight

    // Calculate the target height (multiple of 336)
    val targetHeight = Math.ceil(height.toDouble / 336).toInt * 336

    // Calculate the padding for top and bottom
    val topPadding = (targetHeight - height) / 2
    val bottomPadding = targetHeight - height - topPadding

    // No padding for left and right
    val leftPadding = 0
    val rightPadding = 0

    // Create a new BufferedImage with the padded dimensions
    val paddedImage = new BufferedImage(width, targetHeight, BufferedImage.TYPE_INT_RGB)

    // Create Graphics2D object to draw the padded image
    val g2d: Graphics2D = paddedImage.createGraphics()

    // Set white background for the padding (fill)
    g2d.setColor(Color.WHITE)
    g2d.fillRect(0, 0, width, targetHeight)

    // Draw the original image onto the center of the new padded image
    g2d.drawImage(image, leftPadding, topPadding, null)

    // Dispose of the Graphics2D context
    g2d.dispose()

    // Return the new padded image
    paddedImage
  }

  def transposeImage(img: BufferedImage): BufferedImage = {
    val transposedImage = new BufferedImage(img.getHeight, img.getWidth, img.getType)
    val g2d = transposedImage.createGraphics()

    g2d.rotate(Math.PI / 2)
    g2d.translate(0, -img.getHeight)
    g2d.drawImage(img, 0, 0, null)
    g2d.dispose()

    transposedImage
  }

  def calc_padded_size(width: Int, height: Int, padding_unit: Int = 336): (Int, Int) = {
    val target_height = Math.ceil(height / padding_unit).intValue * padding_unit
    val top_padding = Math.ceil((target_height - height) / 2).intValue
    val bottom_padding = target_height - height - top_padding
    val left_padding = 0
    val right_padding = 0
    val padded_width = width + left_padding + right_padding
    val padded_height = height + top_padding + bottom_padding
    (padded_width, padded_height)
  }

  def HDTransform(img: BufferedImage, hdNum: Int = 16): BufferedImage = {
    var width = img.getWidth
    var height = img.getHeight
    var transposed = false

    // Transpose the image if width is smaller than height
    var transformedImg = img
    if (width < height) {
      transformedImg = transposeImage(transformedImg)
      transposed = true
      width = transformedImg.getWidth
      height = transformedImg.getHeight
    }

    val ratio = width.toDouble / height.toDouble
    var scale = 1

    // Calculate the scaling factor
    while (scale * math.ceil(scale / ratio) <= hdNum) {
      scale += 1
    }
    scale -= 1

    // New dimensions
    val newWidth = (scale * 336).toInt
    val newHeight = (newWidth / ratio).toInt

    // Resize the image
    transformedImg = resizeBufferedImage(newWidth, newHeight, 2)(transformedImg)

    // Apply padding to make the image 336x336
    transformedImg = padding_336(transformedImg)

    // Transpose back if needed
    if (transposed) {
      transformedImg = transposeImage(transformedImg)
    }

    transformedImg
  }

  // Function to extract a subimage and reset position information
  def getNewSubimage(
      image: BufferedImage,
      x: Int,
      y: Int,
      width: Int,
      height: Int): BufferedImage = {
    // Create a new BufferedImage to store the subimage
    val subImage = new BufferedImage(width, height, image.getType)

    // Create a Graphics2D object to draw the subimage
    val g2d: Graphics2D = subImage.createGraphics()

    // Draw the original image's subimage into the new BufferedImage
    g2d.drawImage(image, 0, 0, width, height, x, y, x + width, y + height, null)

    // Dispose the graphics context to free up resources
    g2d.dispose()

    // Return the new subimage with reset position information
    subImage
  }

  // Function to calculate the shapes (height and width of the image)
  def calculateShapes(images: List[BufferedImage]): Array[Array[Int]] = {
    images.map(img => Array(img.getHeight, img.getWidth)).toArray
  }

  // Function to calculate the number of image tokens
//  def calculateImageTokens(shapes: List[(Int, Int)]): List[Int] = {
//    shapes.map { case (h, w) =>
//      ((h / 336) * (w / 336) + 1) * 144 + 1 + ((h / 336 + 1) * 12)
//    }
//  }

  def calculateImageTokens(shapes: Array[Array[Int]]): List[Int] = {
    shapes.map { case Array(h, w) =>
      ((h / 336) * (w / 336) + 1) * 144 + 1 + ((h / 336 + 1) * 12)
    }.toList
  }

  // Function to reshape the images (assuming each image is already HD transformed)
//  def reshapeImages(
//      images: List[BufferedImage],
//      shapes: List[(Int, Int)]): List[List[BufferedImage]] = {
//    images.zip(shapes).map { case (img, (h, w)) =>
//      val numH = h / 336
//      val numW = w / 336
//      val reshapedImages = new ListBuffer[BufferedImage]
//
//      // Splitting the image into 336x336 crops
//      for (i <- 0 until numH; j <- 0 until numW) {
//        val crop = getNewSubimage(img, j * 336, i * 336, 336, 336)
//        reshapedImages += crop
//      }
//      reshapedImages.toList
//    }
//  }

  def reshapeImages(
      images: List[BufferedImage],
      shapes: Array[Array[Int]]): List[List[BufferedImage]] = {
    images.zip(shapes).map { case (img, Array(h, w)) =>
      val numH = h / 336
      val numW = w / 336
      val reshapedImages = new ListBuffer[BufferedImage]

      // Splitting the image into 336x336 crops
      for (i <- 0 until numH; j <- 0 until numW) {
        val crop = getNewSubimage(img, j * 336, i * 336, 336, 336)
        reshapedImages += crop
      }
      reshapedImages.toList
    }
  }

  // Function to concatenate global and local images (manually)
  def concatenateImages(
      globalImage: BufferedImage,
      localImages: List[BufferedImage]): BufferedImage = {
    val totalWidth = 336 * localImages.size + 336
    val totalHeight = 336
    val concatenatedImage = new BufferedImage(totalWidth, totalHeight, BufferedImage.TYPE_INT_RGB)
    val g2d: Graphics2D = concatenatedImage.createGraphics()

    // Draw global image first
    g2d.drawImage(globalImage, 0, 0, null)

    // Draw each local image next to the global image
    localImages.zipWithIndex.foreach { case (localImage, index) =>
      g2d.drawImage(localImage, (index + 1) * 336, 0, null)
    }

    g2d.dispose()
    concatenatedImage
  }

  // Function to pad the images to a specified number of crops (maxNumCrops)
  def padToMaxNumCrops(image: BufferedImage, maxNumCrops: Int): BufferedImage = {
    val width = image.getWidth
    val height = image.getHeight

    // If the number of crops is less than maxNumCrops, pad with white
    val targetWidth = 336 * maxNumCrops
    val paddedImage = new BufferedImage(targetWidth, height, BufferedImage.TYPE_INT_RGB)
    val g2d: Graphics2D = paddedImage.createGraphics()

    // Fill with white background
    g2d.setColor(Color.WHITE)
    g2d.fillRect(0, 0, targetWidth, height)

    // Draw the original image onto the white background
    g2d.drawImage(image, 0, 0, null)
    g2d.dispose()

    paddedImage
  }

  // Main function that processes the HD transformed images
  def processHdImages(
      hdImages: List[BufferedImage],
      numCrops: Int): (List[BufferedImage], Array[Array[Int]], List[Int]) = {
    // Step 1: Create global images (resize to 336x336)
    // val resizeGlobal =
    val globalImages = hdImages.map(resizeBufferedImage(336, 336, 3))

    // Step 2: Calculate shapes [(h, w)] where h, w are multiples of 336
    val shapes = calculateShapes(hdImages)

    // Step 3: Calculate number of image tokens
    val numImgTokens = calculateImageTokens(shapes)

    // Step 4: Reshape the HD images into 336x336 crops
    val reshapedHdImages = reshapeImages(hdImages, shapes)

    // Step 5: Concatenate global and local images
    val concatenatedImages =
      globalImages.zip(reshapedHdImages).map { case (globalImage, localImages) =>
        concatenateImages(globalImage, localImages)
      }

    // Step 6: Pad to max_num_crops if necessary
    val paddedImages = concatenatedImages.map(padToMaxNumCrops(_, numCrops + 1))
    (paddedImages, shapes, numImgTokens)
  }

  // Function to normalize pixel values of an image crop
  def normalizeImageCrop(
      imgCrop: Array[Array[Array[Int]]],
      mean: Array[Double],
      std: Array[Double]): Array[Array[Array[Float]]] = {
    val channels = imgCrop.length
    val height = imgCrop(0).length
    val width = imgCrop(0)(0).length

    // Create a 3D array for normalized values
    val normalizedCrop = Array.ofDim[Float](channels, height, width)

    for (c <- 0 until channels) {
      for (y <- 0 until height) {
        for (x <- 0 until width) {
          // Normalize the pixel value: (value - mean) / std
          normalizedCrop(c)(y)(x) = (imgCrop(c)(y)(x) / 255.0 - mean(c)).toFloat / std(c).toFloat
        }
      }
    }

    normalizedCrop
  }

  // Helper function to convert a BufferedImage crop to a 3D array (3, 336, 336) for RGB channels
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

  // Function to split an image into 336x336 crops, convert to a 3D array, and normalize if required
  def splitImageToCrops(
      image: BufferedImage,
      cropSize: Int = 336,
      normalize: Boolean = false,
      mean: Array[Double] = Array(0.48145466, 0.4578275, 0.40821073),
      std: Array[Double] = Array(0.26862954, 0.26130258, 0.27577711))
      : (Array[Array[Array[Array[Float]]]], Int) = {
    val height = image.getHeight
    val width = image.getWidth

    // Number of crops along height and width
    val numHCrops = height / cropSize
    val numWCrops = width / cropSize

    // Store the crops in a 4D array (numCrops, 3, 336, 336)
    val cropsBuffer = ArrayBuffer[Array[Array[Array[Float]]]]()

    for (i <- 0 until numHCrops) {
      for (j <- 0 until numWCrops) {
        // Extract a crop of 336x336
        val imgCrop = image.getSubimage(j * cropSize, i * cropSize, cropSize, cropSize)
        // Convert the crop to a 3D array (3, 336, 336)
        val cropArray = imageCropToArray(imgCrop)

        // Normalize the crop if the option is enabled
        val normalizedCrop = if (normalize) {
          normalizeImageCrop(cropArray, mean, std)
        } else {
          // Convert Int array to Double array if normalization is off
          cropArray.map(_.map(_.map(_.toFloat / 255.0.toFloat)))
        }

        cropsBuffer.append(normalizedCrop)
      }
    }

    // Convert ArrayBuffer to an array
    (cropsBuffer.toArray, numHCrops * numWCrops)
  }

  // Function to convert processedImages (BufferedImages) into a 5D array (b, h//336 * w//336, 3, 336, 336)
  def processedImagesTo5DArray(
      processedImages: List[BufferedImage],
      normalize: Boolean = false,
      mean: Array[Double] = Array(0.48145466, 0.4578275, 0.40821073),
      std: Array[Double] = Array(0.26862954, 0.26130258, 0.27577711))
      : (Array[Array[Array[Array[Array[Float]]]]]) = {
    // Store the 5D array (b, h//336 * w//336, 3, 336, 336)
    val batchBuffer = ArrayBuffer[Array[Array[Array[Array[Float]]]]]()
    // Process each image in the batch
    processedImages.foreach { img =>
      // Split the image into crops, convert each crop into a 3D array, and normalize if required
      val (imageCropsArray, numCrops) =
        splitImageToCrops(img, normalize = normalize, mean = mean, std = std)
      batchBuffer.append(imageCropsArray)
    }

    // Convert ArrayBuffer to array (b, numCrops, 3, 336, 336)
    batchBuffer.toArray
  }
}
