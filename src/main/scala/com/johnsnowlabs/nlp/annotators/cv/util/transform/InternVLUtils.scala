package com.johnsnowlabs.nlp.annotators.cv.util.transform
import java.awt.image.BufferedImage
import java.awt.Image
import scala.collection.mutable.ListBuffer

object InternVLUtils {
  def findClosestAspectRatio(
      aspectRatio: Double,
      targetRatios: Seq[(Int, Int)],
      width: Int,
      height: Int,
      imageSize: Int): (Int, Int) = {
    var bestRatioDiff = Double.PositiveInfinity
    var bestRatio = (1, 1)
    val area = width * height

    for ((w, h) <- targetRatios) {
      val targetAspect = w.toDouble / h
      val diff = math.abs(aspectRatio - targetAspect)
      if (diff < bestRatioDiff) {
        bestRatioDiff = diff
        bestRatio = (w, h)
      } else if (diff == bestRatioDiff) {
        if (area > 0.5 * imageSize * imageSize * w * h) {
          bestRatio = (w, h)
        }
      }
    }
    bestRatio
  }

  def resizeImage(image: BufferedImage, newWidth: Int, newHeight: Int): BufferedImage = {
    val resized = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB)
    val g = resized.createGraphics()
    g.drawImage(image.getScaledInstance(newWidth, newHeight, Image.SCALE_SMOOTH), 0, 0, null)
    g.dispose()
    resized
  }

  def cropImage(image: BufferedImage, x: Int, y: Int, w: Int, h: Int): BufferedImage = {
    image.getSubimage(x, y, w, h)
  }

  def dynamicPreprocess(
      image: BufferedImage,
      minNum: Int = 1,
      maxNum: Int = 12,
      imageSize: Int = 448,
      useThumbnail: Boolean = false): Seq[BufferedImage] = {
    val origWidth = image.getWidth
    val origHeight = image.getHeight
    val aspectRatio = origWidth.toDouble / origHeight

    val targetRatios = (for {
      n <- minNum to maxNum
      i <- 1 to n
      j <- 1 to n
      if i * j >= minNum && i * j <= maxNum
    } yield (i, j)).distinct.sortBy { case (w, h) => w * h }

    val bestRatio =
      findClosestAspectRatio(aspectRatio, targetRatios, origWidth, origHeight, imageSize)
    val (ratioW, ratioH) = bestRatio

    val targetWidth = imageSize * ratioW
    val targetHeight = imageSize * ratioH
    val blocks = ratioW * ratioH

    val resizedImage = resizeImage(image, targetWidth, targetHeight)
    val processedImages = ListBuffer[BufferedImage]()

    val cols = targetWidth / imageSize
    for (i <- 0 until blocks) {
      val x = (i % cols) * imageSize
      val y = (i / cols) * imageSize
      val cropped = cropImage(resizedImage, x, y, imageSize, imageSize)
      processedImages += cropped
    }

    if (useThumbnail && processedImages.length != 1) {
      val thumbnail = resizeImage(image, imageSize, imageSize)
      processedImages += thumbnail
    }

    assert(
      processedImages.length == blocks || (useThumbnail && processedImages.length == blocks + 1))
    processedImages.toList
  }
}
