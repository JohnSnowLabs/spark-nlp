package com.johnsnowlabs.nlp.util.io

import java.awt.image.{BufferedImage, DataBufferByte}
import java.awt.geom.AffineTransform
import java.io.File
import java.awt.{Color, Image}

import com.johnsnowlabs.nlp.util.io.OcrHelper.toBufferedImage
import javax.media.jai.PlanarImage


trait ImageProcessing {

  /*
   *  based on http://users.iit.demokritos.gr/~bgat/ICDAR2011_skew.pdf
   * */
  protected def correctSkew(image: BufferedImage, angle:Double, resolution:Double): BufferedImage = {
    val correctionAngle = detectSkewAngle(thresholdAndInvert(image, 205, 255), angle, resolution)
    rotate(image, correctionAngle.toDouble, true)
  }


  /*
    * adaptive scaling of image according to font size
    * image will be scaled up or down so that letters have desired size
    * fontSize: in pixels
    * */
  protected def correctScale(image: BufferedImage, desiredFontSize:Int): BufferedImage = {
    val detectedFontSize = detectFontSize(thresholdAndInvert(image, 205, 255))
    val scaleFactor = desiredFontSize.toFloat / detectedFontSize
    reScaleImage(image, scaleFactor)
  }

  def reScaleImage(image: PlanarImage, factor: Float):BufferedImage = {
    reScaleImage(image.getAsBufferedImage(), factor)
  }

  def reScaleImage(image: BufferedImage, factor: Float):BufferedImage = {
    val width = image.getWidth * factor
    val height = image.getHeight * factor
    val scaledImg = image.
      getScaledInstance(width.toInt, height.toInt, Image.SCALE_AREA_AVERAGING)
    toBufferedImage(scaledImg)
  }


  /*
  * rotate an image, angle is in degrees
  *
  * adapted from https://stackoverflow.com/questions/30204114/rotating-an-image-object
  * */
  private def rotate(image:BufferedImage, angle:Double, keepWhite:Boolean = false):BufferedImage = {
    // The size of the original image
    val w = image.getWidth
    val h = image.getHeight

    // The angle of the rotation in radians
    val rads = Math.toRadians(angle)

    // Calculate the amount of space the image will need in
    // order not be clipped when it's rotated
    val sin = Math.abs(Math.sin(rads))
    val cos = Math.abs(Math.cos(rads))
    val newWidth = Math.floor(w * cos + h * sin).toInt
    val newHeight = Math.floor(h * cos + w * sin).toInt

    // A new image, into which the original will be painted
    val rotated = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_BYTE_GRAY)
    val g2d = rotated.createGraphics

    // try to keep background white
    if(keepWhite) {
      g2d.setBackground(Color.WHITE)
      g2d.fillRect(0, 0, rotated.getWidth, rotated.getHeight)
    }

    // The transformation which will be used to actually rotate the image
    // The translation, actually makes sure that the image is positioned onto
    // the viewable area of the image
    val at = new AffineTransform
    at.translate((newWidth - w) / 2, (newHeight - h) / 2)

    // Rotate about the center of the image
    val x = w / 2
    val y = h / 2
    at.rotate(rads, x, y)
    g2d.setTransform(at)

    // And we paint the original image onto the new image
    g2d.drawImage(image, 0, 0, null)
    g2d.dispose()
    rotated

  }

  /*
  * threshold and invert image
  * */

  protected def thresholdAndInvert(bi: BufferedImage, threshold:Int, maxVal:Int):BufferedImage = {

    // convert to grayscale
    val gray = new BufferedImage(bi.getWidth, bi.getHeight, BufferedImage.TYPE_BYTE_GRAY)
    val g = gray.createGraphics()
    g.drawImage(bi, 0, 0, null)
    g.dispose()

    // init
    val dest = new BufferedImage(bi.getWidth(), bi.getHeight(), BufferedImage.TYPE_BYTE_GRAY)
    val outputData = dest.getRaster().getDataBuffer().asInstanceOf[DataBufferByte].getData
    val inputData = gray.getRaster().getDataBuffer().asInstanceOf[DataBufferByte].getData

    // handle the unsigned type
    val converted = inputData.map(signedByte2UnsignedInt)

    outputData.indices.par.foreach { idx =>
      if (converted(idx) < threshold) {
        outputData(idx) = 0.toByte
      }
      else
        outputData(idx) = maxVal.toByte
    }
    dest
  }

  /* for debugging purposes only */
  protected def dumpImage(bi:BufferedImage, filename:String) = {
    import javax.imageio.ImageIO
    val outputfile = new File(filename)
    ImageIO.write(bi, "png", outputfile)
  }

  def signedByte2UnsignedInt(byte:Byte): Int = {
    if (byte < 0) 256 + byte
    else byte
  }

  def unsignedInt2signedByte(inte:Int): Byte = {
    if (inte <= 127 && inte <= 255)
      (inte - 256).toByte
    else
      inte.toByte
  }

  private def criterionFunc(projections: Array[Int]): Double =
    projections.map(col => Math.pow(col, 2)).sum

  private def minAreaRectShape(pointList: List[(Int, Int)]): (Int, Int) = {
    val maxX = pointList.maxBy(_._2)._2
    val minX = pointList.minBy(_._2)._2

    val maxY = pointList.maxBy(_._1)._1
    val minY = pointList.minBy(_._1)._1
    (maxX - minX, maxY - minY)
  }

  private def minAreaRectCoordinates(pointList: List[(Int, Int)]): (Int, Int, Int, Int) = {
    val maxX = pointList.maxBy(_._2)._2
    val minX = pointList.minBy(_._2)._2

    val maxY = pointList.maxBy(_._1)._1
    val minY = pointList.minBy(_._1)._1
    (minX, minY, maxX, maxY)
  }

  private def detectSkewAngle(image: BufferedImage, halfAngle:Double, resolution:Double): Double = {
    val angle_score = Range.Double(-halfAngle, halfAngle + resolution, resolution).par.map { angle =>
        var pointList: List[(Int, Int)] = List.empty
        val rotImage = rotate(image, angle)
        val projections: Array[Int] = Array.fill(rotImage.getWidth)(0)
        val rotImageData = rotImage.getRaster().getDataBuffer().asInstanceOf[DataBufferByte].getData
        val (imgW, imgH) = (rotImage.getWidth, rotImage.getHeight)

        Range(0, imgW).foreach { i =>
          Range(0, imgH).foreach { j =>
            val pixVal = rotImageData(j * imgW + i) // check best way to access data here
            if (pixVal == -1) {
              pointList =  (j, i) :: pointList
              projections(i) += 1
            }
          }
        }
        val (w, h) = minAreaRectShape(pointList)
        val score = criterionFunc(projections) / (w * h).toDouble
        (angle, score)
    }.toMap

  angle_score.maxBy(_._2)._1
  }

  def autocorrelation(projections: Array[Int]) = {
   // possible sizes
   Range(5, 104).map { shift =>
     (shift, projections.drop(shift).zip(projections.dropRight(shift)).map{case (x,y) => x * y / 4}.sum)
   }
  }

  def findLocalMax(shifts:List[(Int, Int)]) = {
    val gtBefore = shifts.zip(shifts.tail).map{case (x, y) => y._2 > x._2}
    val gtAfter = shifts.zip(shifts.tail).map{case (x, y) => x._2 > y._2}.tail

    val combined = gtBefore.zip(gtAfter).map{case (x,y) => x && y}
    val winnerIdx = combined.indexWhere(identity)
    shifts(winnerIdx + 1)._1

  }

  def detectFontSize(image: BufferedImage) = {
    val imageData = image.getRaster().getDataBuffer().asInstanceOf[DataBufferByte].getData
    var pointList: List[(Int, Int)] = List.empty
    val projections: Array[Int] = Array.fill(image.getHeight)(0)
    val (imgW, imgH) = (image.getWidth, image.getHeight)


    // detect square surrounding text
    Range(0, imgW).foreach { i =>
      Range(0, imgH).foreach { j =>
        val pixVal = imageData(j * imgW + i) // check best way to access data here
        if (pixVal == -1) {
          pointList =  (j, i) :: pointList
          projections(j) += 1
        }
      }
    }

    // TODO horizontal projections over cropped area
    val (minX, minY, maxX, maxY) = minAreaRectCoordinates(pointList)

    findLocalMax(autocorrelation(projections).toList)
  }
}
