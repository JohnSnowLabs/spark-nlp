package com.johnsnowlabs.nlp.util.io

import java.awt.image.{BufferedImage, DataBufferByte}
import java.awt.geom.AffineTransform
import java.io.File
import java.awt.Color


trait ImageProcessing {

  /*
   *  based on http://users.iit.demokritos.gr/~bgat/ICDAR2011_skew.pdf
   * */
  protected def correctSkew(image: BufferedImage, angle:Double, resolution:Double): BufferedImage = {
    val correctionAngle = detectSkewAngle(thresholdAndInvert(image, 205, 255), angle, resolution)
    rotate(image, correctionAngle.toDouble, true)
  }

  /*
  * angle is in degrees
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
        outputData(idx) = maxVal.toByte
      }
      else
        outputData(idx) = 0.toByte
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

  private def minAreaRect(pointList: List[(Int, Int)]): (Int, Int) = {
    val maxX = pointList.maxBy(_._2)._2
    val minX = pointList.minBy(_._2)._2

    val maxY = pointList.maxBy(_._1)._1
    val minY = pointList.minBy(_._1)._1
    (maxX - minX, maxY - minY)
  }

  private def detectSkewAngle(image: BufferedImage, halfAngle:Double, resolution:Double): Double = {
    val angle_score = Range.Double(-halfAngle, halfAngle + resolution, resolution).par.map { angle =>
        var pointList: List[(Int, Int)] = List.empty
        val rotImage = rotate(image, angle)
        val projections: Array[Int] = Array.fill(rotImage.getWidth)(0)
        val rotImageData = rotImage.getRaster().getDataBuffer().asInstanceOf[DataBufferByte].getData
        val (imgW, imgH) = (rotImage.getWidth, rotImage.getHeight)

        //dumpImage(rotImage, angle.toString + ".png")
        Range(0, imgW).foreach { i =>
          var j: Int = 0
          Range(0, imgH).foreach { j =>
            val pixVal = rotImageData(j * imgW + i) // check best way to access data here
            if (pixVal == -1) {
              pointList =  (j, i) :: pointList
              projections(i) += 1
            }
          }
        }
        val (w, h) = minAreaRect(pointList)
        val score = criterionFunc(projections) / (w * h).toDouble
        (angle, score)
    }.toMap

  angle_score.maxBy(_._2)._1
  }
}
