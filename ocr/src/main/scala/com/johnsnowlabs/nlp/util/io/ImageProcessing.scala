package com.johnsnowlabs.nlp.util.io

import java.awt.image.{BufferedImage, DataBufferByte}
import java.io.File


trait ImageProcessing {

  /*
   *  based on http://users.iit.demokritos.gr/~bgat/ICDAR2011_skew.pdf
   * */
  def correctSkew(image: BufferedImage): BufferedImage = {
    val angle = detectSkewAngle(thresholdAndInvert(image, 205, 255), 5)
    rotate(image, angle.toDouble)
  }

  /*
  * angle is in degrees
  * */
  def rotate(image:BufferedImage, angle:Double):BufferedImage = {
    val g2d = image.createGraphics()
    g2d.rotate(Math.toRadians(angle))
    image
  }

  /*
  * threshold and invert image
  * */

  def thresholdAndInvert(bi: BufferedImage, threshold:Int, maxVal:Int):BufferedImage = {

    // convert to grayscale
    val gray = new BufferedImage(bi.getWidth, bi.getHeight, BufferedImage.TYPE_BYTE_GRAY)
    val g = gray.createGraphics()
    g.drawImage(bi, 0, 0, null)
    g.dispose()

    // init
    val dest = new BufferedImage(bi.getWidth(), bi.getHeight(), BufferedImage.TYPE_BYTE_GRAY)
    val outputData = dest.getRaster().getDataBuffer().asInstanceOf[DataBufferByte].getData
    val inputData = gray.getRaster().getDataBuffer().asInstanceOf[DataBufferByte].getData

    // handle the unsigned type signedByte2UnsignedInt =fromUnsigned
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
  def dumpImage(bi:BufferedImage, filename:String) = {
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

  def criterionFunc(projections: Array[Int]): Double =
    projections.map(col => Math.pow(col, 2)).sum

  def minAreaRect(pointList: Array[(Int, Int)]): (Int, Int) = {
    val maxX = pointList.maxBy(_._2)._2
    val minX = pointList.minBy(_._2)._2

    val maxY = pointList.maxBy(_._1)._1
    val minY = pointList.minBy(_._1)._1
    (maxX - minX, maxY - minY)
  }

  def detectSkewAngle(image: BufferedImage, halfAngle:Int): Int = {
    var angle_score = Map[Int, Double]()

    Range(-halfAngle, halfAngle + 1).foreach { angle =>
        var pointList: Array[(Int, Int)] = Array.empty
        val rotImage = rotate(image, angle)
        val projections: Array[Int] = Array.fill(rotImage.getWidth)(0)
        val rotImageData = rotImage.getRaster().getDataBuffer().asInstanceOf[DataBufferByte].getData
        val (imgW, imgH) = (rotImage.getWidth, rotImage.getHeight)


        dumpImage(rotImage, angle.toString + ".png")
        Range(0, imgW).par.foreach { i =>
          var j: Int = 0
          Range(0, imgH).foreach { j =>
            val pixVal = rotImageData(j * imgW + i) // check best way to access data here
            if (pixVal == 255.toByte) {
              pointList = pointList :+ (j, i)
              projections(i) += 1
            }
          }
        }
      val (w, h) = minAreaRect(pointList)
      val score = criterionFunc(projections) / (w * h).toDouble
      angle_score = angle_score + (angle -> score)
    }
  angle_score.maxBy(_._2)._1
  }
}
