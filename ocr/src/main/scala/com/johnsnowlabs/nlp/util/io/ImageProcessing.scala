package com.johnsnowlabs.nlp.util.io

import java.awt.image.{BufferedImage, DataBufferByte}
import java.awt.geom.AffineTransform
import java.awt.{Color, Image}
import java.io.File

import javax.media.jai.PlanarImage
import com.johnsnowlabs.nlp.util.io.OcrHelper.toBufferedImage
import org.apache.commons.math3.stat.descriptive.moment.StandardDeviation



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
  protected def correctScale(image: BufferedImage, desiredFontSize:Int, maxSize:Int): BufferedImage = {
    val detectedFontSize = detectFontSize(thresholdAndInvert(image, 205, 255), maxSize)
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
    // TODO this is redundant, remove after merge
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
        val rotImage = rotate(image, angle)
        val projections: Array[Int] = Array.fill(rotImage.getWidth)(0)
        val rotImageData = rotImage.getRaster().getDataBuffer().asInstanceOf[DataBufferByte].getData
        val (imgW, imgH) = (rotImage.getWidth, rotImage.getHeight)

        var upMost = imgH
        var downMost = 0
        var leftMost = imgW
        var rightMost = 0

        Range(0, imgW).foreach { i =>
          Range(0, imgH).foreach { j =>
            val pixVal = rotImageData(j * imgW + i) // check best way to access data here
            if (pixVal == -1) {
              projections(i) += 1

              // find min area rectangle in-situ
              if (i < leftMost)
                leftMost = i

              if (i > rightMost)
                rightMost = i

              if (j > downMost)
                downMost = j

              if (j < upMost)
                upMost = j
            }
          }
        }


        val (w, h) = (rightMost- leftMost, downMost - upMost)
        val score = criterionFunc(projections) / (w * h).toDouble
        (angle, score)
    }.toMap

  angle_score.maxBy(_._2)._1
  }

  def autocorrelation(projections: Array[Int], maxFontSize:Int) = {
   // possible sizes
   Range(1, 6 * maxFontSize).map { shift =>
     (shift, projections.drop(shift).zip(projections.dropRight(shift)).map{case (x,y) => x * y / 4}.sum)
   }
  }

  def findLocalMax(shifts:List[(Int, Int)]):Int = {
    val gtBefore = shifts.zip(shifts.tail).map{case (x, y) => y._2 > x._2}
    val gtAfter = shifts.zip(shifts.tail).map{case (x, y) => x._2 > y._2}.tail

    val combined = gtBefore.zip(gtAfter).map{case (x,y) => x && y}
    val maxLocations = combined.zipWithIndex.filter(_._1).map(_._2.toDouble)
    val strideLens = maxLocations.zip(maxLocations.tail).map{case (x1, x2) => x2 - x1}

    val mean = strideLens.sum / strideLens.length
    val stdCalc = new StandardDeviation()
    val std = stdCalc.evaluate(strideLens.toArray)

    /* remove the ones far from the mean*/
    val filtered = strideLens.filter(len => Math.abs(len - mean) < std)
    filtered.head.toInt
  }

  private def findHighEnergyLen(projections: Array[Int], periodSize: Int) = {
    val result = Range(0, periodSize).map { shift =>
      analyzeWindow(projections.slice(shift, periodSize + shift))
    }

      result.maxBy(_._1)
  }

  protected def analyzeWindow(ints: Array[Int]): (Double, Int) = {
    /* max size of the central lobe */
    val maxPossibleRadio = ints.length / 4

    var lowerB = ints.length / 2 - 1
    var higherB = ints.length / 2 + 1

    var sumLow = ints.slice(0, lowerB).sum + ints.slice(higherB + 1, ints.length).sum
    var sumHigh = ints.slice(lowerB, higherB + 1).sum

    var scoresRadios: Seq[(Double, Int)] = Seq.empty

    Range(1, maxPossibleRadio + 1).foreach { radius =>
      val Eh = sumHigh.toDouble / (2.0 * radius + ints.size % 2)
      val El = sumLow.toDouble / (ints.length - 2.0 * radius)

      lowerB -= 1; higherB += 1

      /* reuse previous sums, don't recompute */
      sumLow -= ints(lowerB) + ints(higherB)
      sumHigh += ints(lowerB) + ints(higherB)

      scoresRadios = scoresRadios :+ (Eh / El, radius)
    }

    /* return low energy len + score */
    scoresRadios.maxBy(_._1)

  }


  def detectFontSize(image: BufferedImage, maxSize:Int) = {
    val imageData = image.getRaster().getDataBuffer().asInstanceOf[DataBufferByte].getData
    val projections: Array[Int] = Array.fill(image.getHeight)(0)
    val (imgW, imgH) = (image.getWidth, image.getHeight)

    // detect square surrounding text
    Range(0, imgW).foreach { i =>
      Range(0, imgH).foreach { j =>
        val pixVal = imageData(j * imgW + i)
        if (pixVal == -1) {
          projections(j) += 1
        }
      }
    }

    // now we get font size + interlining
    val periodSize = findLocalMax(autocorrelation(projections, 150).toList)

    val highEnLen = findHighEnergyLen(projections, periodSize)._2 * 2 + periodSize % 2
    periodSize - highEnLen
  }
}
