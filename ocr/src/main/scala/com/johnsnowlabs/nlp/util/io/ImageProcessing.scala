package com.johnsnowlabs.nlp.util.io

import java.awt.image.BufferedImage

trait ImageProcessing {



  /*
   *  based on http://users.iit.demokritos.gr/~bgat/ICDAR2011_skew.pdf
   * */
  def correctSkew(image: BufferedImage): BufferedImage = {

    image
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
  def tresholdAndInvert(image: BufferedImage) = image

}
