package com.johnsnowlabs.nlp.util.io

import java.awt.Rectangle
import java.awt.image.BufferedImage

import javax.imageio.IIOImage
import net.sourceforge.tess4j.ITessAPI.TessPageIteratorLevel
import net.sourceforge.tess4j.ITessAPI.ETEXT_DESC
import net.sourceforge.tess4j.util.ImageIOHelper
import net.sourceforge.tess4j._
class TesseractAccess extends Tesseract {

  def initialize() = {
    this.init()
  }


  /**
    * Performs OCR operation.
    *
    * @param bi   a buffered image
    * @param rect the bounding rectangle defines the region of the image to be
    * recognized. A rectangle of zero dimension or <code>null</code> indicates
    *             the whole image.
    * @return the recognized text
    * @throws TesseractException
    */
  @throws[TesseractException]
  def doOCRWithConfidence(bi: BufferedImage, rect: Rectangle, level:Int): (String, Float) = {
    import scala.collection.JavaConversions._

    init()
    setTessVariables()

    val oimage = ImageIOHelper.getIIOImageList(bi).head
    setImage(oimage.getRenderedImage(), rect)
    getAPI.TessBaseAPIRecognize(getHandle, new ETEXT_DESC)
    val ri = getAPI.TessBaseAPIGetIterator(getHandle)

    val strBuffer = new StringBuilder
    var score = 0.0f
    var rCount = 0.0f
    do {
       val symbol = getAPI.TessResultIteratorGetUTF8Text(ri, level)
       score += getAPI.TessResultIteratorConfidence(ri, level)
       if (symbol != null)
          strBuffer ++= symbol.getString(0)
       rCount += 1.0
       getAPI.TessDeleteText(symbol)
    } while (getAPI.TessResultIteratorNext(ri, level) == ITessAPI.TRUE)
    dispose()
    (strBuffer.toString(), score/rCount)
  }

}
