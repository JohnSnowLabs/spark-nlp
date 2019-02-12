package com.johnsnowlabs.nlp.util.io

import java.awt.image.BufferedImage
import java.util

import org.apache.pdfbox.contentstream.PDFStreamEngine
import org.apache.pdfbox.contentstream.operator.{DrawObject, Operator}
import org.apache.pdfbox.contentstream.operator.state._
import org.apache.pdfbox.cos.{COSBase, COSName}
import org.apache.pdfbox.pdmodel.PDPage
import org.apache.pdfbox.pdmodel.graphics.PDXObject
import org.apache.pdfbox.pdmodel.graphics.form.PDFormXObject
import org.apache.pdfbox.pdmodel.graphics.image.PDImageXObject
import org.apache.pdfbox.util.Matrix


/* represents a PDF page containing multiple images, supports merging into single image */
class MultiImagePDFPage(page: PDPage) extends PDFStreamEngine {

  var mergedImage:Option[BufferedImage] = None
  var imageChunks:Seq[(BufferedImage, Float, Float)] = Seq.empty

  addOperator(new Concatenate)
  addOperator(new DrawObject)
  addOperator(new SetGraphicsStateParameters)
  addOperator(new Save)
  addOperator(new Restore)
  addOperator(new SetMatrix)

  def getMergedImages() : Option[BufferedImage] = mergedImage.orElse{
    processPage(page)
    mergedImage = Some(mergeChunks)
    mergedImage
  }

  override protected def processOperator(operator: Operator, operands: util.List[COSBase]): Unit = {

    if ("Do" == operator.getName) {
      val objectName: COSName = operands.get(0).asInstanceOf[COSName]
      val xobject: PDXObject = getResources.getXObject(objectName)
      if (xobject.isInstanceOf[PDImageXObject]) {
        val image: PDImageXObject = xobject.asInstanceOf[PDImageXObject]
        System.out.println("Found image [" + objectName.getName + "]")
        val ctmNew: Matrix = getGraphicsState.getCurrentTransformationMatrix
        imageChunks = imageChunks :+ (image.getImage, ctmNew.getTranslateX, ctmNew.getTranslateY)
        // position in user space units. 1 unit = 1/72 inch at 72 dpi
        println("position in PDF = " + ctmNew.getTranslateX + ", " + ctmNew.getTranslateY + " in user space units")
        println("width = " + image.getWidth)
      }
      else if (xobject.isInstanceOf[PDFormXObject]) {
        val form: PDFormXObject = xobject.asInstanceOf[PDFormXObject]
        showForm(form)
      }
    }
    else super.processOperator(operator, operands)
  }

  private def mergeChunks: BufferedImage = {
    // sort them by their vertical position
    val sortedChunks = imageChunks.sortBy(_._3).reverse.map(_._1)
    val maxWidth = imageChunks.map(_._1.getWidth).max
    val totalHeight = imageChunks.map(_._1.getHeight).sum
    // A new image, into which the different chunks will be merged
    val combined = new BufferedImage(maxWidth, totalHeight, BufferedImage.TYPE_BYTE_GRAY)
    val g2d = combined.createGraphics

    var currentY = 0
    sortedChunks.foreach { chunk =>
      g2d.drawImage(chunk, 0, currentY, null)
      currentY += chunk.getHeight
    }
    g2d.dispose()
    combined
  }


}
