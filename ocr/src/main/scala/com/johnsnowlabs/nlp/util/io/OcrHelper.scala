package com.johnsnowlabs.nlp.util.io

import java.awt.Image
import java.awt.image.{RenderedImage, BufferedImage}
import java.io.{File, FileInputStream, FileNotFoundException, InputStream}

import javax.media.jai.PlanarImage
import net.sourceforge.tess4j.ITessAPI.{TessOcrEngineMode, TessPageIteratorLevel, TessPageSegMode}
import net.sourceforge.tess4j.Tesseract
import net.sourceforge.tess4j.util.LoadLibs
import org.apache.pdfbox.pdmodel.graphics.form.PDFormXObject
import org.apache.pdfbox.pdmodel.graphics.image.PDImageXObject
import org.apache.pdfbox.pdmodel.{PDDocument, PDResources}
import org.apache.spark.sql.{DataFrame, SparkSession}


/*
 * Perform OCR/text extraction().
 * Receives a path to a set of PDFs
 * Returns one annotation for every region found on every page,
 * {result: text, metadata:{source_file: path, pagen_number: number}}
 *
 * can produce multiple annotations for each file, and for each page.
 */


object PageSegmentationMode {

  val AUTO = TessPageSegMode.PSM_AUTO
  val SINGLE_BLOCK = TessPageSegMode.PSM_SINGLE_BLOCK
  val SINGLE_WORD = TessPageSegMode.PSM_SINGLE_WORD
}

object PageIteratorLevel {

  val BLOCK = TessPageIteratorLevel.RIL_BLOCK
  val PARAGRAPH = TessPageIteratorLevel.RIL_PARA
  val WORD = TessPageIteratorLevel.RIL_WORD
}

object OcrHelper {

  @transient
  private var tesseractAPI : Tesseract = _

  var minTextLayerSize: Int = 10
  var pageSegmentationMode: Int = TessPageSegMode.PSM_AUTO
  var engineMode: Int = TessOcrEngineMode.OEM_LSTM_ONLY
  var pageIteratorLevel: Int = TessPageIteratorLevel.RIL_BLOCK

  /* if defined we resize the image multiplying both width and height by this value */
  var scalingFactor: Option[Float] = None

  def setPageSegMode(mode: Int) = {
    pageSegmentationMode = mode
  }

  def setPageIteratorLevel(mode: Int) = {
    pageSegmentationMode = mode
  }

  def setScalingFactor(factor:Float) = {
    scalingFactor = Some(factor)
  }

  var extractTextLayer: Boolean = true

  case class OcrRow(region: String, metadata: Map[String, String])

  private def getListOfFiles(dir: String): List[(String, FileInputStream)] = {
    val path = new File(dir)
    if (path.exists && path.isDirectory) {
      path.listFiles.filter(_.isFile).map(f => (f.getName, new FileInputStream(f))).toList
    } else if (path.exists && path.isFile) {
      List((path.getName, new FileInputStream(path)))
    } else {
      throw new FileNotFoundException("Path does not exist or is not a valid file or directory")
    }
  }

  def createDataset(spark: SparkSession, inputPath: String, outputCol: String, metadataCol: String): DataFrame = {
    import spark.implicits._
    val sc = spark.sparkContext

    val files = sc.binaryFiles(inputPath)
    files.flatMap {case (fileName, stream) =>
      doOcr(stream.open).map{case (pageN, region) => OcrRow(region, Map("source" -> fileName, "pagenum" -> pageN.toString))}
    }.toDF(outputCol, metadataCol)
  }

  def createMap(inputPath: String): Map[String, String] = {
    val files = getListOfFiles(inputPath)
    files.flatMap {case (fileName, stream) =>
      doOcr(stream).map{case (_, region) => (fileName, region)}
    }.toMap
  }

  private def tesseract:Tesseract = {
    if (tesseractAPI == null)
      tesseractAPI = initTesseract()
    tesseractAPI
  }

  private def initTesseract():Tesseract = {
    val api = new Tesseract()
    val tessDataFolder = LoadLibs.extractTessResources("tessdata")
    api.setDatapath(tessDataFolder.getAbsolutePath)
    api.setPageSegMode(pageSegmentationMode)
    api.setOcrEngineMode(engineMode)
    api
  }

  def reScaleImage(image: PlanarImage, factor: Float) = {
    val width = image.getWidth * factor
    val height = image.getHeight * factor
    image.getAsBufferedImage().
    getScaledInstance(width.toInt, height.toInt, Image.SCALE_SMOOTH)
  }

  /*
    * path: the path of the PDF
    * returns sequence of (pageNumber:Int, textRegion:String)
    *
    * */
  private def doOcr(fileStream:InputStream):Seq[(Int, String)] = {
    import scala.collection.JavaConversions._
    val pdfDoc = PDDocument.load(fileStream)
    val numPages = pdfDoc.getNumberOfPages

    /* try to extract a text layer from each page, default to OCR if not present */
    val result = Range(1, numPages + 1).flatMap { pageNum =>
      val textContent = extractText(pdfDoc, pageNum)
      // if no text layer present, do the OCR
      if (textContent.length < minTextLayerSize) {

        val renderedImage = getImageFromPDF(pdfDoc, pageNum - 1)
        val image = PlanarImage.wrapRenderedImage(renderedImage)

        val bufferedImage = scalingFactor.map { factor =>
          // scaling factor provided
          reScaleImage(image, factor)
        }.map(toBufferedImage).
          // no factor provided
          getOrElse(image.getAsBufferedImage)

        // Disable this completely for demo purposes
        val regions = tesseract.getSegmentedRegions(bufferedImage, pageIteratorLevel)
        regions.map{rectangle => (pageNum, tesseract.doOCR(bufferedImage, rectangle))}

      }
      else
        Seq((pageNum, textContent))
    }

    /* TODO: beware PDF box may have a potential memory leak according to,
     * https://issues.apache.org/jira/browse/PDFBOX-3388
     */
    pdfDoc.close()
    result
  }

  /*
  * extracts a text layer from a PDF.
  * */
  private def extractText(document: PDDocument, pageNum:Int):String = {
    import org.apache.pdfbox.text.PDFTextStripper
    val pdfTextStripper = new PDFTextStripper
    pdfTextStripper.setStartPage(pageNum)
    pdfTextStripper.setEndPage(pageNum)
    pdfTextStripper.getText(document)
  }

  /* TODO refactor, assuming single image */
  private def getImageFromPDF(document: PDDocument, pageNumber:Int): RenderedImage = {
    import scala.collection.JavaConversions._
    val page = document.getPages.get(pageNumber)
    getImagesFromResources(page.getResources)(0)
  }

  private def getImagesFromResources(resources: PDResources): java.util.ArrayList[RenderedImage]= {
    val images = new java.util.ArrayList[RenderedImage]
    import scala.collection.JavaConversions._
    for (xObjectName <- resources.getXObjectNames) {
      val xObject = resources.getXObject(xObjectName)
      xObject match {
        case _: PDFormXObject => images.addAll(getImagesFromResources(xObject.asInstanceOf[PDFormXObject].getResources))
        case _: PDImageXObject => images.add(xObject.asInstanceOf[PDImageXObject].getImage)
        case _ =>
      }
    }
    images
  }

  def toBufferedImage(img: Image): BufferedImage = {
    if (img.isInstanceOf[BufferedImage]) return img.asInstanceOf[BufferedImage]

    // Create a buffered image with transparency
    val bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB)
    // Draw the image on to the buffered image
    val bGr = bimage.createGraphics
    bGr.drawImage(img, 0, 0, null)
    bGr.dispose()
    // Return the buffered image
    bimage
  }



}
