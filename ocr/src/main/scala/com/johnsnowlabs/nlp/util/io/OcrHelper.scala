package com.johnsnowlabs.nlp.util.io

import java.awt.Image
import java.awt.image.{BufferedImage, DataBufferByte, RenderedImage}
import java.io.{File, FileInputStream, FileNotFoundException, InputStream}

import javax.media.jai.PlanarImage
import net.sourceforge.tess4j.ITessAPI.{TessOcrEngineMode, TessPageIteratorLevel, TessPageSegMode}
import net.sourceforge.tess4j.Tesseract
import net.sourceforge.tess4j.util.LoadLibs
import org.apache.pdfbox.pdmodel.graphics.form.PDFormXObject
import org.apache.pdfbox.pdmodel.graphics.image.PDImageXObject
import org.apache.pdfbox.pdmodel.{PDDocument, PDResources}
import org.apache.spark.sql.{Dataset, SparkSession}
import org.slf4j.LoggerFactory


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

object EngineMode {

  val OEM_LSTM_ONLY = TessOcrEngineMode.OEM_LSTM_ONLY
  val DEFAULT = TessOcrEngineMode.OEM_DEFAULT
}

object PageIteratorLevel {

  val BLOCK = TessPageIteratorLevel.RIL_BLOCK
  val PARAGRAPH = TessPageIteratorLevel.RIL_PARA
  val WORD = TessPageIteratorLevel.RIL_WORD
}

object Kernels {
  val SQUARED = 0
}

object OCRMethod {
  val TEXT_LAYER = "text"
  val IMAGE_LAYER = "image"
}


object OcrHelper extends ImageProcessing {

  private val logger = LoggerFactory.getLogger("OcrHelper")

  @transient
  private var tesseractAPI : Tesseract = _

  private var preferredMethod: String = OCRMethod.TEXT_LAYER
  private var fallbackMethod: Boolean = true
  private var minSizeBeforeFallback: Int = 0

  /** Tesseract exclusive settings */
  private var pageSegmentationMode: Int = TessPageSegMode.PSM_AUTO
  private var engineMode: Int = TessOcrEngineMode.OEM_LSTM_ONLY
  private var pageIteratorLevel: Int = TessPageIteratorLevel.RIL_BLOCK
  private var kernelSize:Option[Int] = None
  private var splitPages: Boolean = true

  /* if defined we resize the image multiplying both width and height by this value */
  var scalingFactor: Option[Float] = None

  /* skew correction parameters */
  private var halfAngle: Option[Double] = None
  private var resolution: Option[Double] = None

  def setPreferredMethod(value: String): Unit = {
    require(value == OCRMethod.TEXT_LAYER || value == OCRMethod.IMAGE_LAYER, s"OCR Method must be either" +
      s"'${OCRMethod.TEXT_LAYER}' or '${OCRMethod.IMAGE_LAYER}'")
    preferredMethod = value
  }

  def getPreferredMethod: String = preferredMethod

  def setFallbackMethod(value: Boolean): Unit = {
    fallbackMethod = value
  }

  def getFallbackMethod: Boolean = fallbackMethod

  def setMinSizeBeforeFallback(value: Int): Unit = {
    minSizeBeforeFallback = value
  }

  def getMinSizeBeforeFallback: Int = minSizeBeforeFallback

  def setPageSegMode(mode: Int): Unit = {
    pageSegmentationMode = mode
  }

  def getPageSegMode: Int = {
    pageSegmentationMode
  }

  def setEngineMode(mode: Int): Unit = {
    engineMode = mode
  }

  def getEngineMode: Int = {
    engineMode
  }

  def setPageIteratorLevel(level: Int): Unit = {
    pageIteratorLevel = level
  }

  def getPageIteratorLevel: Int = {
    pageIteratorLevel
  }

  def setScalingFactor(factor:Float): Unit = {
    if (factor == 1.0f)
      scalingFactor = None
    else
      scalingFactor = Some(factor)
  }

  def setSplitPages(value: Boolean): Unit = splitPages = value

  def getSplitPages: Boolean = splitPages

  def useErosion(useIt: Boolean, kSize:Int = 2, kernelShape:Int = Kernels.SQUARED): Unit = {
    if (!useIt)
      kernelSize = None
    else
      kernelSize = Some(kSize)
  }

  case class OcrRow(text: String, filename: String, pagenum: Int, method: String)

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

  def createDataset(spark: SparkSession, inputPath: String): Dataset[OcrRow] = {
    import spark.implicits._
    val sc = spark.sparkContext

    val files = sc.binaryFiles(inputPath)
    files.flatMap {case (fileName, stream) =>
      doOcr(stream.open).map{case (pageN, region, method) => OcrRow(region, fileName, pageN, method)}
    }.filter(_.text.nonEmpty).toDS

  }

  def createMap(inputPath: String): Map[String, String] = {
    val files = getListOfFiles(inputPath)
    files.flatMap {case (fileName, stream) =>
      doOcr(stream).map{case (_, region, _) => (fileName, region)}
    }.filter(_._2.nonEmpty).toMap
  }

  /*
  * Enable/disable automatic skew(rotation) correction,
  *
  * @halfAngle, half the angle(in degrees) that will be considered for correction.
  * @resolution, the step size(in degrees) that will be used for generating correction
  * angle candidates.
  *
  * For example, for halfAngle = 2.0, and resolution 0.5,
  * candidates {-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2} will be evaluated.
  * */
  def setAutomaticSkewCorrection(useIt:Boolean, halfAngle:Double = 5.0, resolution:Double = 1.0) = {
    if(useIt) {
      this.halfAngle = Some(halfAngle)
      this.resolution = Some(resolution)
    } else {
      this.halfAngle = None
      this.resolution = None
    }
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
    val scaledImg = image.getAsBufferedImage().
    getScaledInstance(width.toInt, height.toInt, Image.SCALE_AREA_AVERAGING)
    toBufferedImage(scaledImg)
  }

  /* erode the image */
  def erode(bi: BufferedImage, kernelSize: Int) = {
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
    val converted = inputData.map(fromUnsigned)

    // define the boundaries of the squared kernel
    val width = bi.getWidth
    val rowIdxs = Range(-kernelSize, kernelSize + 1).map(_ * width)
    val colIdxs = Range(-kernelSize, kernelSize + 1)

    // convolution and nonlinear op (minimum)
    outputData.indices.par.foreach { idx =>
      var acc = Int.MaxValue
      for (ri <- rowIdxs; ci <- colIdxs) {
        val index = idx + ri + ci
        if (index > -1 && index < converted.length)
          if(acc > converted(index))
            acc = converted(index)
      }
      outputData(idx) = fromSigned(acc)
    }
    dest
  }


  def fromUnsigned(byte:Byte): Int = {
    if (byte > 0)
      byte
    else
      byte + 255
  }

  def fromSigned(integer:Int): Byte = {
    if (integer > 0 && integer < 127)
      integer.toByte
    else
      (integer - 255).toByte
  }

  private def tesseractMethod(
                               pdfDoc: PDDocument,
                               startPage: Int,
                               endPage: Int): Option[Seq[String]] = {
    import scala.collection.JavaConversions._

    val renderedImages = getImageFromPDF(pdfDoc, startPage - 1, endPage - 1)

    val imageRegions = renderedImages.flatMap(render => {
      val image = PlanarImage.wrapRenderedImage(render)

      // correct skew if parameters are provided
      val skewCorrected = halfAngle.flatMap{angle => resolution.map {res =>
        correctSkew(image.getAsBufferedImage, angle, res)
      }}.getOrElse(image.getAsBufferedImage)

<<<<<<< HEAD
      // rescale if factor provided
      val scaledImage = scalingFactor.map { factor =>
        reScaleImage(image, factor)
=======
      // rescale if factor provided, or automatic scaling enabled

      val factor = scalingFactor.orElse(
        desiredSize.flatMap(size => detectFontSize(skewCorrected).map(_.toFloat).map(size / _)))

      val scaledImage = factor.map { factor =>
        reScaleImage(skewCorrected, factor)
>>>>>>> parent of 08734703... change variable name
      }.getOrElse(skewCorrected)

      // erode if kernel provided
      val dilatedImage = kernelSize.map {kernelRadio =>
        erode(scaledImage, kernelRadio)
      }.getOrElse(scaledImage)

      // obtain regions and run OCR on each region
      val regions = {
        /** Some ugly image scenarios cause a null pointer in tesseract. Avoid here.*/
        try {
          tesseract.getSegmentedRegions(scaledImage, pageIteratorLevel).map(Some(_)).toList
        } catch {
          case _: NullPointerException =>
            logger.info(s"Tesseract failed to process a document. Falling back to text layer.")
            List()
        }
      }
      regions.flatMap(_.map { rectangle =>
        tesseract.doOCR(dilatedImage, rectangle)
      })
    })

    if (splitPages)
      Option(imageRegions)
    else
      Option(Seq(imageRegions.mkString(System.lineSeparator())))


  }

  private def pdfboxMethod(pdfDoc: PDDocument, startPage: Int, endPage: Int): Option[Seq[String]] = {

    Option(extractText(pdfDoc, startPage, endPage))

  }

  private def pageOcr(tesseract: Tesseract, pdfDoc: PDDocument, startPage: Int, endPage: Int): Seq[(Int, String, String)] = {

    var decidedMethod = preferredMethod

    val result = preferredMethod match {

      case OCRMethod.IMAGE_LAYER => tesseractMethod(pdfDoc, startPage, endPage)
        .map(_.map(_.trim))
        .filter(content => content.forall(_.nonEmpty) && (minSizeBeforeFallback == 0 || content.forall(_.length >= minSizeBeforeFallback)))
        .orElse(if (fallbackMethod) {decidedMethod = OCRMethod.TEXT_LAYER; pdfboxMethod(pdfDoc, startPage, endPage)} else None)

      case OCRMethod.TEXT_LAYER => pdfboxMethod(pdfDoc, startPage, endPage)
        .map(_.map(_.trim))
        .filter(content => content.forall(_.nonEmpty) && (minSizeBeforeFallback == 0 || content.forall(_.length >= minSizeBeforeFallback)))
        .orElse(if (fallbackMethod) {decidedMethod = OCRMethod.IMAGE_LAYER; tesseractMethod(pdfDoc, startPage, endPage)} else None)

      case _ => throw new IllegalArgumentException(s"Invalid OCR Method. Must be '${OCRMethod.TEXT_LAYER}' or '${OCRMethod.IMAGE_LAYER}'")
    }

    result.map(_.map(content => (endPage - startPage + 1, content, decidedMethod))).getOrElse(Seq.empty[(Int, String, String)])
  }

  /*
   * fileStream: a stream to PDF files
   * returns sequence of (pageNumber:Int, textRegion:String)
   *
   * */
  private def doOcr(fileStream:InputStream):Seq[(Int, String, String)] = {
    val pdfDoc = PDDocument.load(fileStream)
    val numPages = pdfDoc.getNumberOfPages
    val tesseract = initTesseract()

    require(numPages >= 1, "pdf input stream cannot be empty")

    /* try to extract a text layer from each page, default to OCR if not present */
    val result = if (splitPages) {
      Range(1, numPages + 1).flatMap { pageNum =>
        pageOcr(tesseract, pdfDoc, pageNum, pageNum)
      }
    } else {
      pageOcr(tesseract, pdfDoc, 1, numPages)
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
  private def extractText(document: PDDocument, startPage: Int, endPage: Int): Seq[String] = {
    import org.apache.pdfbox.text.PDFTextStripper
    val pdfTextStripper = new PDFTextStripper
    pdfTextStripper.setStartPage(startPage)
    pdfTextStripper.setEndPage(endPage)
    Seq(pdfTextStripper.getText(document))
  }

  /* TODO refactor, assuming single image */
  private def getImageFromPDF(document: PDDocument, startPage: Int, endPage: Int): Seq[RenderedImage] = {
    import scala.collection.JavaConversions._
    Range(startPage, endPage + 1).flatMap(numPage => {
      val page = document.getPage(numPage)
      getImagesFromResources(page.getResources).headOption
    })
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
