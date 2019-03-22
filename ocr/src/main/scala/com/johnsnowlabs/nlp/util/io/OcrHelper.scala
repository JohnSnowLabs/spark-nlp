package com.johnsnowlabs.nlp.util.io

import java.awt.Image
import java.awt.image.{BufferedImage, DataBufferByte, RenderedImage}
import java.io._

import javax.imageio.ImageIO
import javax.media.jai.PlanarImage
import net.sourceforge.tess4j.ITessAPI.{TessOcrEngineMode, TessPageIteratorLevel, TessPageSegMode}
import net.sourceforge.tess4j.Tesseract
import net.sourceforge.tess4j.util.LoadLibs
import org.apache.pdfbox.pdmodel.PDDocument
import org.apache.spark.sql.{Dataset, SparkSession}
import org.slf4j.LoggerFactory

import scala.util.{Failure, Success, Try}


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
  val IMAGE_FILE = "image_file"
}

case class OcrRow(text: String, filename: String, pagenum: Int, method: String)

class OcrHelper extends ImageProcessing with Serializable {

  private def logger = LoggerFactory.getLogger("OcrHelper")
  private val imageFormats = Seq(".png", ".jpg")

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
  private var splitRegions: Boolean = true

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

  /* here we make sure '!pageSplit && regionSplit' cannot happen
  *  if regions are split, then you cannot merge pages(it's not possible).
  * */
  def setSplitPages(value: Boolean): Unit = {
    splitPages = value

    if(!splitPages)
      splitRegions = false
  }

  def getSplitPages: Boolean = splitPages

  def setSplitRegions(value: Boolean): Unit = {
    splitRegions = value

    if(splitRegions)
      splitPages = true
  }

  def getSplitRegions: Boolean = splitRegions

  def useErosion(useIt: Boolean, kSize:Int = 2, kernelShape:Int = Kernels.SQUARED): Unit = {
    if (!useIt)
      kernelSize = None
    else
      kernelSize = Some(kSize)
  }

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
    files.flatMap {
      // here we handle images directly
      case (fileName, stream) if imageFormats.exists(fileName.endsWith)=>
          doImageOcr(stream.open).map{case (pageN, region, method) => OcrRow(region, fileName, pageN, method)}

      case (fileName, stream) =>
          doPDFOcr(stream.open, fileName).map{case (pageN, region, method) => OcrRow(region, fileName, pageN, method)}
    }.filter(_.text.nonEmpty).toDS
  }

  /* WARNING: this only makes sense with splitPages == false, otherwise the map creation discards information(complete pages)
  (multiple pages per file is not supported) */
  def createMap(inputPath: String): Map[String, String] = {
    val files = getListOfFiles(inputPath)
    files.flatMap {case (fileName, stream) =>
      doPDFOcr(stream, fileName).map{case (_, region, _) => (fileName, region)}
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


  def binarize(bi: BufferedImage):BufferedImage = {

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

    // convolution and nonlinear op (minimum)
    // TODO must use adaptive thresholding
    outputData.indices.par.foreach { idx =>
      if (converted(idx) < 170) {
        outputData(idx) = fromSigned(2)
      }
      else
        outputData(idx) = fromSigned(250)
    }
    dest
  }

  // TODO: Sequence return type should be enough
  private def tesseractMethod(renderedImages:Seq[RenderedImage]): Option[Seq[(String, Int)]] = this.synchronized {
    import scala.collection.JavaConversions._

    val imageRegions = renderedImages.map(render => {
      val image = PlanarImage.wrapRenderedImage(render)

      // correct skew if parameters are provided
      val skewCorrected = halfAngle.flatMap{angle => resolution.map {res =>
        correctSkew(image.getAsBufferedImage, angle, res)
      }}.getOrElse(image.getAsBufferedImage)

      // rescale if factor provided
      var scaledImage = scalingFactor.map { factor =>
        reScaleImage(image, factor)
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


    (splitPages, splitRegions) match {
      case (true, true) =>
        Option(imageRegions.zipWithIndex.map {case (pageRegions, pagenum) =>
          pageRegions.map(r => (r, pagenum))}.flatten)
      case (true, false) =>
        // this merges regions within each page, splits the pages
        Option(imageRegions.zipWithIndex.
              map { case (pageRegions, pagenum) =>
                (pageRegions.mkString(System.lineSeparator()), pagenum)})
      case _ =>
        // don't split pages either regions, => everything coming from page 0
        Option(Seq((imageRegions.
          map {pageRegions => pageRegions.mkString(System.lineSeparator)}.
          mkString(System.lineSeparator), 0)))
     }
  }

  private def pdfboxMethod(pdfDoc: PDDocument, startPage: Int, endPage: Int): Option[Seq[(String, Int)]] = {
    println("log: extracting w/PDFBox")
    // TODO check this is getting the right page num
    val range = startPage to endPage
    if (splitPages)
      Some(Range(startPage, endPage + 1).flatMap(pagenum =>
        extractText(pdfDoc, pagenum, pagenum).map(t => (t, pagenum - 1))))
    else
      Some(extractText(pdfDoc, startPage, endPage).zipWithIndex)
  }

  private def pageOcr(pdfDoc: PDDocument, startPage: Int, endPage: Int): Seq[(Int, String, String)] = {

    var decidedMethod = preferredMethod

    val result = preferredMethod match {

      case OCRMethod.IMAGE_LAYER => tesseractMethod(getImageFromPDF(pdfDoc, startPage - 1, endPage - 1))
        .map(_.map{case (r, i) => (r.trim, i)})
        .filter(_.nonEmpty)
        .filter(content => minSizeBeforeFallback == 0 || (content.forall(_._1.length >= minSizeBeforeFallback)))
        .orElse(if (fallbackMethod) {decidedMethod = OCRMethod.TEXT_LAYER; pdfboxMethod(pdfDoc, startPage, endPage)} else None)

      case OCRMethod.TEXT_LAYER => pdfboxMethod(pdfDoc, startPage, endPage)
        .map(_.map{case (r, i) => (r.trim, i)})
        .filter(content => minSizeBeforeFallback == 0 || content.forall(_._1.length >= minSizeBeforeFallback))
        .orElse(if (fallbackMethod) {decidedMethod = OCRMethod.IMAGE_LAYER; tesseractMethod(getImageFromPDF(pdfDoc, startPage - 1, endPage - 1))} else None)

      case _ => throw new IllegalArgumentException(s"Invalid OCR Method. Must be '${OCRMethod.TEXT_LAYER}' or '${OCRMethod.IMAGE_LAYER}'")
    }
    result.map(_.map{ case (content, pagenum) => (pagenum, content, decidedMethod)})
      .getOrElse(Seq.empty[(Int, String, String)])
  }

  /*
   * fileStream: a stream to PDF files
   * filename: name of the original file(used for failure login)
   * returns sequence of (pageNumber:Int, textRegion:String)
   *
   * */
  private def doPDFOcr(fileStream:InputStream, filename:String):Seq[(Int, String, String)] = {
    val pagesTry = Try(PDDocument.load(fileStream)).map { pdfDoc =>
      val numPages = pdfDoc.getNumberOfPages
      require(numPages >= 1, "pdf input stream cannot be empty")

      val result = pageOcr(pdfDoc, 1, numPages)

      /* TODO: beware PDF box may have a potential memory leak according to,
     * https://issues.apache.org/jira/browse/PDFBOX-3388
     */
      pdfDoc.close()
      result
    }

    pagesTry match {
      case Failure(e) =>
        logger.error(s"""found a problem trying to open file filename""")
        logger.error(pagesTry.toString)
        Seq.empty
      case Success(content) =>
        content
    }
  }

  private def doImageOcr(fileStream:InputStream):Seq[(Int, String, String)] = {
    val image = ImageIO.read(fileStream)
    tesseractMethod(Seq(image)).map { _.map { case (region, pagenum) =>
         (pagenum, region, OCRMethod.IMAGE_FILE)
       }
    }.getOrElse(Seq.empty)
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

  private def getImageFromPDF(document: PDDocument, startPage: Int, endPage: Int): Seq[BufferedImage] = {
    import scala.collection.JavaConversions._
    Range(startPage, endPage + 1).flatMap(numPage => {
      val page = document.getPage(numPage)
      val multiImage = new MultiImagePDFPage(page)
      multiImage.getMergedImages
    })
  }
}
