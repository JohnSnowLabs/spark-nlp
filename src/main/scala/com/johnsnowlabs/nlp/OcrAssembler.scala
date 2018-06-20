package com.johnsnowlabs.nlp

import java.awt.Rectangle
import java.awt.image.RenderedImage
import java.io.InputStream
import java.util

import javax.media.jai.PlanarImage
import com.johnsnowlabs.nlp.{Annotation, HasAnnotatorType, HasOutputAnnotationCol}
import net.sourceforge.tess4j.ITessAPI.{TessOcrEngineMode, TessPageIteratorLevel, TessPageSegMode}
import net.sourceforge.tess4j.Tesseract
import net.sourceforge.tess4j.util.LoadLibs
import org.apache.pdfbox.pdmodel.graphics.form.PDFormXObject
import org.apache.pdfbox.pdmodel.graphics.image.PDImageXObject
import org.apache.pdfbox.pdmodel.{PDDocument, PDResources}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.util.{DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{ArrayType, MetadataBuilder, StructField, StructType}
import org.apache.spark.sql.functions._

import scala.collection.Map

/*
 * Perform OCR/text extraction().
 * Receives a path to a set of PDFs
 * Returns one annotation for every region found on every page,
 * {result: text, metadata:{source_file: path, pagen_number: number}}
 *
 * can produce multiple annotations for each file, and for each page.
 */


class OcrAnnotator(override val uid: String) extends Transformer
  with DefaultParamsWritable
  with HasAnnotatorType
  with HasOutputAnnotationCol {

  val inputPath: Param[String] = new Param[String](this, "inputPath", "input path containing the file(s) to be recognized")
  val extractTextLayer: Param[Boolean] = new Param[Boolean](this, "extractTextLayer", "indicates that non graphical textual information should be extracted from PDFs")
  val pageSegmentationMode:Param[Int] = new Param[Int](this, "pageSegmentationMode", "Tesseract's page segmentation mode")
  val engineMode:Param[Int] = new Param[Int](this, "engineMode", "Tesseract's engine mode")

  def setInputPath(str: String) = {
    set(inputPath, str)
    this
  }

  def setPageSegmentationMode(mode: Int) = {
    set(pageSegmentationMode, mode)
    this
  }

  @transient
  var tesseractAPI : Tesseract = null

  setDefault(inputPath -> "*.pdf",
    extractTextLayer -> true,
    pageSegmentationMode -> TessPageSegMode.PSM_SINGLE_BLOCK,
    engineMode -> TessOcrEngineMode.OEM_LSTM_ONLY,
    outputCol -> "ocr_text_regions"
  )

  override def transform(dataset: Dataset[_]): DataFrame = {
    import dataset.sqlContext.implicits._
    val sc = dataset.sqlContext.sparkContext

    val files = sc.binaryFiles(getOrDefault(inputPath))
    files.flatMap {case (fileName, stream) =>
      doOcr(stream.open).map{case (pageN, region) => (fileName, region, pageN)}
    }.toDF. // TODO this naming _1, _2, etc is not very robust
      withColumn(getOrDefault(outputCol), createAnnotations(col("_1"), col("_2"), col("_3"))).
      drop("_1", "_2", "_3")

  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override val annotatorType: AnnotatorType = "OCR"

  override def transformSchema(schema: StructType): StructType = {
    val metadataBuilder: MetadataBuilder = new MetadataBuilder()
    metadataBuilder.putString("annotatorType", annotatorType)
    val outputFields = schema.fields :+
      StructField(getOutputCol, ArrayType(Annotation.dataType), nullable = false, metadataBuilder.build)
    StructType(outputFields)
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
    api.setPageSegMode(getOrDefault(pageSegmentationMode))
    api.setOcrEngineMode(getOrDefault(engineMode))
    api
  }

  def this() = this(Identifiable.randomUID("ocr"))


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
      if (textContent.size < 10) { // if no text layer present, do the OCR
        val renderedImage = getImageFromPDF(pdfDoc, pageNum - 1)
        val bufferedImage = PlanarImage.wrapRenderedImage(renderedImage).getAsBufferedImage()

        // Disable this completely for demo purposes
        //val regions = tesseract.getSegmentedRegions(bufferedImage, TessPageIteratorLevel.RIL_BLOCK)
        val regions = Seq.empty[Rectangle]

        regions.map{rectangle => (pageNum, tesseract.doOCR(bufferedImage, rectangle))}
      }
      else {
        Seq((pageNum, textContent))
      }
    }

    /* TODO: beware PDF box may have a potential memory leak according to,
     * https://issues.apache.org/jira/browse/PDFBOX-3388
     */
    pdfDoc.close
    result
  }

  private def createAnnotations = udf { (path:String, region:String, pageN:Int) =>
    Seq(Annotation(annotatorType, 0, region.size, region,
      Map("source_file" -> path, "page_number" -> pageN.toString)))
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

  import java.util
  private def getImagesFromResources(resources: PDResources):util.ArrayList[RenderedImage]= {
    val images = new util.ArrayList[RenderedImage]
    import scala.collection.JavaConversions._
    for (xObjectName <- resources.getXObjectNames) {
      val xObject = resources.getXObject(xObjectName)
      if (xObject.isInstanceOf[PDFormXObject])
        images.addAll(getImagesFromResources(xObject.asInstanceOf[PDFormXObject].getResources))
      else if (xObject.isInstanceOf[PDImageXObject])
        images.add(xObject.asInstanceOf[PDImageXObject].getImage)
    }
    images
  }

}
