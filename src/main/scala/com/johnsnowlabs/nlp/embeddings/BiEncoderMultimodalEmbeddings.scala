/*
 * Copyright 2017-2026 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.ml.ai.{
  BiEncoderEmbeddingPair,
  BiEncoderMultimodal,
  BiEncoderMultimodalOnnx
}
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, WriteOnnxModel}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadJsonStringAsset,
  loadTextAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.ONNX
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, IMAGE, SENTENCE_EMBEDDINGS}
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.serialization.MapFeature
import com.johnsnowlabs.nlp.util.SparkNlpConfig
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.util.ZipArchiveUtil
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.{IntArrayParam, IntParam, Param, ParamMap}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.{Metadata, MetadataBuilder, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.json4s.{DefaultFormats, JValue}
import org.json4s.jackson.JsonMethods.parse

import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption.REPLACE_EXISTING
import java.util.UUID
import scala.collection.{Map => CMap}

/** Dual-encoder multimodal embeddings annotator.
  *
  * The output is written to two derived columns based on `outputCol`:
  * `${outputCol}_doc_embeddings` and `${outputCol}_image_embeddings`.
  */
class BiEncoderMultimodalEmbeddings(override val uid: String)
    extends Transformer
    with HasInputAnnotationCols
    with HasOutputAnnotatorType
    with HasOutputAnnotationCol
    with HasImageFeatureProperties
    with HasEngine
    with WriteOnnxModel {

  def this() = this(Identifiable.randomUID("BI_ENCODER_MULTIMODAL_EMBEDDINGS"))

  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, IMAGE)
  override val outputAnnotatorType: AnnotatorType = SENTENCE_EMBEDDINGS

  val batchSize: IntParam = new IntParam(this, "batchSize", "Size of every batch.")

  val vocabulary: MapFeature[String, Int] =
    new MapFeature[String, Int](this, "vocabulary").setProtected()

  val merges: MapFeature[(String, String), Int] =
    new MapFeature[(String, String), Int](this, "merges").setProtected()

  val addedTokens: MapFeature[String, Int] =
    new MapFeature[String, Int](this, "addedTokens").setProtected()

  val instruction: Param[String] = new Param[String](
    this,
    "instruction",
    "Instruction prompt prepended to both text and image encoder inputs.")

  val minPixels: IntParam =
    new IntParam(this, "minPixels", "Minimum number of pixels allowed after smart resize.")

  val maxPixels: IntParam =
    new IntParam(this, "maxPixels", "Maximum number of pixels allowed after smart resize.")

  val spatialMergeSize: IntParam = new IntParam(
    this,
    "spatialMergeSize",
    "Spatial merge size used to map image patches into prompt image tokens.")

  val patchSize: IntParam =
    new IntParam(this, "patchSize", "Patch size used by the vision tower.")

  val temporalPatchSize: IntParam =
    new IntParam(this, "temporalPatchSize", "Temporal patch size used by the vision tower.")

  val bosTokenId: IntParam = new IntParam(this, "bosTokenId", "Beginning-of-sequence token id.")

  val eosTokenId: IntParam = new IntParam(this, "eosTokenId", "End-of-sequence token id.")

  val padTokenId: IntParam = new IntParam(this, "padTokenId", "Padding token id.")

  val imageTokenId: IntParam =
    new IntParam(this, "imageTokenId", "Special token id used for image placeholders.")

  val textModelFile: Param[String] =
    new Param[String](this, "textModelFile", "Serialized ONNX file name for the text encoder.")

  val imageModelFile: Param[String] =
    new Param[String](this, "imageModelFile", "Serialized ONNX file name for the image encoder.")

  val textDataFile: Param[String] =
    new Param[String](
      this,
      "textDataFile",
      "Optional ONNX external data file for the text encoder.")

  val imageDataFile: Param[String] = new Param[String](
    this,
    "imageDataFile",
    "Optional ONNX external data file for the image encoder.")

  val imagePromptFixTokenIds: IntArrayParam = new IntArrayParam(
    this,
    "imagePromptFixTokenIds",
    "Additional token ids appended to image prompts to match exported processor output.")

  def setBatchSize(value: Int): this.type = {
    require(value > 0, "batchSize must be greater than 0")
    set(batchSize, value)
  }

  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  def setMerges(value: Map[(String, String), Int]): this.type = set(merges, value)

  def setAddedTokens(value: Map[String, Int]): this.type = set(addedTokens, value)

  def setInstruction(value: String): this.type = set(instruction, value)

  def setMinPixels(value: Int): this.type = set(minPixels, value)

  def setMaxPixels(value: Int): this.type = set(maxPixels, value)

  def setSpatialMergeSize(value: Int): this.type = set(spatialMergeSize, value)

  def setPatchSize(value: Int): this.type = set(patchSize, value)

  def setTemporalPatchSize(value: Int): this.type = set(temporalPatchSize, value)

  def setBosTokenId(value: Int): this.type = set(bosTokenId, value)

  def setEosTokenId(value: Int): this.type = set(eosTokenId, value)

  def setPadTokenId(value: Int): this.type = set(padTokenId, value)

  def setImageTokenId(value: Int): this.type = set(imageTokenId, value)

  private[johnsnowlabs] def setTextModelFile(value: String): this.type = set(textModelFile, value)

  private[johnsnowlabs] def setImageModelFile(value: String): this.type =
    set(imageModelFile, value)

  private[johnsnowlabs] def setTextDataFile(value: String): this.type = set(textDataFile, value)

  private[johnsnowlabs] def setImageDataFile(value: String): this.type =
    set(imageDataFile, value)

  def setImagePromptFixTokenIds(value: Array[Int]): this.type = set(imagePromptFixTokenIds, value)

  def getBatchSize: Int = $(batchSize)

  def getVocabulary: Map[String, Int] = $$(vocabulary)

  def getMerges: Map[(String, String), Int] = $$(merges)

  def getAddedTokens: Map[String, Int] = $$(addedTokens)

  def getInstruction: String = $(instruction)

  def getMinPixels: Int = $(minPixels)

  def getMaxPixels: Int = $(maxPixels)

  def getSpatialMergeSize: Int = $(spatialMergeSize)

  def getPatchSize: Int = $(patchSize)

  def getTemporalPatchSize: Int = $(temporalPatchSize)

  def getBosTokenId: Int = $(bosTokenId)

  def getEosTokenId: Int = $(eosTokenId)

  def getPadTokenId: Int = $(padTokenId)

  def getImageTokenId: Int = $(imageTokenId)

  def getTextModelFile: String = $(textModelFile)

  def getImageModelFile: String = $(imageModelFile)

  def getTextDataFile: Option[String] = get(textDataFile)

  def getImageDataFile: Option[String] = get(imageDataFile)

  def getImagePromptFixTokenIds: Array[Int] = $(imagePromptFixTokenIds)

  def getOutputDocEmbeddingsCol: String = s"${getOutputCol}_doc_embeddings"

  def getOutputImageEmbeddingsCol: String = s"${getOutputCol}_image_embeddings"

  setDefault(
    batchSize -> 8,
    engine -> ONNX.name,
    instruction -> "You are a helpful assistant.",
    textModelFile -> BiEncoderMultimodalEmbeddings.textModelFile,
    imageModelFile -> BiEncoderMultimodalEmbeddings.imageModelFile,
    imagePromptFixTokenIds -> Array.empty[Int])

  private var _model: Option[Broadcast[BiEncoderMultimodal]] = None

  private val outputTypeMetadata: Metadata =
    new MetadataBuilder().putString("annotatorType", SENTENCE_EMBEDDINGS).build()

  private case class RowBatchInput(
      baseValues: Seq[Any],
      documents: Seq[Annotation],
      images: Seq[AnnotationImage])

  private val itemIdPrimaryKeys =
    Seq("item_id", "source_file", "source", "file_name", "file", "origin")

  private val pageMetadataKeys = Seq("page_number", "page_num", "page")

  private[johnsnowlabs] def setModelIfNotSet(
      spark: SparkSession,
      model: BiEncoderMultimodal): this.type = {
    if (_model.isEmpty) {
      _model = Some(spark.sparkContext.broadcast(model))
    }
    this
  }

  private def createPreprocessor: Preprocessor =
    Preprocessor(
      do_normalize = getDoNormalize,
      do_resize = getDoResize,
      feature_extractor_type = getFeatureExtractorType,
      image_mean = getImageMean,
      image_std = getImageStd,
      resample = getResample,
      size = getSize)

  private[johnsnowlabs] def setModelIfNotSet(
      spark: SparkSession,
      textOnnxWrapper: OnnxWrapper,
      imageOnnxWrapper: OnnxWrapper): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new BiEncoderMultimodalOnnx(
            textOnnxWrapper = textOnnxWrapper,
            imageOnnxWrapper = imageOnnxWrapper,
            vocabulary = getVocabulary,
            merges = getMerges,
            addedTokens = getAddedTokens,
            preprocessor = createPreprocessor,
            bosTokenId = getBosTokenId,
            eosTokenId = getEosTokenId,
            padTokenId = getPadTokenId,
            imageTokenId = getImageTokenId,
            spatialMergeSize = getSpatialMergeSize,
            patchSize = getPatchSize,
            temporalPatchSize = getTemporalPatchSize,
            minPixels = getMinPixels,
            maxPixels = getMaxPixels,
            instruction = getInstruction,
            imagePromptFixTokenIds = getImagePromptFixTokenIds)))
    }
    this
  }

  private[johnsnowlabs] def getModelIfNotSet: BiEncoderMultimodal =
    _model
      .map(_.value)
      .getOrElse(
        throw new IllegalStateException(
          s"No multimodal model backend has been configured for $uid"))

  override def transform(dataset: Dataset[_]): DataFrame = {
    require(
      validateSchema(dataset.schema),
      s"Wrong or missing inputCols annotators in $uid.\n" +
        msgHelper(dataset.schema) +
        s"\nMake sure such annotators exist with types: ${inputAnnotatorTypes.mkString(", ")}")

    val inputDataFrame = dataset.toDF()
    val outputSchema = transformSchema(inputDataFrame.schema)
    val (docInputCol, imageInputCol) = resolveInputCols(inputDataFrame.schema)
    val outputDocCol = getOutputDocEmbeddingsCol
    val outputImageCol = getOutputImageEmbeddingsCol

    val baseColumnNames =
      outputSchema.fields
        .map(_.name)
        .filterNot(name => name == outputDocCol || name == outputImageCol)
    val baseColumnIndexes = baseColumnNames.map(inputDataFrame.schema.fieldIndex)
    val docInputIndex = inputDataFrame.schema.fieldIndex(docInputCol)
    val imageInputIndex = inputDataFrame.schema.fieldIndex(imageInputCol)

    implicit val encoder: ExpressionEncoder[Row] =
      SparkNlpConfig.getEncoder(inputDataFrame, outputSchema)

    val mappedDataFrame = inputDataFrame.mapPartitions { rows =>
      rows.grouped($(batchSize)).flatMap { batchRows =>
        val batchInputs =
          batchRows.map(
            extractRowBatchInput(
              _,
              baseColumnIndexes,
              docInputIndex,
              imageInputIndex,
              docInputCol,
              imageInputCol))
        val flatDocuments = batchInputs.flatMap(_.documents)
        val flatImages = batchInputs.flatMap(_.images)

        require(
          flatDocuments.length == flatImages.length,
          s"Aligned multimodal inputs must have the same number of DOCUMENT and IMAGE annotations. " +
            s"Found ${flatDocuments.length} documents and ${flatImages.length} images.")

        val predictions =
          if (flatDocuments.nonEmpty) getModelIfNotSet.predict(flatDocuments, flatImages)
          else Seq.empty[BiEncoderEmbeddingPair]

        require(
          predictions.length == flatDocuments.length,
          s"Model backend returned ${predictions.length} predictions for ${flatDocuments.length} aligned pairs.")

        predictions.foreach { prediction =>
          require(
            prediction.docEmbedding.length == prediction.imageEmbedding.length,
            "Text and image embeddings must share the same dimension.")
        }

        var offset = 0
        batchInputs.iterator.map { rowInput =>
          val count = rowInput.documents.length
          val rowPredictions = predictions.slice(offset, offset + count)
          offset += count

          val docOutputs = rowInput.documents.zip(rowPredictions).map { case (annotation, pair) =>
            annotationToRow(buildDocumentEmbeddingAnnotation(annotation, pair.docEmbedding))
          }
          val imageOutputs = rowInput.images.zip(rowPredictions).map { case (annotation, pair) =>
            annotationToRow(buildImageEmbeddingAnnotation(annotation, pair.imageEmbedding))
          }

          Row.fromSeq(rowInput.baseValues ++ Seq(docOutputs, imageOutputs))
        }
      }
    }

    val withInputMetadata = inputDataFrame.schema.fields
      .filter(field => mappedDataFrame.columns.contains(field.name))
      .foldLeft(mappedDataFrame) { (dataFrame, field) =>
        dataFrame.withColumn(field.name, dataFrame.col(field.name).as(field.name, field.metadata))
      }

    withInputMetadata
      .withColumn(outputDocCol, col(outputDocCol).as(outputDocCol, outputTypeMetadata))
      .withColumn(outputImageCol, col(outputImageCol).as(outputImageCol, outputTypeMetadata))
  }

  override def transformSchema(schema: StructType): StructType = {
    require(
      validateSchema(schema),
      s"Wrong or missing inputCols annotators in $uid.\n" +
        msgHelper(schema) +
        s"\nMake sure such annotators exist with types: ${inputAnnotatorTypes.mkString(", ")}")

    val outputDoc =
      StructField(
        getOutputDocEmbeddingsCol,
        Annotation.arrayType,
        nullable = false,
        outputTypeMetadata)
    val outputImage =
      StructField(
        getOutputImageEmbeddingsCol,
        Annotation.arrayType,
        nullable = false,
        outputTypeMetadata)

    val baseFields =
      schema.fields
        .filterNot(field =>
          field.name == getOutputDocEmbeddingsCol || field.name == getOutputImageEmbeddingsCol)
    StructType(baseFields ++ Array(outputDoc, outputImage))
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)

    getEngine match {
      case ONNX.name =>
        getModelIfNotSet match {
          case onnxModel: BiEncoderMultimodalOnnx =>
            writeOnnxModels(
              path,
              spark,
              Seq(
                (onnxModel.textOnnxWrapper, getTextModelFile),
                (onnxModel.imageOnnxWrapper, getImageModelFile)),
              BiEncoderMultimodalEmbeddings.suffix)
          case other =>
            throw new IllegalStateException(
              s"Cannot serialize multimodal backend ${other.getClass.getSimpleName} as ONNX.")
        }
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  private def validateSchema(schema: StructType): Boolean =
    inputAnnotatorTypes.forall(checkSchema(schema, _))

  private def resolveInputCols(schema: StructType): (String, String) = {
    val colsByType =
      getInputCols.map(name => name -> schema(name).metadata.getString("annotatorType")).toMap
    val docCol = colsByType.collectFirst {
      case (name, annotatorType) if annotatorType == DOCUMENT => name
    }.get
    val imageCol = colsByType.collectFirst {
      case (name, annotatorType) if annotatorType == IMAGE => name
    }.get
    (docCol, imageCol)
  }

  private def extractRowBatchInput(
      row: Row,
      baseColumnIndexes: Seq[Int],
      docInputIndex: Int,
      imageInputIndex: Int,
      docInputCol: String,
      imageInputCol: String): RowBatchInput = {
    val documentAnnotations = toTextAnnotations(row.getAs[Seq[Row]](docInputIndex))
    val imageAnnotations = toImageAnnotations(row.getAs[Seq[Row]](imageInputIndex))

    require(
      documentAnnotations.length == imageAnnotations.length,
      s"Aligned columns $docInputCol and $imageInputCol must contain the same number of annotations per row. " +
        s"Found ${documentAnnotations.length} documents and ${imageAnnotations.length} images.")

    RowBatchInput(
      baseValues = baseColumnIndexes.map(row.get),
      documents = documentAnnotations,
      images = imageAnnotations)
  }

  private def buildDocumentEmbeddingAnnotation(
      annotation: Annotation,
      embedding: Array[Float]): Annotation = {
    val metadata = Option(annotation.metadata).getOrElse(Map.empty[String, String])
    val itemId =
      existingNonEmpty(metadata, Seq("item_id")).getOrElse(buildDocumentItemId(annotation))

    Annotation(
      annotatorType = SENTENCE_EMBEDDINGS,
      begin = annotation.begin,
      end = annotation.end,
      result = annotation.result,
      metadata = metadata ++ Map(
        "modality" -> "text",
        "item_id" -> itemId,
        "embedding_dim" -> embedding.length.toString),
      embeddings = embedding)
  }

  private def buildImageEmbeddingAnnotation(
      annotation: AnnotationImage,
      embedding: Array[Float]): Annotation = {
    val metadata = Option(annotation.metadata).getOrElse(Map.empty[String, String])
    val itemId =
      existingNonEmpty(metadata, Seq("item_id")).getOrElse(buildImageItemId(annotation))
    val resultText = Option(annotation.origin).filter(_.nonEmpty).getOrElse(itemId)

    Annotation(
      annotatorType = SENTENCE_EMBEDDINGS,
      begin = 0,
      end = math.max(resultText.length - 1, 0),
      result = resultText,
      metadata = metadata ++ Map(
        "modality" -> "image",
        "item_id" -> itemId,
        "embedding_dim" -> embedding.length.toString),
      embeddings = embedding)
  }

  private def buildDocumentItemId(annotation: Annotation): String = {
    val metadata = Option(annotation.metadata).getOrElse(Map.empty[String, String])
    val prefix = existingNonEmpty(metadata, itemIdPrimaryKeys).getOrElse("text")
    val parts = Seq(
      Some(prefix),
      firstPresentMetadata(metadata, pageMetadataKeys).map(page => s"page=$page"),
      existingNonEmpty(metadata, Seq("slide_index")).map(slide => s"slide=$slide"),
      existingNonEmpty(metadata, Seq("paragraph_index")).map(paragraph =>
        s"paragraph=$paragraph"),
      existingNonEmpty(metadata, Seq("chunk_index")).map(chunk => s"chunk=$chunk"),
      Some(s"span=${annotation.begin}-${annotation.end}"),
      Some("modality=text")).flatten
    parts.mkString("|")
  }

  private def buildImageItemId(annotation: AnnotationImage): String = {
    val metadata = Option(annotation.metadata).getOrElse(Map.empty[String, String])
    val prefix =
      existingNonEmpty(metadata, itemIdPrimaryKeys)
        .orElse(Option(annotation.origin).filter(_.nonEmpty))
        .getOrElse("image")
    val parts = Seq(
      Some(prefix),
      Option(annotation.origin)
        .filter(origin => origin.nonEmpty && origin != prefix)
        .map(origin => s"origin=$origin"),
      firstPresentMetadata(metadata, pageMetadataKeys).map(page => s"page=$page"),
      existingNonEmpty(metadata, Seq("slide_index")).map(slide => s"slide=$slide"),
      existingNonEmpty(metadata, Seq("coord")).map(coord => s"coord=$coord"),
      existingNonEmpty(metadata, Seq("orderImageIndex")).map(idx => s"image_index=$idx"),
      Some(s"size=${annotation.width}x${annotation.height}"),
      Some("modality=image")).flatten
    parts.mkString("|")
  }

  private def firstPresentMetadata(
      metadata: CMap[String, String],
      keys: Seq[String]): Option[String] =
    keys.iterator.flatMap(key => metadata.get(key).filter(_.nonEmpty)).toSeq.headOption

  private def existingNonEmpty(
      metadata: CMap[String, String],
      keys: Seq[String]): Option[String] =
    firstPresentMetadata(metadata, keys)

  private def toTextAnnotations(rows: Seq[Row]): Seq[Annotation] =
    Option(rows).getOrElse(Seq.empty).map(Annotation(_))

  private def toImageAnnotations(rows: Seq[Row]): Seq[AnnotationImage] =
    Option(rows).getOrElse(Seq.empty).map(AnnotationImage(_))

  private def annotationToRow(annotation: Annotation): Row =
    Row(
      annotation.annotatorType,
      annotation.begin,
      annotation.end,
      annotation.result,
      annotation.metadata,
      annotation.embeddings)
}

trait ReadBiEncoderMultimodalEmbeddingsDLModel
    extends ParamsAndFeaturesReadable[BiEncoderMultimodalEmbeddings] {

  private def discoverOnnxDataFile(modelPath: String, prefix: String): Option[String] =
    Option(new File(modelPath).listFiles())
      .getOrElse(Array.empty[File])
      .find(file =>
        file.isFile && file.getName.endsWith(".onnx.data") && file.getName.toLowerCase.contains(
          prefix.toLowerCase))
      .map(_.getName)

  private def loadTokenizerAssets(localModelPath: String)
      : (Map[String, Int], Map[String, Int], Map[(String, String), Int]) = {
    implicit val formats: DefaultFormats.type = DefaultFormats
    val tokenizerPath = s"$localModelPath/assets/tokenizer.json"
    val tokenizerExists = new File(tokenizerPath).exists()

    if (tokenizerExists) {
      val tokenizerConfig: JValue = parse(loadJsonStringAsset(localModelPath, "tokenizer.json"))
      var vocab: Map[String, Int] =
        (tokenizerConfig \ "model" \ "vocab").extract[Map[String, Int]]

      val merges = (tokenizerConfig \ "model" \ "merges")
        .extract[List[Array[String]]]
        .filter(_.length == 2)
        .map { case Array(left, right) => (left, right) }
        .zipWithIndex
        .toMap

      val addedTokens = (tokenizerConfig \ "added_tokens")
        .extractOpt[List[Map[String, Any]]]
        .getOrElse(Nil)
        .map { token =>
          token("content").asInstanceOf[String] -> token("id").asInstanceOf[BigInt].intValue()
        }
        .toMap

      addedTokens.foreach { case (content, id) =>
        vocab += (content -> id)
      }

      (vocab, addedTokens, merges)
    } else {
      var vocab: Map[String, Int] =
        parse(loadJsonStringAsset(localModelPath, "vocab.json")).extract[Map[String, Int]]

      val addedTokensPath = s"$localModelPath/assets/added_tokens.json"
      val addedTokens =
        if (new File(addedTokensPath).exists()) {
          parse(loadJsonStringAsset(localModelPath, "added_tokens.json"))
            .extract[Map[String, BigInt]]
            .map { case (token, id) => token -> id.intValue() }
        } else {
          Map.empty[String, Int]
        }

      addedTokens.foreach { case (content, id) =>
        vocab += (content -> id)
      }

      val merges = loadTextAsset(localModelPath, "merges.txt")
        .map(_.trim)
        .filter(line => line.nonEmpty && !line.startsWith("#"))
        .map(_.split(" "))
        .filter(_.length == 2)
        .map { case Array(left, right) => (left, right) }
        .zipWithIndex
        .toMap

      (vocab, addedTokens, merges)
    }
  }

  private def loadLocalOnnxWrapper(
      modelPath: String,
      spark: SparkSession,
      modelFileName: String,
      dataFileName: Option[String]): OnnxWrapper = {
    val modelFile = new File(modelPath, modelFileName)
    require(modelFile.exists(), s"ONNX model file $modelFileName not found under $modelPath")
    val stagedModelFileName = s"${UUID.randomUUID().toString.takeRight(12)}_$modelFileName"
    val stagedModelDir =
      Files.createTempDirectory(s"${UUID.randomUUID().toString.takeRight(12)}_onnx")
    val stagedModelFile = stagedModelDir.resolve(stagedModelFileName).toFile
    Files.copy(modelFile.toPath, stagedModelFile.toPath, REPLACE_EXISTING)
    spark.sparkContext.addFile(stagedModelFile.getAbsolutePath)

    val dataPath = dataFileName.map { fileName =>
      val dataFile = new File(modelPath, fileName)
      require(dataFile.exists(), s"ONNX external data file $fileName not found under $modelPath")
      spark.sparkContext.addFile(dataFile.getAbsolutePath)
      dataFile.getAbsolutePath
    }

    new OnnxWrapper(Some(stagedModelFileName), dataPath)
  }

  private def readSavedOnnxWrapper(
      path: String,
      spark: SparkSession,
      modelFileName: String,
      dataFileName: Option[String]): OnnxWrapper = {
    val uri = new java.net.URI(path.replaceAllLiterally("\\", "/"))
    val fileSystem = FileSystem.get(uri, spark.sparkContext.hadoopConfiguration)
    val tmpFolder = Files
      .createTempDirectory(
        s"${UUID.randomUUID().toString.takeRight(12)}${BiEncoderMultimodalEmbeddings.suffix}")
      .toFile

    val localArchive = new File(tmpFolder, modelFileName)
    fileSystem.copyToLocalFile(
      new Path(path, modelFileName),
      new Path(localArchive.getAbsolutePath))

    val localDataPath = dataFileName.map { fileName =>
      val dataSource = new Path(path, fileName)
      require(
        fileSystem.exists(dataSource),
        s"ONNX external data file $fileName not found under $path")
      val localDataFile = new File(tmpFolder, fileName)
      fileSystem.copyToLocalFile(dataSource, new Path(localDataFile.getAbsolutePath))
      spark.sparkContext.addFile(localDataFile.getAbsolutePath)
      localDataFile.getAbsolutePath
    }

    val unzippedFolder = ZipArchiveUtil.unzip(localArchive, Some(tmpFolder.getAbsolutePath))
    val extractedModelFile = Option(new File(unzippedFolder).listFiles())
      .getOrElse(Array.empty[File])
      .find(file =>
        file.isFile && file.getName.endsWith(
          ".onnx") && file.getAbsolutePath != localArchive.getAbsolutePath)
      .getOrElse(throw new IllegalStateException(
        s"No extracted ONNX file found inside serialized archive $modelFileName."))

    val stagedModelFileName =
      s"${UUID.randomUUID().toString.takeRight(12)}_${extractedModelFile.getName}"
    val stagedModelFile = new File(tmpFolder, stagedModelFileName)
    Files.copy(extractedModelFile.toPath, stagedModelFile.toPath, REPLACE_EXISTING)
    spark.sparkContext.addFile(stagedModelFile.getAbsolutePath)
    new OnnxWrapper(Some(stagedModelFileName), localDataPath)
  }

  def readModel(
      instance: BiEncoderMultimodalEmbeddings,
      path: String,
      spark: SparkSession): Unit = {
    instance.getEngine match {
      case ONNX.name =>
        val textOnnxWrapper =
          readSavedOnnxWrapper(path, spark, instance.getTextModelFile, instance.getTextDataFile)
        val imageOnnxWrapper =
          readSavedOnnxWrapper(path, spark, instance.getImageModelFile, instance.getImageDataFile)
        instance.setModelIfNotSet(spark, textOnnxWrapper, imageOnnxWrapper)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): BiEncoderMultimodalEmbeddings = {
    implicit val formats: DefaultFormats.type = DefaultFormats

    val (localModelPath, detectedEngine) =
      modelSanityCheck(
        modelPath,
        customOnnxModelNames = Some(
          List(
            BiEncoderMultimodalEmbeddings.textModelFile,
            BiEncoderMultimodalEmbeddings.imageModelFile)))

    if (detectedEngine != ONNX.name) {
      throw new Exception(notSupportedEngineError)
    }

    val modelConfig: JValue = parse(loadJsonStringAsset(localModelPath, "config.json"))
    val generationConfig: JValue =
      parse(loadJsonStringAsset(localModelPath, "generation_config.json"))
    val preprocessorConfigJsonString =
      loadJsonStringAsset(localModelPath, "preprocessor_config.json")
    val preprocessorConfigJson: JValue = parse(preprocessorConfigJsonString)
    val preprocessorConfig = Preprocessor.loadPreprocessorConfig(preprocessorConfigJsonString)

    val (vocabulary, addedTokens, merges) = loadTokenizerAssets(localModelPath)

    val bosTokenId = (modelConfig \ "bos_token_id").extract[Int]
    val eosTokenId = (modelConfig \ "eos_token_id").extract[Int]
    val padTokenId =
      (generationConfig \ "pad_token_id").extractOpt[Int].getOrElse(bosTokenId)
    val imageTokenId = (modelConfig \ "image_token_id").extract[Int]
    val spatialMergeSize = (modelConfig \ "vision_config" \ "spatial_merge_size").extract[Int]
    val patchSize = (modelConfig \ "vision_config" \ "patch_size").extract[Int]
    val temporalPatchSize = (modelConfig \ "vision_config" \ "temporal_patch_size").extract[Int]
    val minPixels = (preprocessorConfigJson \ "min_pixels").extract[Int]
    val maxPixels = (preprocessorConfigJson \ "max_pixels").extract[Int]

    val textDataFile = discoverOnnxDataFile(localModelPath, "text")
    val imageDataFile = discoverOnnxDataFile(localModelPath, "image")

    val annotatorModel = new BiEncoderMultimodalEmbeddings()
      .setVocabulary(vocabulary)
      .setMerges(merges)
      .setAddedTokens(addedTokens)
      .setDoNormalize(preprocessorConfig.do_normalize)
      .setDoResize(preprocessorConfig.do_resize)
      .setFeatureExtractorType(preprocessorConfig.feature_extractor_type)
      .setImageMean(preprocessorConfig.image_mean)
      .setImageStd(preprocessorConfig.image_std)
      .setResample(preprocessorConfig.resample)
      .setSize(preprocessorConfig.size)
      .setMinPixels(minPixels)
      .setMaxPixels(maxPixels)
      .setSpatialMergeSize(spatialMergeSize)
      .setPatchSize(patchSize)
      .setTemporalPatchSize(temporalPatchSize)
      .setBosTokenId(bosTokenId)
      .setEosTokenId(eosTokenId)
      .setPadTokenId(padTokenId)
      .setImageTokenId(imageTokenId)
      .setTextModelFile(BiEncoderMultimodalEmbeddings.textModelFile)
      .setImageModelFile(BiEncoderMultimodalEmbeddings.imageModelFile)
      .setImagePromptFixTokenIds(Array(bosTokenId, bosTokenId))

    textDataFile.foreach(annotatorModel.setTextDataFile)
    imageDataFile.foreach(annotatorModel.setImageDataFile)

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    val textOnnxWrapper =
      loadLocalOnnxWrapper(localModelPath, spark, annotatorModel.getTextModelFile, textDataFile)
    val imageOnnxWrapper =
      loadLocalOnnxWrapper(localModelPath, spark, annotatorModel.getImageModelFile, imageDataFile)

    annotatorModel.setModelIfNotSet(spark, textOnnxWrapper, imageOnnxWrapper)
  }
}

trait ReadablePretrainedBiEncoderMultimodalEmbeddings
    extends ParamsAndFeaturesReadable[BiEncoderMultimodalEmbeddings]
    with HasPretrained[BiEncoderMultimodalEmbeddings] {

  override val defaultModelName: Some[String] = Some("ops_mm_embedding_v1_2b")

  /** Java compliant-overrides */
  override def pretrained(): BiEncoderMultimodalEmbeddings = super.pretrained()

  override def pretrained(name: String): BiEncoderMultimodalEmbeddings = super.pretrained(name)

  override def pretrained(name: String, lang: String): BiEncoderMultimodalEmbeddings =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): BiEncoderMultimodalEmbeddings =
    super.pretrained(name, lang, remoteLoc)
}

object BiEncoderMultimodalEmbeddings
    extends ReadablePretrainedBiEncoderMultimodalEmbeddings
    with ReadBiEncoderMultimodalEmbeddingsDLModel {
  val suffix: String = "_bi_encoder_multimodal"
  val textModelFile: String = "text_model.onnx"
  val imageModelFile: String = "image_model.onnx"
}
