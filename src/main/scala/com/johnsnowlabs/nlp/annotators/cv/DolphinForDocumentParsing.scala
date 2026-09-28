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

package com.johnsnowlabs.nlp.annotators.cv

import com.johnsnowlabs.ml.ai.Dolphin
import com.johnsnowlabs.ml.ai.util.DolphinUtils
import com.johnsnowlabs.ml.ai.util.DolphinUtils.{LayoutElement, MappedBox, Release}
import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.onnx.OnnxWrapper.EncoderDecoderWrappers
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadJsonStringAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.ONNX
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, IMAGE}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.cv.feature_extractor.Preprocessor
import com.johnsnowlabs.nlp.annotators.cv.util.transform.DonutImageUtils
import com.johnsnowlabs.nlp.annotators.tokenizer.bpe.{BpeTokenizer, DolphinTokenizer}
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import com.johnsnowlabs.reader.util.HTMLParser
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, IntParam, Param}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import org.json4s.jackson.JsonMethods.parse
import org.json4s.{DefaultFormats, JValue}

import java.awt.image.BufferedImage
import scala.util.Try
import scala.util.control.NonFatal

/** Document image parsing with ByteDance Dolphin.
  *
  * Dolphin follows an analyze-then-parse design: stage 1 generates the page's layout elements in
  * reading order, then each element is cropped and parsed independently in stage 2, batched by
  * type. Tables come back as HTML, formulas as LaTeX, text as markdown.
  *
  * The task prompt is the decoder prefix; there is no chat template. Dolphin 1.5 and 1.0 load
  * through the same path. The release is recognised from its stage-1 output, which decides the
  * box geometry and whether formulas and code get their own prompts (1.5 only).
  *
  * ==Example==
  * {{{
  * val imageAssembler = new ImageAssembler()
  *   .setInputCol("image")
  *   .setOutputCol("image_assembler")
  *
  * val dolphin = DolphinForDocumentParsing.pretrained()
  *   .setInputCols("image_assembler")
  *   .setOutputCol("elements")
  *   .setParsingMode("page")
  *
  * new Pipeline().setStages(Array(imageAssembler, dolphin))
  * }}}
  *
  * @groupname param Parameters
  */
class DolphinForDocumentParsing(override val uid: String)
    extends AnnotatorModel[DolphinForDocumentParsing]
    with HasBatchedAnnotateImage[DolphinForDocumentParsing]
    with HasImageFeatureProperties
    with HasRescaleFactor
    with WriteOnnxModel
    with HasEngine {

  def this() = this(Identifiable.randomUID("DolphinForDocumentParsing"))

  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(IMAGE)
  override val outputAnnotatorType: AnnotatorType = DOCUMENT

  /** What the input image is:
    *   - `page` (default): a whole page. Stage 1 + stage 2.
    *   - `layout`: a whole page; stage 1 only.
    *   - `table` / `text` / `formula` / `code`: an already cropped region; stage 2 only.
    *     `formula` and `code` are Dolphin 1.5 prompts.
    *
    * @group param
    */
  val parsingMode: Param[String] =
    new Param[String](this, "parsingMode", "One of: page, layout, table, text, formula, code")

  /** @group param */
  val outputFormat: Param[String] =
    new Param[String](this, "outputFormat", "One of: elements, markdown")

  /** Stage-2 crops decoded per forward pass.
    *
    * Measured optimum is 4: aggregate throughput at batch 4 matched or beat batch 16 on every
    * task, while the KV cache at batch 16 with a 4096 window needs over 10 GB. The reference's
    * `max_batch_size=16` is counterproductive on CPU, not merely memory-hungry.
    *
    * @group param
    */
  val elementBatchSize: IntParam =
    new IntParam(this, "elementBatchSize", "Element crops decoded per forward pass")

  /** Generation stops at the decoder's 4096-position window, prompt included, whatever the cap.
    *
    * @group param
    */
  val maxOutputLength: IntParam =
    new IntParam(this, "maxOutputLength", "Maximum tokens generated per element")

  /** @group param */
  val layoutMaxOutputLength: IntParam =
    new IntParam(this, "layoutMaxOutputLength", "Maximum tokens generated for the layout pass")

  /** 1.0 (off) matches Dolphin 1.5's reference pipeline; Dolphin 1.0's used 1.1.
    *
    * @group param
    */
  val repetitionPenalty: DoubleParam =
    new DoubleParam(this, "repetitionPenalty", "Repetition penalty; 1.0 disables it")

  /** @group param */
  val customPrompt: Param[String] =
    new Param[String](
      this,
      "customPrompt",
      "Overrides the prompt in the table, text, formula and code modes")

  /** Dolphin 1.0 only: its boxes are widened off ink as its reference does. 1.5 boxes are used as
    * generated.
    *
    * @group param
    */
  val adjustBoxEdges: BooleanParam =
    new BooleanParam(
      this,
      "adjustBoxEdges",
      "Widen Dolphin 1.0 boxes so edges do not cut through ink")

  /** Page mode drops elements whose crop is this small, as upstream does.
    *
    * @group param
    */
  val minElementSize: IntParam =
    new IntParam(
      this,
      "minElementSize",
      "Drop elements whose crop width or height is this many pixels or fewer")

  /** @group setParam */
  def setParsingMode(value: String): this.type = {
    require(
      DolphinForDocumentParsing.ParsingModes.contains(value),
      s"parsingMode must be one of ${DolphinForDocumentParsing.ParsingModes.mkString(", ")}")
    set(parsingMode, value)
  }

  /** @group setParam */
  def setOutputFormat(value: String): this.type = {
    require(
      DolphinForDocumentParsing.OutputFormats.contains(value),
      s"outputFormat must be one of ${DolphinForDocumentParsing.OutputFormats.mkString(", ")}")
    set(outputFormat, value)
  }

  /** @group setParam */
  def setElementBatchSize(value: Int): this.type = {
    require(value > 0, "elementBatchSize must be positive")
    set(elementBatchSize, value)
  }

  /** @group setParam */
  def setMaxOutputLength(value: Int): this.type = {
    require(value > 0, "maxOutputLength must be positive")
    set(maxOutputLength, value)
  }

  /** @group setParam */
  def setLayoutMaxOutputLength(value: Int): this.type = set(layoutMaxOutputLength, value)

  /** @group setParam */
  def setRepetitionPenalty(value: Double): this.type = set(repetitionPenalty, value)

  /** @group setParam */
  def setCustomPrompt(value: String): this.type = set(customPrompt, value)

  /** @group setParam */
  def setAdjustBoxEdges(value: Boolean): this.type = set(adjustBoxEdges, value)

  /** @group setParam */
  def setMinElementSize(value: Int): this.type = set(minElementSize, value)

  /** @group getParam */
  def getParsingMode: String = $(parsingMode)

  /** @group getParam */
  def getOutputFormat: String = $(outputFormat)

  /** @group getParam */
  def getElementBatchSize: Int = $(elementBatchSize)

  /** @group getParam */
  def getMaxOutputLength: Int = $(maxOutputLength)

  /** @group getParam */
  def getLayoutMaxOutputLength: Int = $(layoutMaxOutputLength)

  /** @group getParam */
  def getRepetitionPenalty: Double = $(repetitionPenalty)

  /** @group getParam */
  def getCustomPrompt: String = $(customPrompt)

  /** @group getParam */
  def getAdjustBoxEdges: Boolean = $(adjustBoxEdges)

  /** @group getParam */
  def getMinElementSize: Int = $(minElementSize)

  setDefault(
    batchSize -> 1,
    parsingMode -> "page",
    outputFormat -> "elements",
    elementBatchSize -> 4,
    maxOutputLength -> Dolphin.MaxPositions,
    layoutMaxOutputLength -> Dolphin.MaxPositions,
    repetitionPenalty -> 1.0d,
    customPrompt -> "",
    adjustBoxEdges -> true,
    minElementSize -> 3,
    size -> 896,
    doNormalize -> true,
    doResize -> true,
    doRescale -> true,
    rescaleFactor -> 1 / 255.0d,
    resample -> 2,
    imageMean -> Array(0.485d, 0.456d, 0.406d),
    imageStd -> Array(0.229d, 0.224d, 0.225d),
    featureExtractorType -> "DonutFeatureExtractor")

  protected[nlp] val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary")

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  protected[nlp] val merges: MapFeature[(String, String), Int] = new MapFeature(this, "merges")

  /** @group setParam */
  def setMerges(value: Map[(String, String), Int]): this.type = set(merges, value)

  protected[nlp] val generationConfig: StructFeature[GenerationConfig] =
    new StructFeature[GenerationConfig](this, "generationConfig").setProtected()

  /** @group setParam */
  def setGenerationConfig(value: GenerationConfig): this.type = set(generationConfig, value)

  /** @group getParam */
  def getGenerationConfig: GenerationConfig = $$(generationConfig)

  private var _model: Option[Broadcast[Dolphin]] = None

  def setModelIfNotSet(
      spark: SparkSession,
      onnxWrappers: Option[EncoderDecoderWrappers],
      preprocessor: Preprocessor): this.type = {
    require(
      onnxWrappers.isDefined,
      "DolphinForDocumentParsing requires the ONNX encoder/decoder/decoder-with-past wrappers")
    if (_model.isEmpty) {
      val tokenizer = BpeTokenizer
        .forModel(
          "dolphin",
          merges = $$(merges),
          vocab = $$(vocabulary),
          specialTokens = Some(DolphinTokenizer.specialTokens($$(vocabulary))))
        .asInstanceOf[DolphinTokenizer]

      // Turn a silent accuracy regression into a loud failure at load: a wrong first prompt token
      // still produces plausible-looking output.
      DolphinTokenizer.GoldenPromptIds.foreach { case (prompt, expected) =>
        val actual = tokenizer.encodePrompt(prompt)
        require(
          actual sameElements expected,
          s"Dolphin prompt tokenization changed for '$prompt': " +
            s"got ${actual.mkString(",")}, expected ${expected.mkString(",")}")
      }

      _model = Some(
        spark.sparkContext.broadcast(
          new Dolphin(
            onnxWrappers = onnxWrappers.get,
            tokenizer = tokenizer,
            preprocessor = preprocessor,
            generationConfig = getGenerationConfig)))
    }
    this
  }

  def getModelIfNotSet: Dolphin = _model.get.value

  // ------------------------------------------------------------------ inference

  private def cropPrompt(mode: String): String = {
    val custom = $(customPrompt)
    if (custom.nonEmpty) custom
    else
      mode match {
        case "table" => DolphinTokenizer.TablePrompt
        case "formula" => DolphinTokenizer.FormulaPrompt
        case "code" => DolphinTokenizer.CodePrompt
        case _ => DolphinTokenizer.TextPrompt
      }
  }

  /** Stage-2 prompt for a page element. Dolphin 1.0 reads everything but tables as text. */
  private def elementPrompt(label: String, release: Release): String =
    (release, label.toLowerCase) match {
      case (_, "tab") => DolphinTokenizer.TablePrompt
      case (Release.V1_5, "equ") => DolphinTokenizer.FormulaPrompt
      case (Release.V1_5, "code") => DolphinTokenizer.CodePrompt
      case _ => DolphinTokenizer.TextPrompt
    }

  /** A batch never mixes prompts: the exported decoder has no attention-mask input, so every row
    * must have the same prompt length.
    */
  private def parseGrouped(
      model: Dolphin,
      crops: Seq[(Int, BufferedImage)],
      prompt: String): Map[Int, String] =
    crops
      .grouped($(elementBatchSize))
      .flatMap { group =>
        val texts =
          model.parse(group.map(_._2).toArray, prompt, $(maxOutputLength), $(repetitionPenalty))
        group.map(_._1).zip(texts)
      }
      .toMap

  private def baseMetadata(annot: AnnotationImage): Map[String, String] =
    Map(
      "origin" -> annot.origin,
      "width" -> annot.width.toString,
      "height" -> annot.height.toString,
      "nChannels" -> annot.nChannels.toString,
      "mode" -> annot.mode.toString) ++
      annot.metadata.get("pageNumber").map("pageNumber" -> _)

  private def elementAnnotation(
      label: String,
      text: String,
      readingOrder: Int,
      readingGroup: Int,
      normalized: Array[Double],
      box: Option[MappedBox],
      offset: Int,
      base: Map[String, String]): Annotation = {
    val tableJson =
      if (DolphinUtils.isTable(label))
        Try(HTMLParser.tableElementToJson(HTMLParser.parseFirstTableElement(text))).toOption
      else None

    val metadata = base ++ Map(
      "elementType" -> DolphinUtils.elementTypeFor(label),
      "dolphinLabel" -> label,
      "readingOrder" -> readingOrder.toString,
      "readingGroup" -> readingGroup.toString,
      "bboxNormalized" -> normalized.map(v => f"$v%.4f").mkString(",")) ++
      box.map(b =>
        "bbox" -> s"${b.original.x1},${b.original.y1},${b.original.x2},${b.original.y2}") ++
      tableJson.map("tableJson" -> _)

    Annotation(
      annotatorType = DOCUMENT,
      begin = offset,
      end = offset + math.max(text.length - 1, 0),
      result = text,
      metadata = metadata)
  }

  /** Turn located, parsed elements into output annotations.
    *
    * Package-private so the assembly — offsets, metadata, and the markdown branch — is testable
    * without loading 2.6 GB of ONNX weights.
    */
  private[cv] def assemble(
      ordered: Seq[(LayoutElement, MappedBox, String)],
      raw: String,
      base: Map[String, String]): Seq[Annotation] =
    if ($(outputFormat) == "markdown") {
      val markdown = DolphinUtils.toMarkdown(ordered.map { case (e, _, text) => (e.label, text) })
      Seq(
        Annotation(
          annotatorType = DOCUMENT,
          begin = 0,
          end = math.max(markdown.length - 1, 0),
          result = markdown,
          metadata = base ++ Map("elementCount" -> ordered.length.toString, "layoutRaw" -> raw)))
    } else {
      var offset = 0
      ordered.map { case (element, box, text) =>
        val annotation = elementAnnotation(
          element.label,
          text,
          element.readingOrder,
          element.readingGroup,
          element.bbox,
          Some(box),
          offset,
          base)
        offset += text.length + 1
        annotation
      }
    }

  private def processImage(annot: AnnotationImage): Seq[Annotation] = {
    val model = getModelIfNotSet
    val base = baseMetadata(annot)
    val image =
      DonutImageUtils.toRgbBuffered(annot.result, annot.width, annot.height, annot.nChannels)

    $(parsingMode) match {
      case "page" => parsePage(model, image, base)
      case "layout" => detectLayout(model, image, base)
      case mode =>
        val text = model
          .parse(Array(image), cropPrompt(mode), $(maxOutputLength), $(repetitionPenalty))
          .head
        val label = DolphinForDocumentParsing.CropLabels(mode)
        Seq(elementAnnotation(label, text, 0, 0, Array(0.0, 0.0, 1.0, 1.0), None, 0, base))
    }
  }

  private def stageOne(model: Dolphin, image: BufferedImage): (String, DolphinUtils.Layout) = {
    val raw = model
      .parse(
        Array(image),
        DolphinTokenizer.LayoutPrompt,
        $(layoutMaxOutputLength),
        $(repetitionPenalty))
      .head
    (raw, DolphinUtils.parseLayout(raw))
  }

  private def adjustsEdges(release: Release): Boolean =
    release == Release.V1_0 && $(adjustBoxEdges)

  private def detectLayout(
      model: Dolphin,
      image: BufferedImage,
      base: Map[String, String]): Seq[Annotation] = {
    val (raw, layout) = stageOne(model, image)
    // only edge adjustment reads the padded pixels
    val (dims, binary) =
      if (adjustsEdges(layout.release)) {
        val (padded, paddedDims) = DonutImageUtils.padToSquare(image)
        (paddedDims, Some(DolphinUtils.binarizeOtsu(padded)))
      } else (DonutImageUtils.squareDims(image), None)

    var previous: Option[MappedBox] = None
    layout.elements.map { element =>
      val box = DolphinUtils.locate(element.bbox, layout.release, dims, binary, previous)
      previous = Some(box)
      elementAnnotation(
        element.label,
        "",
        element.readingOrder,
        element.readingGroup,
        element.bbox,
        Some(box),
        0,
        base + ("layoutRaw" -> raw))
    }
  }

  private def parsePage(
      model: Dolphin,
      image: BufferedImage,
      base: Map[String, String]): Seq[Annotation] = {
    val (raw, layout) = stageOne(model, image)
    // markdown mode emits one annotation per page, even for an empty page
    if (layout.elements.isEmpty) return assemble(Seq.empty, raw, base)

    val (padded, dims) = DonutImageUtils.padToSquare(image)
    val binary =
      if (adjustsEdges(layout.release)) Some(DolphinUtils.binarizeOtsu(padded)) else None

    var previous: Option[MappedBox] = None
    val located: Seq[(LayoutElement, MappedBox, BufferedImage)] = layout.elements.flatMap {
      element =>
        val box = DolphinUtils.locate(element.bbox, layout.release, dims, binary, previous)
        previous = Some(box)
        DolphinUtils.crop(padded, box.padded, $(minElementSize)).map(crop => (element, box, crop))
    }

    val parsed: Map[Int, String] = located.zipWithIndex
      .collect {
        case ((element, _, crop), index) if !DolphinUtils.isFigure(element.label) =>
          (elementPrompt(element.label, layout.release), (index, crop))
      }
      .groupBy(_._1)
      .flatMap { case (prompt, group) => parseGrouped(model, group.map(_._2), prompt) }

    val ordered = located.zipWithIndex.map { case ((element, box, _), index) =>
      (element, box, parsed.getOrElse(index, ""))
    }
    assemble(ordered, raw, base)
  }

  /** `batchProcess` zips rows with results, so this must return exactly one `Seq` per input row
    * and in the same order — a shorter result silently truncates the DataFrame.
    *
    * Note `AnnotationImage.text` is deliberately ignored: `Reader2Image` defaults its
    * `promptTemplate` to `qwen2vl-chat` and would otherwise inject chat markup that Dolphin would
    * happily treat as the task prompt. The task comes from `parsingMode`.
    */
  override def batchAnnotate(
      batchedAnnotations: Seq[Array[AnnotationImage]]): Seq[Seq[Annotation]] =
    batchedAnnotations.map { annotations =>
      annotations.filter(_.result.nonEmpty).toSeq.flatMap { annot =>
        Try(processImage(annot)).recover { case NonFatal(e) =>
          Seq(
            Annotation(
              annotatorType = DOCUMENT,
              begin = 0,
              end = 0,
              result = "",
              metadata = baseMetadata(annot) + ("exception" -> e.getMessage)))
        }.get
      }
    }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    getEngine match {
      case ONNX.name =>
        val wrappers = getModelIfNotSet.onnxWrappers
        writeOnnxModels(
          path,
          spark,
          Seq(
            (wrappers.encoder, DolphinForDocumentParsing.EncoderFile),
            (wrappers.decoder, DolphinForDocumentParsing.DecoderFile),
            (wrappers.decoderWithPast, DolphinForDocumentParsing.DecoderWithPastFile)),
          DolphinForDocumentParsing.suffix)
      case other => throw new Exception(notSupportedEngineError + s" Received: $other")
    }
  }
}

trait ReadablePretrainedDolphinModel
    extends ParamsAndFeaturesReadable[DolphinForDocumentParsing]
    with HasPretrained[DolphinForDocumentParsing] {

  override val defaultModelName: Some[String] = Some("dolphin_1_5")

  override def pretrained(): DolphinForDocumentParsing = super.pretrained()

  override def pretrained(name: String): DolphinForDocumentParsing = super.pretrained(name)

  override def pretrained(name: String, lang: String): DolphinForDocumentParsing =
    super.pretrained(name, lang)

  override def pretrained(
      name: String,
      lang: String,
      remoteLoc: String): DolphinForDocumentParsing =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadDolphinDLModel extends ReadOnnxModel {
  this: ParamsAndFeaturesReadable[DolphinForDocumentParsing] =>

  override val onnxFile: String = "dolphin_onnx"

  def readModel(instance: DolphinForDocumentParsing, path: String, spark: SparkSession): Unit = {

    val preprocessor = Preprocessor(
      do_normalize = instance.getDoNormalize,
      do_resize = instance.getDoResize,
      feature_extractor_type = "DonutFeatureExtractor",
      image_mean = instance.getImageMean,
      image_std = instance.getImageStd,
      resample = instance.getResample,
      do_rescale = instance.getDoRescale,
      rescale_factor = instance.getRescaleFactor,
      size = instance.getSize)

    instance.getEngine match {
      case ONNX.name =>
        val wrappers = readOnnxModels(
          path,
          spark,
          Seq(
            DolphinForDocumentParsing.EncoderFile,
            DolphinForDocumentParsing.DecoderFile,
            DolphinForDocumentParsing.DecoderWithPastFile),
          DolphinForDocumentParsing.suffix,
          dataFilePostfix = ".onnx_data")

        instance.setModelIfNotSet(
          spark,
          Some(
            EncoderDecoderWrappers(
              encoder = wrappers(DolphinForDocumentParsing.EncoderFile),
              decoder = wrappers(DolphinForDocumentParsing.DecoderFile),
              decoderWithPast = wrappers(DolphinForDocumentParsing.DecoderWithPastFile))),
          preprocessor)

      case other => throw new Exception(notSupportedEngineError + s" Received: $other")
    }
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): DolphinForDocumentParsing = {
    val (localModelPath, detectedEngine) =
      modelSanityCheck(modelPath, isEncoderDecoder = true, withPast = true)

    implicit val formats: DefaultFormats.type = DefaultFormats

    val preprocessorConfig =
      Preprocessor.loadPreprocessorConfig(
        loadJsonStringAsset(localModelPath, "preprocessor_config.json"))

    // Flags we model by construction rather than honour dynamically. Fail loudly if a future
    // checkpoint changes them instead of silently preprocessing it wrong.
    val rawPreprocessor: JValue =
      parse(loadJsonStringAsset(localModelPath, "preprocessor_config.json"))
    Seq(
      "do_align_long_axis" -> true,
      "do_thumbnail" -> true,
      "do_pad" -> true,
      "do_crop_margin" -> false)
      .foreach { case (flag, expected) =>
        (rawPreprocessor \ flag).extractOpt[Boolean].foreach { actual =>
          require(
            actual == expected,
            s"DolphinForDocumentParsing assumes $flag == $expected, but the checkpoint sets $actual")
        }
      }

    // config.json has no top-level token ids for a VisionEncoderDecoder -- they live under
    // `decoder`, and generation_config.json is the authoritative source.
    val modelConfig: JValue = parse(loadJsonStringAsset(localModelPath, "config.json"))
    val generationJson: JValue =
      Try(parse(loadJsonStringAsset(localModelPath, "generation_config.json")))
        .getOrElse(modelConfig \ "decoder")

    def tokenId(name: String, fallback: Int): Int =
      (generationJson \ name)
        .extractOpt[Int]
        .orElse((modelConfig \ "decoder" \ name).extractOpt[Int])
        .getOrElse(fallback)

    val vocabSize = (modelConfig \ "decoder" \ "vocab_size").extractOpt[Int].getOrElse(73921)
    (modelConfig \ "decoder" \ "max_position_embeddings").extractOpt[Int].foreach { positions =>
      require(
        positions == Dolphin.MaxPositions,
        s"DolphinForDocumentParsing assumes ${Dolphin.MaxPositions} decoder positions, " +
          s"but the checkpoint declares $positions")
    }

    val generationConfig = GenerationConfig(
      bosId = tokenId("bos_token_id", 0),
      padId = tokenId("pad_token_id", 1),
      eosId = tokenId("eos_token_id", 2),
      vocabSize = vocabSize,
      beginSuppressTokens = None,
      suppressTokenIds = None,
      forcedDecoderIds = None)

    // Dolphin ships a fast tokenizer: 50,000 BPE pieces plus 23,944 added tokens (CJK characters,
    // [TMP_n] markers, and the HTML table structure tokens). Merged they are exactly 73,921.
    val tokenizerJson: JValue = parse(loadJsonStringAsset(localModelPath, "tokenizer.json"))
    val baseVocab = (tokenizerJson \ "model" \ "vocab").extract[Map[String, Int]]
    val addedTokens = (tokenizerJson \ "added_tokens")
      .extract[List[Map[String, Any]]]
      .map(entry =>
        entry("content").asInstanceOf[String] -> entry("id").asInstanceOf[BigInt].intValue)
      .toMap
    val vocab = baseVocab ++ addedTokens
    require(
      vocab.size == vocabSize,
      s"tokenizer.json yielded ${vocab.size} entries but the decoder declares vocab_size $vocabSize")

    val bytePairs = (tokenizerJson \ "model" \ "merges")
      .extract[List[List[String]]]
      .filter(_.length == 2)
      .map(pair => (pair.head, pair(1)))
      .zipWithIndex
      .toMap

    val annotatorModel = new DolphinForDocumentParsing()
      .setVocabulary(vocab)
      .setMerges(bytePairs)
      .setGenerationConfig(generationConfig)
      .setDoNormalize(preprocessorConfig.do_normalize)
      .setDoResize(preprocessorConfig.do_resize)
      .setFeatureExtractorType(preprocessorConfig.feature_extractor_type)
      .setImageMean(preprocessorConfig.image_mean)
      .setImageStd(preprocessorConfig.image_std)
      .setResample(preprocessorConfig.resample)
      .setSize(preprocessorConfig.size)
      .setDoRescale(preprocessorConfig.do_rescale)
      .setRescaleFactor(preprocessorConfig.rescale_factor)

    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case ONNX.name =>
        val encoder = OnnxWrapper.read(
          spark,
          localModelPath,
          zipped = false,
          useBundle = true,
          modelName = "encoder_model")
        val decoder = OnnxWrapper.read(
          spark,
          localModelPath,
          zipped = false,
          useBundle = true,
          modelName = "decoder_model")
        val decoderWithPast = OnnxWrapper.read(
          spark,
          localModelPath,
          zipped = false,
          useBundle = true,
          modelName = "decoder_with_past_model")

        annotatorModel.setModelIfNotSet(
          spark,
          Some(EncoderDecoderWrappers(encoder, decoder, decoderWithPast)),
          preprocessorConfig)

      case other => throw new Exception(notSupportedEngineError + s" Received: $other")
    }

    annotatorModel
  }
}

object DolphinForDocumentParsing extends ReadablePretrainedDolphinModel with ReadDolphinDLModel {

  val suffix: String = "_dolphin"

  val EncoderFile: String = "encoder_model.onnx"
  val DecoderFile: String = "decoder_model.onnx"
  val DecoderWithPastFile: String = "decoder_with_past_model.onnx"

  /** Single-crop modes and the layout label their annotation carries. */
  val CropLabels: Map[String, String] =
    Map("table" -> "tab", "text" -> "text", "formula" -> "equ", "code" -> "code")

  val ParsingModes: Set[String] = Set("page", "layout") ++ CropLabels.keySet
  val OutputFormats: Set[String] = Set("elements", "markdown")
}
