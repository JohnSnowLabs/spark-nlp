/*
 * Copyright 2017-2022 John Snow Labs
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

package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.annotators.cv.util.io.ImageIOUtils
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.{PipelineModel, Transformer}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.JavaConverters._
import scala.util.{Failure, Success, Try}

class LightPipeline(
    val pipelineModel: PipelineModel,
    parseEmbeddings: Boolean = false,
    val outputCols: Array[String] = Array.empty[String]) {

  def this(pipelineModel: PipelineModel, parseEmbeddings: Boolean) =
    this(pipelineModel, parseEmbeddings, Array.empty[String])

  def this(
      pipelineModel: PipelineModel,
      parseEmbeddings: Boolean,
      outputCols: java.util.List[String]) = {
    this(pipelineModel, parseEmbeddings, outputCols.asScala.toArray)
  }

  private var ignoreUnsupported = false

  def setIgnoreUnsupported(v: Boolean): Unit = ignoreUnsupported = v

  def getIgnoreUnsupported: Boolean = ignoreUnsupported

  def getStages: Array[Transformer] = pipelineModel.stages

  def transform(dataFrame: Dataset[_]): DataFrame = {
    val transformedDf = pipelineModel.transform(dataFrame)

    if (outputCols.nonEmpty) {

      val documentAssemblers = pipelineModel.stages.toList
        .filter(s => s.isInstanceOf[DocumentAssembler])
        .map(s => s.asInstanceOf[DocumentAssembler])

      val idColName = documentAssemblers.headOption match {
        case Some(docAssembler) if docAssembler.isDefined(docAssembler.idCol) =>
          docAssembler.getIdCol
        case _ =>
          "doc_id"
      }
      val mandatoryCols = Seq(idColName, "document") ++ outputCols

      val allCols = (dataFrame.columns ++ transformedDf.columns).distinct
      val existingCols = allCols.filter(c => mandatoryCols.contains(c))

      transformedDf.select(existingCols.head, existingCols.tail: _*)
    } else {
      transformedDf
    }
  }

  def fullAnnotate(targets: Array[String]): Array[Map[String, Seq[IAnnotation]]] = {
    targets.par
      .map(target => fullAnnotate(target))
      .toArray
  }

  def fullAnnotate(target: String, optionalTarget: String = ""): Map[String, Seq[IAnnotation]] = {
    if (target.contains("/") && ResourceHelper.validFile(target)) {
      fullAnnotateImage(target, optionalTarget)
    } else {
      fullAnnotateInternal(target, optionalTarget)
    }
  }

  def fullAnnotate(
      targets: Array[String],
      optionalTargets: Array[String]): Array[Map[String, Seq[IAnnotation]]] = {

    if (targets.length != optionalTargets.length) {
      throw new UnsupportedOperationException(
        "targets and optionalTargets must be of the same length")
    }

    if (targets.head.contains("/") && ResourceHelper.validFile(targets.head)) {
      fullAnnotateImages(targets, optionalTargets)
    } else {
      (targets zip optionalTargets).par.map { case (target, optionalTarget) =>
        fullAnnotate(target, optionalTarget)
      }.toArray
    }
  }

  def fullAnnotateImages(
      pathToImages: Array[String],
      texts: Array[String] = Array.empty): Array[Map[String, Seq[IAnnotation]]] = {
    val safeTexts = if (texts.isEmpty) Array.fill(pathToImages.length)("") else texts
    (pathToImages zip safeTexts).par.map { case (imageFilePath, text) =>
      fullAnnotateImage(imageFilePath, text)
    }.toArray
  }

  def fullAnnotateImage(pathToImage: String, text: String = ""): Map[String, Seq[IAnnotation]] = {
    if (!ResourceHelper.validFile(pathToImage)) {
      Map()
    } else fullAnnotateInternal(pathToImage, text)
  }

  def fullAnnotate(audio: Array[Double]): Map[String, Seq[IAnnotation]] = {
    // We need this since py4j converts python floats to java Doubles
    fullAnnotate(audio.map(_.toFloat))
  }

  def fullAnnotate(audio: Array[Float]): Map[String, Seq[IAnnotation]] = {
    fullAnnotateInternal(target = "", audio = audio)
  }

  def fullAnnotate(audios: Array[Array[Float]]): Array[Map[String, Seq[IAnnotation]]] = {
    audios.par.map(audio => fullAnnotate(audio)).toArray
  }

  def fullAnnotate(
      ids: Array[Int],
      texts: Array[String]): Array[Map[String, Seq[IAnnotation]]] = {

    require(ids.length == texts.length, "ids and texts must have the same length")

    (ids zip texts).par.map { case (id, text) =>
      fullAnnotateInternal(target = text, id = Some(id))
    }.toArray
  }

  private def fullAnnotateInternal(
      target: String,
      optionalTarget: String = "",
      audio: Array[Float] = Array.empty,
      id: Option[Int] = None,
      startWith: Map[String, Seq[IAnnotation]] = Map.empty[String, Seq[IAnnotation]])
      : Map[String, Seq[IAnnotation]] = {
    val annotations = getStages.foldLeft(startWith)((annotations, transformer) => {
      transformer match {
        case documentAssembler: DocumentAssembler =>
          processDocumentAssembler(documentAssembler, target, annotations, id)
        case multiDocumentAssembler: MultiDocumentAssembler =>
          processMultipleDocumentAssembler(
            multiDocumentAssembler,
            target,
            optionalTarget,
            annotations)
        case imageAssembler: ImageAssembler =>
          processImageAssembler(target, optionalTarget, imageAssembler, annotations)
        case audioAssembler: AudioAssembler =>
          processAudioAssembler(audio, audioAssembler, annotations)
        case lazyAnnotator: AnnotatorModel[_] if lazyAnnotator.getLazyAnnotator => annotations
        case recursiveAnnotator: HasRecursiveTransform[_] with AnnotatorModel[_] =>
          processRecursiveAnnotator(recursiveAnnotator, annotations)
        case annotatorModel: AnnotatorModel[_] =>
          processAnnotatorModel(annotatorModel, annotations)
        case finisher: Finisher => annotations.filterKeys(finisher.getInputCols.contains)
        case graphFinisher: GraphFinisher => processGraphFinisher(graphFinisher, annotations)
        case rawModel: RawAnnotator[_] => processRowAnnotator(rawModel, annotations)
        case pipeline: PipelineModel =>
          new LightPipeline(pipeline, parseEmbeddings, outputCols)
            .fullAnnotateInternal(target, optionalTarget, audio, id, annotations)
        case _ => annotations
      }
    })

    if (outputCols.nonEmpty) {
      val documentAssemblers = pipelineModel.stages.toList
        .filter(s => s.isInstanceOf[DocumentAssembler])
        .map(s => s.asInstanceOf[DocumentAssembler])

      val idColName = documentAssemblers.headOption match {
        case Some(docAssembler) if docAssembler.isDefined(docAssembler.idCol) =>
          docAssembler.getIdCol
        case _ =>
          "doc_id"
      }

      val filteredAnnotations = annotations.filter { case (colName, _) =>
        outputCols.contains(colName) ||
        colName.equalsIgnoreCase("document") ||
        colName.equalsIgnoreCase(idColName)
      }
      filteredAnnotations
    } else {
      annotations
    }

  }

  private def processDocumentAssembler(
      documentAssembler: DocumentAssembler,
      target: String,
      annotations: Map[String, Seq[IAnnotation]],
      id: Option[Int] = None): Map[String, Seq[IAnnotation]] = {

    val documentAnnots = documentAssembler.assemble(target, Map("sentence" -> "0"))

    val updatedDocumentAnnots = id match {
      case Some(docId) =>
        val idStr = docId.toString
        documentAnnots.map { ann =>
          ann.copy(metadata = ann.metadata + ("id" -> idStr))
        }
      case None =>
        documentAnnots
    }

    var result = annotations.updated(documentAssembler.getOutputCol, updatedDocumentAnnots)

    id.foreach { docId =>
      val idStr = docId.toString
      val idLength = idStr.length

      val idAnnotation = Annotation(
        annotatorType = AnnotatorType.DUMMY,
        begin = 0,
        end = idLength,
        result = idStr,
        metadata = Map("id" -> idStr))

      val idColName =
        if (documentAssembler.isDefined(documentAssembler.idCol))
          documentAssembler.getIdCol
        else
          "doc_id"

      result = result + (idColName -> Seq(idAnnotation))
    }

    result
  }

  private def processMultipleDocumentAssembler(
      multiDocumentAssembler: MultiDocumentAssembler,
      target: String,
      optionalTarget: String,
      annotations: Map[String, Seq[IAnnotation]]): Map[String, Seq[IAnnotation]] = {

    var multiDocumentAnnotations: Map[String, Seq[IAnnotation]] = Map()
    val output = multiDocumentAssembler.getOutputCols zip Array(target, optionalTarget)
    output.foreach { outputTuple =>
      val outputCol = outputTuple._1
      val input = outputTuple._2
      multiDocumentAnnotations = multiDocumentAnnotations ++ annotations.updated(
        outputCol,
        multiDocumentAssembler.assemble(input, Map("sentence" -> "0")))
    }

    annotations ++ multiDocumentAnnotations
  }

  private def processImageAssembler(
      target: String,
      text: String,
      imageAssembler: ImageAssembler,
      annotations: Map[String, Seq[IAnnotation]]): Map[String, Seq[IAnnotation]] = {
    val currentImageFields = ImageIOUtils.imagePathToImageFields(target)
    annotations.updated(
      imageAssembler.getOutputCol,
      imageAssembler.assemble(currentImageFields, Map.empty[String, String], Some(text)))
  }

  private def processAudioAssembler(
      audio: Array[Float],
      audioAssembler: AudioAssembler,
      annotations: Map[String, Seq[IAnnotation]]): Map[String, Seq[IAnnotation]] = {
    annotations.updated(
      audioAssembler.getOutputCol,
      audioAssembler.assemble(audio, Map.empty[String, String]))
  }

  private def processAnnotatorModel(
      annotatorModel: AnnotatorModel[_],
      annotations: Map[String, Seq[IAnnotation]]): Map[String, Seq[IAnnotation]] = {
    annotatorModel match {
      case annotator: HasSimpleAnnotate[_] =>
        processAnnotator(annotator, annotations)
      case batchedAnnotator: HasBatchedAnnotate[_] =>
        processBatchedAnnotator(batchedAnnotator, annotations)
      case batchedAnnotatorImage: HasBatchedAnnotateImage[_] =>
        processBatchedAnnotatorImage(batchedAnnotatorImage, annotations)
      case batchedAnnotatorAudio: HasBatchedAnnotateAudio[_] =>
        processBatchedAnnotatorAudio(batchedAnnotatorAudio, annotations)
    }
  }

  private def processBatchedAnnotator(
      batchedAnnotator: AnnotatorModel[_] with HasBatchedAnnotate[_],
      annotations: Map[String, Seq[IAnnotation]]): Map[String, Seq[IAnnotation]] = {
    val combinedAnnotations =
      getCombinedAnnotations(batchedAnnotator.getInputCols, annotations)
    val batchedAnnotations = Seq(combinedAnnotations.map(_.asInstanceOf[Annotation]))

    // Benchmarks proved that parallel execution in LightPipeline gains more speed than batching entries (which require non parallel collections)
    annotations.updated(
      batchedAnnotator.getOutputCol,
      batchedAnnotator.batchAnnotate(batchedAnnotations).head)
  }

  private def processBatchedAnnotatorImage(
      batchedAnnotatorImage: AnnotatorModel[_] with HasBatchedAnnotateImage[_],
      annotations: Map[String, Seq[IAnnotation]]): Map[String, Seq[IAnnotation]] = {
    val combinedAnnotations =
      getCombinedAnnotations(batchedAnnotatorImage.getInputCols, annotations)
    val batchedAnnotations = Seq(combinedAnnotations.map(_.asInstanceOf[AnnotationImage]))

    val outputCol = batchedAnnotatorImage.getOutputCol
    val annotateResult = batchedAnnotatorImage.batchAnnotate(batchedAnnotations)
    annotations.updated(outputCol, annotateResult.head)
  }

  private def processBatchedAnnotatorAudio(
      batchedAnnotateAudio: AnnotatorModel[_] with HasBatchedAnnotateAudio[_],
      annotations: Map[String, Seq[IAnnotation]]): Map[String, Seq[IAnnotation]] = {
    val combinedAnnotations =
      getCombinedAnnotations(batchedAnnotateAudio.getInputCols, annotations)
    val batchedAnnotations = Seq(combinedAnnotations.map(_.asInstanceOf[AnnotationAudio]))

    annotations.updated(
      batchedAnnotateAudio.getOutputCol,
      batchedAnnotateAudio.batchAnnotate(batchedAnnotations).head)
  }

  private def processAnnotator(
      annotator: AnnotatorModel[_] with HasSimpleAnnotate[_],
      annotations: Map[String, Seq[IAnnotation]]): Map[String, Seq[IAnnotation]] = {
    val inputCols = getAnnotatorInputCols(annotator)
    val combinedAnnotations = getCombinedAnnotations(inputCols, annotations)

    annotations.updated(
      annotator.getOutputCol,
      annotator.annotate(combinedAnnotations.map(_.asInstanceOf[Annotation])))
  }

  private def processRecursiveAnnotator(
      recursiveAnnotator: HasRecursiveTransform[_] with AnnotatorModel[_],
      annotations: Map[String, Seq[IAnnotation]]): Map[String, Seq[IAnnotation]] = {
    val combinedAnnotations =
      getCombinedAnnotations(recursiveAnnotator.getInputCols, annotations)

    annotations.updated(
      recursiveAnnotator.getOutputCol,
      recursiveAnnotator.annotate(
        combinedAnnotations.map(_.asInstanceOf[Annotation]),
        pipelineModel))
  }

  private def processGraphFinisher(
      graphFinisher: GraphFinisher,
      annotations: Map[String, Seq[IAnnotation]]): Map[String, Seq[IAnnotation]] = {
    val finisherStages = getStages
      .filter(stage => stage.isInstanceOf[GraphFinisher])
      .map(stage => stage.asInstanceOf[GraphFinisher].getInputCol)
    val metadata = annotations
      .filter(annotation => finisherStages.contains(annotation._1))
      .flatMap(annotation => annotation._2.flatMap(a => a.asInstanceOf[Annotation].metadata))

    val annotated = graphFinisher.annotate(metadata)
    annotations.updated(graphFinisher.getOutputCol, annotated)
  }

  private def processRowAnnotator(
      rawAnnotator: RawAnnotator[_],
      annotations: Map[String, Seq[IAnnotation]]): Map[String, Seq[IAnnotation]] = {
    if (ignoreUnsupported) annotations
    else
      throw new IllegalArgumentException(
        s"model ${rawAnnotator.uid} does not support LightPipeline." +
          s" Call setIgnoreUnsupported(boolean) on LightPipeline to ignore")
  }

  private def getCombinedAnnotations(
      inputCols: Array[String],
      annotations: Map[String, Seq[IAnnotation]]): Array[IAnnotation] = {
    val combinedAnnotations = inputCols.foldLeft(Array.empty[IAnnotation])((inputs, name) =>
      inputs ++ annotations.getOrElse(name, Nil))

    combinedAnnotations
  }

  private def getAnnotatorInputCols(annotator: AnnotatorModel[_]): Array[String] = {
    var inputCols = annotator.getInputCols
    if (annotator.optionalInputAnnotatorTypes.nonEmpty) {
      val optionalColumns = getStages
        .filter(stage => stage.isInstanceOf[AnnotatorModel[_]])
        .filter(stage =>
          annotator.optionalInputAnnotatorTypes.contains(
            stage.asInstanceOf[AnnotatorModel[_]].outputAnnotatorType))
        .map(stage => stage.asInstanceOf[AnnotatorModel[_]].getOutputCol)

      inputCols = inputCols ++ optionalColumns
    }

    inputCols.distinct
  }

  def fullAnnotateJava(target: String): java.util.Map[String, java.util.List[IAnnotation]] = {
    fullAnnotate(target)
      .mapValues(_.map { annotation =>
        castToJavaAnnotation(annotation)
      }.asJava)
      .asJava
  }

  def fullAnnotateJava(
      target: String,
      optionalTarget: String): java.util.Map[String, java.util.List[IAnnotation]] = {
    fullAnnotate(target, optionalTarget)
      .mapValues(_.map { annotation =>
        castToJavaAnnotation(annotation)
      }.asJava)
      .asJava
  }

  private def castToJavaAnnotation(annotation: IAnnotation): IAnnotation = {
    Try(annotation.asInstanceOf[Annotation]) match {
      case Success(annotation) => {
        var embeddings = Array.emptyFloatArray
        if (parseEmbeddings) {
          embeddings = annotation.embeddings
        }
        JavaAnnotation(
          annotation.annotatorType,
          annotation.begin,
          annotation.end,
          annotation.result,
          annotation.metadata.asJava,
          embeddings)
      }
      case Failure(_) => annotation
    }
  }

  def fullAnnotateJava(targets: java.util.ArrayList[String])
      : java.util.List[java.util.Map[String, java.util.List[IAnnotation]]] = {
    targets.asScala.par
      .map(target => fullAnnotateJava(target))
      .toList
      .asJava
  }

  def fullAnnotateJava(
      targets: java.util.ArrayList[String],
      optionalTargets: java.util.ArrayList[String])
      : java.util.List[java.util.Map[String, java.util.List[IAnnotation]]] = {
    (targets.asScala zip optionalTargets.asScala).par
      .map { case (target, optionalTarget) =>
        fullAnnotateJava(target, optionalTarget)
      }
      .toList
      .asJava
  }

  def fullAnnotateImageJava(
      pathToImage: String): java.util.Map[String, java.util.List[IAnnotation]] = {
    fullAnnotateImage(pathToImage).mapValues(_.asJava).asJava
  }

  def fullAnnotateImageJava(
      pathToImages: java.util.ArrayList[String],
      texts: java.util.ArrayList[String])
      : java.util.List[java.util.Map[String, java.util.List[IAnnotation]]] = {
    if (texts.isEmpty) {
      pathToImages.asScala.par
        .map { imageFilePath =>
          fullAnnotateInternal(imageFilePath).mapValues(_.asJava).asJava
        }
        .toList
        .asJava
    } else {

      if (pathToImages.size != texts.size) {
        throw new IllegalArgumentException(
          "pathToImages and texts must have the same number of elements.")
      }
      val imageTextPairs = pathToImages.asScala.zip(texts.asScala).par

      imageTextPairs
        .map { case (imageFilePath, text) =>
          fullAnnotateImage(imageFilePath, text).mapValues(_.asJava).asJava
        }
        .toList
        .asJava
    }
  }

  def fullAnnotateSingleAudioJava(
      audio: java.util.ArrayList[Double]): java.util.Map[String, java.util.List[IAnnotation]] = {
    fullAnnotate(audio.asScala.toArray).mapValues(_.asJava).asJava
  }

  def fullAnnotateAudiosJava(audios: java.util.ArrayList[java.util.ArrayList[Double]])
      : java.util.List[java.util.Map[String, java.util.List[IAnnotation]]] = {
    audios.asScala.par
      .map { audio =>
        fullAnnotate(audio.asScala.toArray).mapValues(_.asJava).asJava
      }
      .toList
      .asJava
  }

  def fullAnnotateWithIdsJava(
      ids: java.util.ArrayList[Integer],
      texts: java.util.ArrayList[String])
      : java.util.List[java.util.Map[String, java.util.List[IAnnotation]]] = {

    val scalaIds = ids.asScala.map(_.toInt).toArray
    val scalaTexts = texts.asScala.toArray
    fullAnnotate(scalaIds, scalaTexts)
      .map { annotations =>
        annotations.mapValues(_.asJava).asJava
      }
      .toList
      .asJava
  }

  def annotate(targets: Array[String]): Array[Map[String, Seq[String]]] = {
    targets.par
      .map(target => annotate(target))
      .toArray
  }

  def annotate(target: String, optionalTarget: String = ""): Map[String, Seq[String]] = {
    val annotations = fullAnnotate(target, optionalTarget)
    annotations.mapValues(_.map {
      case annotation: Annotation =>
        annotation.annotatorType match {
          case AnnotatorType.WORD_EMBEDDINGS | AnnotatorType.SENTENCE_EMBEDDINGS
              if parseEmbeddings =>
            annotation.embeddings.mkString(" ")
          case _ => annotation.result
        }
      case _ => ""
    })
  }

  def annotate(
      targets: Array[String],
      optionalTargets: Array[String]): Array[Map[String, Seq[String]]] = {

    if (targets.length != optionalTargets.length) {
      throw new UnsupportedOperationException(
        "targets and optionalTargets must be of the same length")
    }

    (targets zip optionalTargets).par.map { case (target, optionalTarget) =>
      annotate(target, optionalTarget)
    }.toArray
  }

  def annotate(ids: Array[Int], texts: Array[String]): Array[Map[String, Seq[String]]] = {

    require(ids.length == texts.length, "ids and texts must have the same length")

    val annotationsArray = fullAnnotate(ids, texts)

    annotationsArray.map { annotations =>
      annotations.map { case (colName, annots) =>
        colName -> annots.map {
          case annotation: Annotation =>
            annotation.annotatorType match {
              case AnnotatorType.WORD_EMBEDDINGS | AnnotatorType.SENTENCE_EMBEDDINGS
                  if parseEmbeddings =>
                annotation.embeddings.mkString(" ")
              case _ => annotation.result
            }
          case _ => ""
        }
      }
    }
  }

  def annotateJava(target: String): java.util.Map[String, java.util.List[String]] = {
    annotate(target).mapValues(_.asJava).asJava
  }

  def annotateJava(
      target: String,
      optionalTarget: String): java.util.Map[String, java.util.List[String]] = {
    annotate(target, optionalTarget).mapValues(_.asJava).asJava
  }

  def annotateJava(targets: java.util.ArrayList[String])
      : java.util.List[java.util.Map[String, java.util.List[String]]] = {
    targets.asScala.par
      .map(target => annotateJava(target))
      .toList
      .asJava
  }

  def annotateJava(
      targets: java.util.ArrayList[String],
      optionalTargets: java.util.ArrayList[String])
      : java.util.List[java.util.Map[String, java.util.List[String]]] = {
    (targets.asScala zip optionalTargets.asScala).par
      .map { case (target, optionalTarget) =>
        annotateJava(target, optionalTarget)
      }
      .toList
      .asJava
  }

  def annotateWithIdsJava(ids: java.util.ArrayList[Integer], texts: java.util.ArrayList[String])
      : java.util.List[java.util.Map[String, java.util.List[String]]] = {

    val scalaIds = ids.asScala.map(_.toInt).toArray
    val scalaTexts = texts.asScala.toArray

    annotate(scalaIds, scalaTexts)
      .map { results =>
        results.mapValues(_.asJava).asJava
      }
      .toList
      .asJava
  }

}
