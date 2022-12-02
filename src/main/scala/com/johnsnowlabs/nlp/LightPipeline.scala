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

class LightPipeline(val pipelineModel: PipelineModel, parseEmbeddingsVectors: Boolean = false) {

  private var ignoreUnsupported = false

  def setIgnoreUnsupported(v: Boolean): Unit = ignoreUnsupported = v

  def getIgnoreUnsupported: Boolean = ignoreUnsupported

  def getStages: Array[Transformer] = pipelineModel.stages

  def transform(dataFrame: Dataset[_]): DataFrame = pipelineModel.transform(dataFrame)

  def fullAnnotate(targets: Array[String]): Array[Map[String, Seq[IAnnotation]]] = {
    targets.par
      .map(target => fullAnnotate(target))
      .toArray
  }

  def fullAnnotate(target: String, optionalTarget: String = ""): Map[String, Seq[IAnnotation]] = {
    if (target.contains("/") && ResourceHelper.validFile(target)) {
      fullAnnotateImage(target)
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
      targets.par.map(target => fullAnnotateImage(target)).toArray
    } else {
      (targets zip optionalTargets).par.map { case (target, optionalTarget) =>
        fullAnnotate(target, optionalTarget)
      }.toArray
    }
  }

  def fullAnnotateImage(pathToImages: Array[String]): Array[Map[String, Seq[IAnnotation]]] = {
    pathToImages.par
      .map(imageFilePath => fullAnnotateInternal(imageFilePath))
      .toArray
  }

  def fullAnnotateImage(pathToImage: String): Map[String, Seq[IAnnotation]] = {
    fullAnnotateInternal(pathToImage)
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

  private def fullAnnotateInternal(
      target: String,
      optionalTarget: String = "",
      audio: Array[Float] = Array.empty,
      startWith: Map[String, Seq[IAnnotation]] = Map.empty[String, Seq[IAnnotation]])
      : Map[String, Seq[IAnnotation]] = {
    getStages.foldLeft(startWith)((annotations, transformer) => {
      transformer match {
        case documentAssembler: DocumentAssembler =>
          processDocumentAssembler(documentAssembler, target, annotations)
        case multiDocumentAssembler: MultiDocumentAssembler =>
          processMultipleDocumentAssembler(
            multiDocumentAssembler,
            target,
            optionalTarget,
            annotations)
        case imageAssembler: ImageAssembler =>
          processImageAssembler(target, imageAssembler, annotations)
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
          new LightPipeline(pipeline, parseEmbeddingsVectors)
            .fullAnnotateInternal(target, optionalTarget, audio, annotations)
        case _ => annotations
      }
    })
  }

  private def processDocumentAssembler(
      documentAssembler: DocumentAssembler,
      target: String,
      annotations: Map[String, Seq[IAnnotation]]): Map[String, Seq[IAnnotation]] = {
    annotations.updated(
      documentAssembler.getOutputCol,
      documentAssembler.assemble(target, Map.empty[String, String]))
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
        multiDocumentAssembler.assemble(input, Map.empty[String, String]))
    }

    annotations ++ multiDocumentAnnotations
  }

  private def processImageAssembler(
      target: String,
      imageAssembler: ImageAssembler,
      annotations: Map[String, Seq[IAnnotation]]): Map[String, Seq[IAnnotation]] = {
    val currentImageFields = ImageIOUtils.imagePathToImageFields(target)
    annotations.updated(
      imageAssembler.getOutputCol,
      imageAssembler.assemble(currentImageFields, Map.empty[String, String]))
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

    annotations.updated(
      batchedAnnotatorImage.getOutputCol,
      batchedAnnotatorImage.batchAnnotate(batchedAnnotations).head)
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

    inputCols
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
        JavaAnnotation(
          annotation.annotatorType,
          annotation.begin,
          annotation.end,
          annotation.result,
          annotation.metadata.asJava)
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

  def fullAnnotateImageJava(pathToImages: java.util.ArrayList[String])
      : java.util.List[java.util.Map[String, java.util.List[IAnnotation]]] = {

    pathToImages.asScala.par
      .map { imageFilePath =>
        fullAnnotateInternal(imageFilePath).mapValues(_.asJava).asJava
      }
      .toList
      .asJava
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

  def annotate(targets: Array[String]): Array[Map[String, Seq[String]]] = {
    targets.par
      .map(target => annotate(target))
      .toArray
  }

  def annotate(target: String, optionalTarget: String = ""): Map[String, Seq[String]] = {
    fullAnnotate(target, optionalTarget).mapValues(_.map { iAnnotation =>
      val annotation = iAnnotation.asInstanceOf[Annotation]
      annotation.annotatorType match {
        case AnnotatorType.WORD_EMBEDDINGS | AnnotatorType.SENTENCE_EMBEDDINGS
            if parseEmbeddingsVectors =>
          annotation.embeddings.mkString(" ")
        case _ => annotation.result
      }
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

}
