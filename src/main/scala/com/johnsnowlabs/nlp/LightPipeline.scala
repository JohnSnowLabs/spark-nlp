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
import org.apache.spark.ml.{PipelineModel, Transformer}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.JavaConverters._

class LightPipeline(val pipelineModel: PipelineModel, parseEmbeddingsVectors: Boolean = false) {

  private var ignoreUnsupported = false

  def setIgnoreUnsupported(v: Boolean): Unit = ignoreUnsupported = v

  def getIgnoreUnsupported: Boolean = ignoreUnsupported

  def getStages: Array[Transformer] = pipelineModel.stages

  def transform(dataFrame: Dataset[_]): DataFrame = pipelineModel.transform(dataFrame)

  def fullAnnotate(targets: Array[String]): Array[Map[String, Seq[Annotation]]] = {
    targets.par
      .map(target => fullAnnotate(target))
      .toArray
  }

  def fullAnnotate(target: String, optionalTarget: String = ""): Map[String, Seq[Annotation]] = {
    fullAnnotateInternal(target, optionalTarget).mapValues(_.map(_.asInstanceOf[Annotation]))
  }

  def fullAnnotateImage(imagesFilePath: Array[String]): Array[Map[String, Seq[IAnnotation]]] = {
    imagesFilePath.par
      .map(imageFilePath => fullAnnotateInternal(imageFilePath))
      .toArray
  }

  def fullAnnotateImage(imageFilePath: String): Array[Map[String, Seq[IAnnotation]]] = {

    val files = ImageIOUtils.loadImages(imageFilePath)

    files.map { file =>
      val imageFields = ImageIOUtils.imageFileToImageFields(file)
      fullAnnotateInternal(file.getPath, "", Some(imageFields))
    }

  }

  private def fullAnnotateInternal(
      target: String,
      optionalTarget: String = "",
      imageFields: Option[ImageFields] = None,
      startWith: Map[String, Seq[IAnnotation]] = Map.empty[String, Seq[IAnnotation]])
      : Map[String, Seq[IAnnotation]] = {
    getStages.foldLeft(startWith)((annotations, transformer) => {
      transformer match {
        case documentAssembler: DocumentAssembler =>
          annotations.updated(
            documentAssembler.getOutputCol,
            documentAssembler.assemble(target, Map.empty[String, String]))
        case multiDocumentAssembler: MultiDocumentAssembler =>
          val multiDocumentAnnotations =
            getMultipleDocumentAnnotations(
              multiDocumentAssembler,
              target,
              optionalTarget,
              annotations)
          annotations ++ multiDocumentAnnotations
        case imageAssembler: ImageAssembler =>
          val currentImageFields =
            if (imageFields.isDefined) imageFields.get
            else ImageIOUtils.imagePathToImageFields(target)
          annotations.updated(
            imageAssembler.getOutputCol,
            imageAssembler.assemble(currentImageFields, Map.empty[String, String]))
        case lazyAnnotator: AnnotatorModel[_] if lazyAnnotator.getLazyAnnotator => annotations
        case recursiveAnnotator: HasRecursiveTransform[_] with AnnotatorModel[_] =>
          val combinedAnnotations =
            getCombinedAnnotations(recursiveAnnotator.getInputCols, annotations)
          annotations.updated(
            recursiveAnnotator.getOutputCol,
            recursiveAnnotator.annotate(
              combinedAnnotations.map(_.asInstanceOf[Annotation]),
              pipelineModel))
        case batchedAnnotator: AnnotatorModel[_] with HasBatchedAnnotate[_] =>
          val combinedAnnotations =
            getCombinedAnnotations(batchedAnnotator.getInputCols, annotations)
          val batchedAnnotations = Seq(combinedAnnotations.map(_.asInstanceOf[Annotation]))
          // Benchmarks proved that parallel execution in LightPipeline gains more speed than batching entries (which require non parallel collections)
          annotations.updated(
            batchedAnnotator.getOutputCol,
            batchedAnnotator.batchAnnotate(batchedAnnotations).head)
        case batchedAnnotatorImage: AnnotatorModel[_] with HasBatchedAnnotateImage[_] =>
          val combinedAnnotations =
            getCombinedAnnotations(batchedAnnotatorImage.getInputCols, annotations)
          val batchedAnnotations = Seq(combinedAnnotations.map(_.asInstanceOf[AnnotationImage]))
          annotations.updated(
            batchedAnnotatorImage.getOutputCol,
            batchedAnnotatorImage.batchAnnotate(batchedAnnotations).head)
        case annotator: AnnotatorModel[_] with HasSimpleAnnotate[_] =>
          val inputCols = getAnnotatorInputCols(annotator)
          val combinedAnnotations = getCombinedAnnotations(inputCols, annotations)
          annotations.updated(
            annotator.getOutputCol,
            annotator.annotate(combinedAnnotations.map(_.asInstanceOf[Annotation])))
        case finisher: Finisher =>
          annotations.filterKeys(finisher.getInputCols.contains)
        case graphFinisher: GraphFinisher =>
          val annotated = getGraphFinisherOutput(annotations, graphFinisher)
          annotations.updated(graphFinisher.getOutputCol, annotated)
        case rawModel: RawAnnotator[_] =>
          if (ignoreUnsupported) annotations
          else
            throw new IllegalArgumentException(
              s"model ${rawModel.uid} does not support LightPipeline." +
                s" Call setIgnoreUnsupported(boolean) on LightPipeline to ignore")
        case pipeline: PipelineModel =>
          new LightPipeline(pipeline, parseEmbeddingsVectors)
            .fullAnnotateInternal(target, optionalTarget, None, annotations)
        case _ => annotations
      }
    })
  }

  private def getMultipleDocumentAnnotations(
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

    multiDocumentAnnotations
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
      val optionalColumns =
        getOptionalAnnotatorsOutputCols(annotator.optionalInputAnnotatorTypes)
      inputCols = inputCols ++ optionalColumns
    }

    inputCols
  }

  private def getOptionalAnnotatorsOutputCols(
      optionalInputAnnotatorTypes: Array[String]): Array[String] = {
    val optionalColumns = getStages
      .filter(stage => stage.isInstanceOf[AnnotatorModel[_]])
      .filter(stage =>
        optionalInputAnnotatorTypes.contains(
          stage.asInstanceOf[AnnotatorModel[_]].outputAnnotatorType))
      .map(stage => stage.asInstanceOf[AnnotatorModel[_]].getOutputCol)

    optionalColumns
  }

  private def getGraphFinisherOutput(
      annotations: Map[String, Seq[IAnnotation]],
      graphFinisher: GraphFinisher): Seq[Annotation] = {
    val result = getStages
      .filter(stage => stage.isInstanceOf[GraphFinisher])
      .map(stage => stage.asInstanceOf[GraphFinisher].getInputCol)
    val metadata = annotations
      .filter(annotation => result.contains(annotation._1))
      .flatMap(annotation => annotation._2.flatMap(a => a.asInstanceOf[Annotation].metadata))

    graphFinisher.annotate(metadata)
  }

  def fullAnnotateJava(target: String): java.util.Map[String, java.util.List[JavaAnnotation]] = {
    fullAnnotate(target)
      .mapValues(_.map { annotation =>
        JavaAnnotation(
          annotation.annotatorType,
          annotation.begin,
          annotation.end,
          annotation.result,
          annotation.metadata.asJava)
      }.asJava)
      .asJava
  }

  def fullAnnotateJava(
      target: String,
      optionalTarget: String): java.util.Map[String, java.util.List[JavaAnnotation]] = {
    fullAnnotate(target, optionalTarget)
      .mapValues(_.map { annotation =>
        JavaAnnotation(
          annotation.annotatorType,
          annotation.begin,
          annotation.end,
          annotation.result,
          annotation.metadata.asJava)
      }.asJava)
      .asJava
  }

  def fullAnnotateJava(targets: java.util.ArrayList[String])
      : java.util.List[java.util.Map[String, java.util.List[JavaAnnotation]]] = {
    targets.asScala.par
      .map(target => fullAnnotateJava(target))
      .toList
      .asJava
  }

  def fullAnnotateImageJava(imageFilePath: String)
      : java.util.List[java.util.Map[String, java.util.List[IAnnotation]]] = {

    val files = ImageIOUtils.loadImages(imageFilePath)

    files
      .map { file =>
        val imageFields = ImageIOUtils.imageFileToImageFields(file)
        fullAnnotateInternal(file.getPath, "", Some(imageFields))
          .mapValues(_.asJava)
          .asJava
      }
      .toList
      .asJava
  }

  def fullAnnotateImageJava(imagesFilePath: java.util.ArrayList[String])
      : java.util.List[java.util.Map[String, java.util.List[IAnnotation]]] = {

    imagesFilePath.asScala.par
      .map { imageFilePath =>
        fullAnnotateInternal(imageFilePath).mapValues(_.asJava).asJava
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
    fullAnnotate(target, optionalTarget).mapValues(_.map { annotation =>
      annotation.annotatorType match {
        case AnnotatorType.WORD_EMBEDDINGS | AnnotatorType.SENTENCE_EMBEDDINGS
            if parseEmbeddingsVectors =>
          annotation.embeddings.mkString(" ")
        case _ => annotation.result
      }
    })
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

}
