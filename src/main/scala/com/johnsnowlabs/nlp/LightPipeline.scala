package com.johnsnowlabs.nlp

import org.apache.spark.ml.{PipelineModel, Transformer}
import org.apache.spark.sql.{DataFrame, Dataset}

import scala.collection.JavaConverters._

class LightPipeline(val pipelineModel: PipelineModel, parseEmbeddingsVectors: Boolean = false) {

  private var ignoreUnsupported = false

  def setIgnoreUnsupported(v: Boolean): Unit = ignoreUnsupported = v
  def getIgnoreUnsupported: Boolean = ignoreUnsupported

  def getStages: Array[Transformer] = pipelineModel.stages

  def transform(dataFrame: Dataset[_]): DataFrame = pipelineModel.transform(dataFrame)

  def fullAnnotate(target: String, startWith: Map[String, Seq[Annotation]] = Map.empty[String, Seq[Annotation]]): Map[String, Seq[Annotation]] = {
    getStages.foldLeft(startWith)((annotations, transformer) => {
      transformer match {
        case documentAssembler: DocumentAssembler =>
          annotations.updated(documentAssembler.getOutputCol, documentAssembler.assemble(target, Map.empty[String, String]))
        case lazyAnnotator: AnnotatorModel[_] if lazyAnnotator.getLazyAnnotator => annotations
        case recursiveAnnotator: HasRecursiveTransform[_] with AnnotatorModel[_] =>
          val combinedAnnotations =
            recursiveAnnotator.getInputCols.foldLeft(Seq.empty[Annotation])((inputs, name) => inputs ++ annotations.getOrElse(name, Nil))
          annotations.updated(recursiveAnnotator.getOutputCol, recursiveAnnotator.annotate(combinedAnnotations, pipelineModel))
        case batchedAnnotator: AnnotatorModel[_] with WithBatchAnnotate[_] =>
          val combinedAnnotations = batchedAnnotator.getInputCols.foldLeft(Seq.empty[Annotation])((inputs, name) => inputs ++ annotations.getOrElse(name, Nil))
          // Benchmarks proved that parallel execution in LightPipeline gains more speed than batching entries (which require non parallel collections)
          annotations.updated(batchedAnnotator.getOutputCol, batchedAnnotator.batchAnnotate(Array(combinedAnnotations)).head)
        case annotator: AnnotatorModel[_] with WithAnnotate[_] =>
          val combinedAnnotations =
            annotator.getInputCols.foldLeft(Seq.empty[Annotation])((inputs, name) => inputs ++ annotations.getOrElse(name, Nil))
          annotations.updated(annotator.getOutputCol, annotator.annotate(combinedAnnotations))
        case finisher: Finisher =>
          annotations.filterKeys(finisher.getInputCols.contains)
        case rawModel: RawAnnotator[_] =>
          if (ignoreUnsupported) annotations
          else throw new IllegalArgumentException(s"model ${rawModel.uid} does not support LightPipeline." +
            s" Call setIgnoreUnsupported(boolean) on LightPipeline to ignore")
        case pipeline: PipelineModel =>
          new LightPipeline(pipeline, parseEmbeddingsVectors).fullAnnotate(target, annotations)
        case _ => annotations
      }
    })
  }

  def fullAnnotate(targets: Array[String]): Array[Map[String, Seq[Annotation]]] = {
    targets.par.map(target => {
      fullAnnotate(target)
    }).toArray
  }

  def fullAnnotateJava(target: String): java.util.Map[String, java.util.List[JavaAnnotation]] = {
    fullAnnotate(target).mapValues(_.map(aa =>
      JavaAnnotation(aa.annotatorType, aa.begin, aa.end, aa.result, aa.metadata.asJava)).asJava).asJava
  }

  def fullAnnotateJava(targets: java.util.ArrayList[String]): java.util.List[java.util.Map[String, java.util.List[JavaAnnotation]]] = {
    targets.asScala.par.map(target => {
      fullAnnotateJava(target)
    }).toList.asJava
  }

  def annotate(target: String): Map[String, Seq[String]] = {
    fullAnnotate(target).mapValues(_.map(a => {
      a.annotatorType match {
        case (AnnotatorType.WORD_EMBEDDINGS |
             AnnotatorType.SENTENCE_EMBEDDINGS) if (parseEmbeddingsVectors) =>  a.embeddings.mkString(" ")
        case _ => a.result
      }
    }))
  }

  def annotate(targets: Array[String]): Array[Map[String, Seq[String]]] = {
    targets.par.map(target => {
      annotate(target)
    }).toArray
  }

  def annotateJava(target: String): java.util.Map[String, java.util.List[String]] = {
    annotate(target).mapValues(_.asJava).asJava
  }

  def annotateJava(targets: java.util.ArrayList[String]): java.util.List[java.util.Map[String, java.util.List[String]]] = {
    targets.asScala.par.map(target => {
      annotateJava(target)
    }).toList.asJava
  }

}
