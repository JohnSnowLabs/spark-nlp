package com.johnsnowlabs.nlp

import org.apache.spark.ml.{PipelineModel, Transformer}

import scala.collection.JavaConverters._

class LightPipeline(stages: Array[Transformer]) {

  private var ignoreUnsupported = false

  def this(pipelineModel: PipelineModel) = this(pipelineModel.stages)

  def setIgnoreUnsupported(v: Boolean): Unit = ignoreUnsupported = v
  def getIgnoreUnsupported: Boolean = ignoreUnsupported

  def fullAnnotate(target: String): Map[String, Seq[Annotation]] = {
    stages.foldLeft(Map.empty[String, Seq[Annotation]])((annotations, transformer) => {
      transformer match {
        case documentAssembler: DocumentAssembler =>
          annotations.updated(documentAssembler.getOutputCol, documentAssembler.assemble(target, Map.empty[String, String]))
        case annotator: AnnotatorModel[_] =>
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
          LightPipeline.pip2sparkless(pipeline).fullAnnotate(target)
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
    fullAnnotate(target).mapValues(_.map(_.result))
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

object LightPipeline {
  implicit def pip2sparkless(pipelineModel: PipelineModel): LightPipeline = {
    new LightPipeline(pipelineModel)
  }
}
