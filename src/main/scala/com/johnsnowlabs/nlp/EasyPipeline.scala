package com.johnsnowlabs.nlp

import org.apache.spark.ml.{PipelineModel, Transformer}
import scala.collection.JavaConverters._

class EasyPipeline(stages: Array[Transformer]) {

  def this(pipelineModel: PipelineModel) = this(pipelineModel.stages)

  def annotate(target: String): Map[String, Seq[Annotation]] = {
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
        case _ => annotations
      }
    })
  }

  def linAnnotate(targets: Array[String]): Array[Map[String, Seq[Annotation]]] = {
    targets.map(annotate)
  }

  def linAnnotate(targets: java.util.ArrayList[String]): java.util.List[java.util.Map[String, java.util.List[JavaAnnotation]]] = {
    targets.asScala.map(target => {
      annotate(target)
    }).map(_.map{case (a,b) => (a,b.map(aa =>
      JavaAnnotation(aa.annotatorType, aa.start, aa.end, aa.result, aa.metadata.asJava)
    ).asJava)}.asJava).asJava
  }

  def annotate(targets: Array[String]): Array[Map[String, Seq[Annotation]]] = {
    targets.par.map(target => {
      annotate(target)
    }).toArray
  }

  def annotate(targets: java.util.ArrayList[String]): java.util.List[java.util.Map[String, java.util.List[JavaAnnotation]]] = {
    targets.asScala.par.map(target => {
      annotate(target)
    }).map(_.map{case (a,b) => (a,b.map(aa =>
      JavaAnnotation(aa.annotatorType, aa.start, aa.end, aa.result, aa.metadata.asJava)
    ).asJava)}.toList.toMap.asJava).toList.asJava
  }

}

object EasyPipeline {
  implicit def pip2sparkless(pipelineModel: PipelineModel): EasyPipeline = {
    new EasyPipeline(pipelineModel)
  }
}
