package com.johnsnowlabs.nlp

import org.apache.spark.ml.{PipelineModel, Transformer}
import scala.collection.JavaConverters._

class SparklessPipeline(stages: Array[Transformer]) {

  def this(pipelineModel: PipelineModel) = this(pipelineModel.stages)

  def annotate(target: String): Map[String, Seq[Annotation]] = {
    stages.foldLeft(Map.empty[String, Seq[Annotation]])((annotations, transformer) => {
      transformer match {
        case documentAssembler: DocumentAssembler =>
          annotations.updated(documentAssembler.getOutputCol, documentAssembler.assemble(target, Map.empty[String, String]))
        case annotator: AnnotatorModel[_] =>
          val combinedAnnotations =
            annotator.getInputCols.foldLeft(Seq.empty[Annotation])((inputs, name) => inputs ++ annotations(name))
          annotations.updated(annotator.getOutputCol, annotator.annotate(combinedAnnotations))
        case finisher: Finisher =>
          annotations.filterKeys(finisher.getInputCols.contains)
        case _ => annotations
      }
    })
  }

  def annotate(targets: Array[String]): Map[String, Seq[Annotation]] = {
    targets.flatMap(target => {
      annotate(target)
    }).toMap
  }

  def annotate(targets: java.util.ArrayList[String]): java.util.Map[String, java.util.List[JavaAnnotation]] = {
    targets.asScala.flatMap(target => {
      annotate(target)
    }).map{case (a,b) => (a,b.map(aa =>
      JavaAnnotation(aa.annotatorType, aa.start, aa.end, aa.result, aa.metadata.asJava)
    ).asJava)}.toMap.asJava
  }

  def parAnnotate(targets: Array[String]): Map[String, Seq[Annotation]] = {
    targets.par.flatMap(target => {
      annotate(target)
    }).toArray.toMap
  }

  def parAnnotate(targets: java.util.ArrayList[String]): java.util.Map[String, java.util.List[JavaAnnotation]] = {
    targets.asScala.par.flatMap(target => {
      annotate(target)
    }).map{case (a,b) => (a,b.map(aa =>
      JavaAnnotation(aa.annotatorType, aa.start, aa.end, aa.result, aa.metadata.asJava)
    ).asJava)}.toList.toMap.asJava
  }

}

object SparklessPipeline {
  implicit def pip2sparkless(pipelineModel: PipelineModel): SparklessPipeline = {
    new SparklessPipeline(pipelineModel)
  }
}
