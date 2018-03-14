package com.johnsnowlabs.nlp

import org.apache.spark.ml.{PipelineModel, Transformer}

class SparklessPipeline(stages: Array[Transformer]) {

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

  def parAnnotate(targets: Array[String]): Map[String, Seq[Annotation]] = {
    targets.par.flatMap(target => {
      annotate(target)
    }).toArray.toMap
  }

}

object SparklessPipeline {
  implicit def pip2sparkless(pipelineModel: PipelineModel): SparklessPipeline = {
    convertToSparkless(pipelineModel)
  }

  def convertToSparkless(pipelineModel: PipelineModel): SparklessPipeline = {
    require(pipelineModel.stages.exists(_.isInstanceOf[AnnotatorModel[_]]))
    new SparklessPipeline(pipelineModel.stages)
  }
}
