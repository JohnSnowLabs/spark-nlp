package com.johnsnowlabs.nlp.annotators

import org.apache.spark.ml.{Pipeline, PipelineModel}
import java.nio.file.{Files, Paths}

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.{NAMED_ENTITY, DOCUMENT}
import org.apache.spark.ml.util.MLWriter
import org.scalatest.FlatSpec

trait DeIdentificationBehaviors { this: FlatSpec =>

  def saveModel(model: MLWriter, modelFilePath: String): Unit = {
    it should "save model on disk" ignore {
      model.overwrite().save(modelFilePath)
      assertResult(true){
        Files.exists(Paths.get(modelFilePath))
      }
    }
  }

  val annotations: Seq[Annotation] = Seq(
    Annotation(NAMED_ENTITY, 0, 2, "I-PER", Map("word"->"Bob")),
    Annotation(NAMED_ENTITY, 4, 10, "O", Map("word"->"visited")),
    Annotation(NAMED_ENTITY, 12, 22, "I-LOC", Map("word"->"Switzerland")),
    Annotation(NAMED_ENTITY, 24, 24, "O", Map("word"->"a")),
    Annotation(NAMED_ENTITY, 26, 31, "O", Map("word"->"couple")),
    Annotation(NAMED_ENTITY, 33, 34, "O", Map("word"->"of")),
    Annotation(NAMED_ENTITY, 36, 40, "O", Map("word"->"years")),
    Annotation(NAMED_ENTITY, 42, 44, "O", Map("word"->"ago"))
  )

  def deIdentificationAnnotator(deIdentification: DeIdentification): Unit = {

    it should "get protected entities" in {
      //Arrange
      val expectedProtectedEntities = Seq(
        Annotation(NAMED_ENTITY, 0, 2, "I-PER", Map("word"->"Bob")),
        Annotation(NAMED_ENTITY, 12, 22, "I-LOC", Map("word"->"Switzerland"))
      ).map(annotation => annotation.result).toList

      //Act
      val protectedEntities = deIdentification.getProtectedEntities(annotations)
          .map(annotation => annotation.result).toList

      //Assert
      assert(expectedProtectedEntities == protectedEntities)
    }

    it should "anonymize sentence" in {
      //Arrange
      val expectedAnonymizeSentence = "PER visited LOC a couple of years ago"
      val protectedEntities = deIdentification.getProtectedEntities(annotations)

      //Act
      val anonymizeSentence = deIdentification.getAnonymizeSentence("Bob visited Switzerland a couple of years ago",
        protectedEntities)

      assert(expectedAnonymizeSentence == anonymizeSentence)

    }

    it should "create anonymize annotation" in {
      // Arrange
      val expectedAnonymizeAnnotations = Annotation(DOCUMENT, 0, 37, "PER visited LOC a couple of years ago",
        Map("sentence"->"protected"))

      val protectedEntities = deIdentification.getProtectedEntities(annotations)
      val anonymizeSentence = deIdentification.getAnonymizeSentence("Bob visited Switzerland a couple of years ago",
        protectedEntities)

      // Act
      val anonymizeAnnotator = deIdentification.createAnonymizeAnnotation(annotations,anonymizeSentence)

      //Assert
      assert(expectedAnonymizeAnnotations.result == anonymizeAnnotator.result)
      assert(expectedAnonymizeAnnotations.annotatorType == anonymizeAnnotator.annotatorType)

    }


  }

}
