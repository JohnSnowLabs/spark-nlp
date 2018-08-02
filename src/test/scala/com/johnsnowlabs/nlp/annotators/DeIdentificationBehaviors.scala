package com.johnsnowlabs.nlp.annotators

import java.nio.file.{Files, Paths}

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.annotators.ner.dl.NerDLModel
import org.apache.spark.ml.util.MLWriter
import org.scalatest.FlatSpec

trait DeIdentificationBehaviors { this: FlatSpec =>

  def saveModel(model: MLWriter, modelFilePath: String): Unit = {
    //it should "save model on disk" in {
    model.overwrite().save(modelFilePath)
    assertResult(true){
        Files.exists(Paths.get(modelFilePath))
      }
    //}
  }

  case class TestParams(originalSentence: String,
                        tokenizedSentence: List[String],
                        annotations: Seq[Annotation]
                       )

  case class ExpectedParams(anonymizeSentence: String,
                            unclassifiedEntities: List[String],
                            anonymizeAnnotation: Annotation,
                            protectedEntities: Seq[Annotation])

  def deIdentificationAnnotator(deIdentification: DeIdentificationModel, testParams: TestParams,
                                expectedParams: ExpectedParams): Unit = {

    it should "get sentence from annotations" in {

      //Act
      val sentence = deIdentification.getSentence(testParams.annotations)

      //Assert
      assert(sentence == testParams.originalSentence)

    }

    it should "get protected entities" in {
      //Arrange
      val expectedProtectedEntities = expectedParams.protectedEntities.map(annotation => annotation.result).toList

      //Act
      val protectedEntities = deIdentification.getProtectedEntities(testParams.annotations)
          .map(annotation => annotation.result).toList

      //Assert
      assert(expectedProtectedEntities == protectedEntities)
    }

    /*it should "identified potential unclassified entities" in {
      //Act
      val unclassifiedEntities = deIdentification.getUnclassifiedEntities(testParams.tokenizedSentence)

      //Assert
      assert(unclassifiedEntities == expectedParams.unclassifiedEntities)

    }*/

    it should "anonymize sentence" in {
      //Arrange
      val protectedEntities = deIdentification.getProtectedEntities(testParams.annotations)

      //Act
      val anonymizeSentence = deIdentification.getAnonymizeSentence(testParams.originalSentence, protectedEntities)

      assert(expectedParams.anonymizeSentence == anonymizeSentence)

    }

    it should "create anonymize annotation" in {
      // Arrange
      val protectedEntities = deIdentification.getProtectedEntities(testParams.annotations)
      val anonymizeSentence = deIdentification.getAnonymizeSentence(testParams.originalSentence, protectedEntities)

      // Act
      val anonymizeAnnotator = deIdentification.createAnonymizeAnnotation(anonymizeSentence)

      //Assert
      assert(expectedParams.anonymizeAnnotation.result == anonymizeAnnotator.result)
      assert(expectedParams.anonymizeAnnotation.annotatorType == anonymizeAnnotator.annotatorType)

    }

  }

}
