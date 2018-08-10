package com.johnsnowlabs.nlp.annotators

import java.nio.file.{Files, Paths}

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.annotators.anonymizer.DeIdentificationModel
import com.johnsnowlabs.nlp.annotators.common.IndexedToken
import org.apache.spark.ml.util.MLWriter
import org.scalatest.FlatSpec

trait DeIdentificationBehaviors { this: FlatSpec =>

  def saveModel(model: MLWriter, modelFilePath: String): Unit = {
    model.overwrite().save(modelFilePath)
    assertResult(true){
        Files.exists(Paths.get(modelFilePath))
      }
  }

  case class TestParams(originalSentence: String,
                        tokenizeSentence: Seq[IndexedToken],
                        annotations: Seq[Annotation]
                       )

  case class ExpectedParams(anonymizeSentence: String,
                            regexEntities: Seq[Annotation],
                            anonymizeAnnotation: Annotation,
                            nerEntities: Seq[Annotation],
                            mergedEntities: Seq[Annotation])

  def deIdentificationAnnotator(deIdentification: DeIdentificationModel, testParams: TestParams,
                                expectedParams: ExpectedParams): Unit = {

    it should "get sentence from annotations" in {

      //Act
      val sentence = deIdentification.getSentence(testParams.annotations)

      //Assert
      assert(sentence == testParams.originalSentence)

    }


    it should "get tokens from annotations" in {
      //Act
      val tokens = deIdentification.getTokens(testParams.annotations)

      //Assert
      assert(tokens == testParams.tokenizeSentence)

    }

    it should "get NER entities" in {
      //Arrange
      val expectedProtectedEntities = expectedParams.nerEntities.map(annotation => annotation.result).toList

      //Act
      val protectedEntities = deIdentification.getNerEntities(testParams.annotations)
          .map(annotation => annotation.result).toList

      //Assert
      assert(expectedProtectedEntities == protectedEntities)
    }

    it should "identified regex entities" in {
      //Act
      val regexEntities = deIdentification.getRegexEntities(testParams.tokenizeSentence)

      //Assert
      assert(regexEntities == expectedParams.regexEntities)

    }

    it should "merge ner and regex entities" in {
      //Act
      val entitiesMerged = deIdentification.mergeEntities(expectedParams.nerEntities, expectedParams.regexEntities)

      //Assert
      val expectedEntitiesMerged = expectedParams.mergedEntities.map(entity=>entity.result).toList
      val entitiesMergedAsList = entitiesMerged.map(entity=>entity.result).toList
      assert(entitiesMergedAsList==expectedEntitiesMerged)
    }

    it should "anonymize sentence" in {
      //Act
      val anonymizeSentence = deIdentification.getAnonymizeSentence(testParams.originalSentence,
        expectedParams.mergedEntities)

      assert(expectedParams.anonymizeSentence == anonymizeSentence)

    }

    it should "create anonymize annotation" in {
      // Arrange
      val anonymizeSentence = deIdentification.getAnonymizeSentence(testParams.originalSentence, expectedParams.mergedEntities)

      // Act
      val anonymizeAnnotator = deIdentification.createAnonymizeAnnotation(anonymizeSentence)

      //Assert
      assert(expectedParams.anonymizeAnnotation.result == anonymizeAnnotator.result)
      assert(expectedParams.anonymizeAnnotation.annotatorType == anonymizeAnnotator.annotatorType)

    }

  }

}
