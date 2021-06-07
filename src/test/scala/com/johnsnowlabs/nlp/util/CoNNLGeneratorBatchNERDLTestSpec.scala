package com.johnsnowlabs.nlp.util

import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.scalatest.FlatSpec

import java.nio.charset.StandardCharsets
import java.nio.file.{Files, Paths}


class CoNNLGeneratorBatchNERDLTestSpec extends FlatSpec{
  ResourceHelper.spark //for toDS and toDF

  def formatCoNNL(firstname: String,
                  lastname: String,
                  verb: String,
                  proposition: String,
                  company: String,
                  country: String) = {
    s"""
       |$firstname PER
       |$lastname PER
       |$verb O
       |$proposition O
       |$company ORG
       |$country LOC
       |. O
       |""".stripMargin
  }

  "The (dataframe, pipelinemodel, outputpath) generator" should "make the right CoNNL file" taggedAs SlowTest in {

    val firstnames =  Array("Liam","Olivia","Noah","Emma","Oliver","Ava","William","Sophia","Elijah","Isabella")
    val lastnames =  Array("Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez")
    val verbs =  Array("be","have","do","say","go","get","make","known","think","take")
    val propositions = Array("of","in","to","for","with","on","at","from","by","about")
    val company = Array("Walmart","Amazon.com","PetroChina","Apple","CVS","RoyalDutch","Berkshire","Google","FaceBook","Fiat")
    val country = Array("China","India","USA","Indonesia","Pakistan","Brazil","Nigeria","Bangladesh","Russia","Mexico")

    // Modify simulation index i for more data volume
    for(i <- 1 to 6){
      val nerTagsDatasetStr: Array[String] = {
        for (
          firstname <- firstnames.take(i);
          lastname <- lastnames.take(i);
          verb <- verbs.take(i);
          proposition <- propositions.take(i);
          company <- company.take(i);
          country <- country.take(i)
        ) yield formatCoNNL(firstname,lastname,verb, proposition, company, country)
      }

      val prefix = "-DOCSTART- O\n"
      val suffix = "\n"

      Files.write(
        Paths.get(s"./tmp_ner_fake_connl_$i.txt"),
        (prefix + nerTagsDatasetStr.mkString + suffix)
          .getBytes(StandardCharsets.UTF_8))
    }
  }
}