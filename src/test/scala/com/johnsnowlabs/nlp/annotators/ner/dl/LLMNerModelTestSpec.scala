/*
 * Copyright 2017-2024 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.ner.dl

import com.johnsnowlabs.nlp.{Annotation, AnnotatorType}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.{FastTest, SlowTest}
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class LLMNerModelTestSpec extends AnyFlatSpec {

  it should "extract entities" taggedAs SlowTest in {
    import ResourceHelper.spark.implicits._

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

    // Define few-shot examples to guide the model with comprehensive entity coverage
    val medicalExamples = Array(
      (
        "Patient takes aspirin 81mg daily.",
        """{"extractions": [{"entity": "MEDICATION", "text": "aspirin"}, {"entity": "DOSAGE", "text": "81mg"}, {"entity": "FREQUENCY", "text": "daily"}]}"""),
      (
        "Dr. Michael Chen from Massachusetts General Hospital ordered a chest X-ray and prescribed lisinopril 10mg PO BID for patient Sarah Williams on March 15th, 2024.",
        """{"extractions": [{"entity": "PERSON", "text": "Dr. Michael Chen"}, {"entity": "ORGANIZATION", "text": "Massachusetts General Hospital"}, {"entity": "TEST", "text": "chest X-ray"}, {"entity": "MEDICATION", "text": "lisinopril"}, {"entity": "DOSAGE", "text": "10mg"}, {"entity": "FREQUENCY", "text": "BID"}, {"entity": "PERSON", "text": "Sarah Williams"}]}"""),
      (
        "The cardiology department at Johns Hopkins performed an echocardiogram revealing severe aortic stenosis in patient Robert Martinez, who was started on metoprolol 50mg twice daily.",
        """{"extractions": [{"entity": "LOCATION", "text": "cardiology department"}, {"entity": "ORGANIZATION", "text": "Johns Hopkins"}, {"entity": "TEST", "text": "echocardiogram"}, {"entity": "CONDITION", "text": "severe aortic stenosis"}, {"entity": "PERSON", "text": "Robert Martinez"}, {"entity": "MEDICATION", "text": "metoprolol"}, {"entity": "DOSAGE", "text": "50mg"}, {"entity": "FREQUENCY", "text": "twice daily"}]}"""),
      (
        "Patient Jennifer Thompson received vancomycin 1g IV Q12H at Cleveland Clinic for treatment of MRSA pneumonia diagnosed on January 20th, 2024.",
        """{"extractions": [{"entity": "PERSON", "text": "Jennifer Thompson"}, {"entity": "MEDICATION", "text": "vancomycin"}, {"entity": "DOSAGE", "text": "1g"}, {"entity": "FREQUENCY", "text": "Q12H"}, {"entity": "ORGANIZATION", "text": "Cleveland Clinic"}, {"entity": "CONDITION", "text": "MRSA pneumonia"}]}"""))

    val llmNer = LLMNerModel
      .pretrained("llm-ner-medical")
      .setInputCols("document")
      .setOutputCol("entities")
      .setFewShotExamples(medicalExamples)
      .setNPredict(600)
      .setNGpuLayers(99)
      .setTemperature(0.2f)
      .setNCtx(5120)
      .setTopK(40)
      .setTopP(0.9f)
      .setPenalizeNl(true)
      .setBatchSize(4)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, llmNer))

    val complexSentences = Seq(
      "Dr. Sarah Johnson from Stanford Medical Center prescribed 500mg amoxicillin PO TID to patient John Smith on January 15th, 2024 for acute bronchitis.",
      "The laboratory at Massachusetts General Hospital reported elevated troponin levels of 0.8 ng/mL at 14:30 on March 3rd indicating possible myocardial infarction.",
      "Patient Mary Williams received 10mg morphine IV push at 08:00 for post-operative pain management following her appendectomy at Cleveland Clinic on February 20, 2024.",
      "Dr. Robert Chen ordered a complete metabolic panel and chest X-ray for patient David Rodriguez at Johns Hopkins Hospital due to suspected pneumonia and dehydration.",
      "The oncology department at Memorial Sloan Kettering Cancer Center administered 75mg/m2 cisplatin IV infusion to patient Emily Davis on April 10th as part of her chemotherapy regimen.",
      "Patient Michael Brown was admitted to Mayo Clinic Emergency Department on December 25, 2023 at 23:45 with acute myocardial infarction and underwent emergency cardiac catheterization.",
      "Dr. Jennifer Lee from UCLA Medical Center prescribed metformin 1000mg PO BID and lisinopril 20mg PO daily for patient Thomas Anderson's type 2 diabetes and hypertension.",
      "The radiology department at New York-Presbyterian Hospital performed a contrast-enhanced CT scan of the abdomen and pelvis on patient Amanda White on May 5th at 16:00.",
      "Patient Christopher Martinez received insulin lispro 8 units subcutaneously before meals and insulin glargine 24 units at bedtime at Boston Children's Hospital for type 1 diabetes management.",
      "Dr. Patricia Garcia ordered warfarin 5mg PO daily with weekly INR monitoring for patient Daniel Wilson at Cleveland Clinic following his pulmonary embolism diagnosed on June 15, 2024.",
      "The cardiology team at Cedars-Sinai Medical Center performed a transesophageal echocardiogram on patient Jessica Taylor at 10:30 revealing severe mitral valve regurgitation.",
      "Patient Andrew Thompson received vancomycin 1g IV Q12H and ceftriaxone 2g IV daily at Northwestern Memorial Hospital for treatment of bacterial meningitis starting August 1st, 2024.",
      "Dr. Linda Martinez from Cleveland Clinic prescribed albuterol inhaler 2 puffs Q4H PRN and prednisone 40mg PO daily for 5 days to patient Rebecca Johnson for acute asthma exacerbation.")

    val data = complexSentences.toDF("text").coalesce(1)

    val pipelineModel = pipeline.fit(data)
    val result = pipelineModel.transform(data)

    assert(result.columns.contains("entities"))

    result.show(truncate = false)
  }
}
