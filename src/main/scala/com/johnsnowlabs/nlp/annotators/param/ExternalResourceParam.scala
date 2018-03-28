package com.johnsnowlabs.nlp.annotators.param

import com.johnsnowlabs.nlp.serialization.SerializedExternalResource
import com.johnsnowlabs.nlp.util.io.ExternalResource
import org.apache.spark.ml.util.Identifiable

class ExternalResourceParam(identifiable: Identifiable, name: String, description: String)
  extends AnnotatorParam[ExternalResource, SerializedExternalResource](identifiable, name, description)
