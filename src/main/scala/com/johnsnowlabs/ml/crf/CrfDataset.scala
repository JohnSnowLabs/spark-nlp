package com.johnsnowlabs.ml.crf

case class CrfDataset
(
  instances: Seq[(InstanceLabels, Instance)],
  metadata: DatasetMetadata
)

case class InstanceLabels(labels: Seq[Int])
case class Instance(items: Seq[SparseArray])


