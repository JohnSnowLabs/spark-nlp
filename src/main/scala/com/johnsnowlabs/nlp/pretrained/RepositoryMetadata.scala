package com.johnsnowlabs.nlp.pretrained

import java.sql.Timestamp

/**
  * Describes state of repository
  * Repository could be any s3 folder that has metadata.json describing list of resources inside
  */
case class RepositoryMetadata
(
  // Path to repository metadata file
  metadataFile: String,
  // Path to repository folder
  repoFolder: String,
  // Aws file metadata.json version
  version: String,
  // Last time metadata was downloaded
  lastMetadataDownloaded: Timestamp,
  // List of all available resources in repository
  metadata: List[ResourceMetadata]
)