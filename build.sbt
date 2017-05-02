
// Dependency settings
val scalaLangVersion = "2.11.10"
val sparkVersion = "2.1.0"
val junitVersion = "4.12"
val scalaTestVersion = "3.0.0"
val opennlpVersion = "1.6.0"

lazy val analyticsDependencies = Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
  "org.apache.opennlp" % "opennlp-tools" % opennlpVersion
)

lazy val testDependencies = Seq(
  "junit" % "junit" % junitVersion,
  "org.scalatest" %% "scalatest" % scalaTestVersion
)

lazy val utilDependencies = Seq(
  "com.typesafe" % "config" % "1.3.0"
)

scalacOptions ++= Seq(
  // See other posts in the series for other helpful options
  "-feature",
  "-language:implicitConversions"
)

javaOptions in test += "-Xmx4G"
javaOptions in run += "-Xmx4G"

// Module setup
lazy val root = (project in file("."))
  .settings(
    name := "sparknlp",
    version := "1.0.0",
    organization := "com.jsl.nlp",
    scalaVersion := scalaLangVersion,
    libraryDependencies ++=
      analyticsDependencies ++
      testDependencies ++
      utilDependencies
  )



