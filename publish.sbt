import xerial.sbt.Sonatype.sonatypeCentralHost

homepage := Some(url("https://sparknlp.org"))
organizationName := "John Snow Labs"
organizationHomepage := Some(url("https://www.johnsnowlabs.com/"))
description := "Spark NLP is an open-source text processing library for advanced natural language processing."

scmInfo := Some(
  ScmInfo(
    url("https://github.com/JohnSnowLabs/spark-nlp"),
    "scm:git@github.com:JohnSnowLabs/spark-nlp.git"))

licenses += "Apache-2.0" -> url("https://opensource.org/licenses/Apache-2.0")

// Maven Central publishing settings
credentials += Credentials(Path.userHome / ".sbt" / "sonatype_central_credentials")
// Remove all additional repository other than Maven Central from POM
ThisBuild / pomIncludeRepository := { _ => false }
ThisBuild / publishMavenStyle := true

// new setting for the Central Portal
ThisBuild / publishTo := {
  val centralSnapshots = "https://central.sonatype.com/repository/maven-snapshots/"
  if (isSnapshot.value) Some("central-snapshots" at centralSnapshots)
  else localStaging.value
}

// Use sonatype bundle instead
//publishTo := sonatypePublishToBundle.value

sonatypeProfileName := "com.johnsnowlabs.nlp"

sonatypeCredentialHost := sonatypeCentralHost

// Developers
(ThisBuild / developers) := List(
  Developer(
    id = "saifjsl",
    name = "Saif Addin",
    email = "saif@johnsnowlabs.com",
    url = url("https://github.com/saifjsl")),
  Developer(
    id = "maziyarpanahi",
    name = "Maziyar Panahi",
    email = "maziyar@johnsnowlabs.com",
    url = url("https://github.com/maziyarpanahi")),
  Developer(
    id = "albertoandreottiATgmail",
    name = "Alberto Andreotti",
    email = "alberto@pacific.ai",
    url = url("https://github.com/albertoandreottiATgmail")),
  Developer(
    id = "danilojsl",
    name = "Danilo Burbano",
    email = "danilo@johnsnowlabs.com",
    url = url("https://github.com/danilojsl")),
  Developer(
    id = "rohit13k",
    name = "Rohit Kumar",
    email = "rohit@johnsnowlabs.com",
    url = url("https://github.com/rohit13k")),
  Developer(
    id = "aleksei-ai",
    name = "Aleksei Alekseev",
    email = "aleksei@pacific.ai",
    url = url("https://github.com/aleksei-ai")),
  Developer(
    id = "showy",
    name = "Eduardo Muñoz",
    email = "eduardo@johnsnowlabs.com",
    url = url("https://github.com/showy")),
  Developer(
    id = "C-K-Loan",
    name = "Christian Kasim Loan",
    email = "christian@johnsnowlabs.com",
    url = url("https://github.com/C-K-Loan")),
  Developer(
    id = "wolliq",
    name = "Stefano Lori",
    email = "stefano@johnsnowlabs.com",
    url = url("https://github.com/wolliq")),
  Developer(
    id = "vankov",
    name = "Ivan Vankov",
    email = "ivan@johnsnowlabs.com",
    url = url("https://github.com/vankov")),
  Developer(
    id = "alinapetukhova",
    name = "Alina Petukhova",
    email = "alina@johnsnowlabs.com",
    url = url("https://github.com/alinapetukhova")),
  Developer(
    id = "DevinTDHa",
    name = "Devin Ha",
    email = "devin@johnsnowlabs.com",
    url = url("https://github.com/DevinTDHa")),
  Developer(
    id = "ahmedlone127",
    name = "Khawja Ahmed Lone",
    email = "lone@johnsnowlabs.com",
    url = url("https://github.com/ahmedlone127")))
