import com.johnsnowlabs.nlp.SparkAccessor
import com.johnsnowlabs.pretrained.EnModelsTraining

// ToDo 1. Downlaad datasets before if needs
// ToDo 2. Create Uploader that merges metadata

object TrainingModelsScript extends App {

  val training = new EnModelsTraining(SparkAccessor.spark)
  training.trainAllAndSave("models", true)
}
