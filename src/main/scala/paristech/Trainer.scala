package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{CountVectorizer, IDF, OneHotEncoderEstimator, RegexTokenizer, StringIndexer}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.feature.StopWordsRemover

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

import java.io.File


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Trainer")
      .getOrCreate()


    /** *****************************************************************************
      *
      * TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      * if problems with unimported modules => sbt plugins update
      *
      * *******************************************************************************/

    println("/////////////////////////////////////////////////////////////////////////////////////")
    println("//                    TP 3 : Machine learning avec Spark                           //")
    println("/////////////////////////////////////////////////////////////////////////////////////")




    // import du dataset donné pour le TP3

    val pathToData= "./data"

    val df: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .parquet(s"$pathToData/prepared_trainingset")

    println("Training Dataframe")
    df.show()

    // ////////////////////////////////////////////////////////////////////////////
    // 2. Utiliser les données textuelles
    // ////////////////////////////////////////////////////////////////////////////

    // Stage 1 : Separer les textes en mots avec tokenizer

    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")


    // Stage 2 : Retirer les stop wordscountTokens(col(

    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("filtered")

    //Stage 3 : Conversion en TF-IDF

    val cvModel = new CountVectorizer()
      .setInputCol("filtered")
      .setOutputCol("vectorized")

    // Stage 4 : Trouver la partie IDF

    val idf = new IDF()
      .setInputCol("vectorized")
      .setOutputCol("tfidf")

    ////////////////////////////////////////////////////////////////////////////
    // 3. Convertir les variables catégorielles en variables numériques
    ////////////////////////////////////////////////////////////////////////////

    // Stage 5 : convertir country2 en quantités numériques

    val indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .setHandleInvalid("skip")

    // Stage 6 : convertir currency2 en quantités numériques

    val indexer2 = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .setHandleInvalid("skip")

    // Stage 7 et 8 : One-Hot encoder ces deux catégories (OneHotEncoder est deprecated, on utilise OnHotEncoderEstimator)

    val encoder = new OneHotEncoderEstimator()
      .setInputCols(Array("country_indexed", "currency_indexed"))
      .setOutputCols(Array("country_onehot", "currency_onehot"))

    //////////////////////////////////////////////////////////////////////////////
    // 4. Mettre les données sous une forme utilisable par SparkML
    //////////////////////////////////////////////////////////////////////////////

    // Stage 9

    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_onehot", "currency_onehot"))
      .setOutputCol("features")

    println("OUTPUT FEATURES")

    // Stage 10 : créer/instancier le modèle de classification : choix de la régression logistique

    //df.show()
    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(50)


    //////////////////////////////////////////////////////////////////////////////
    // 5. Création du Pipeline
    //////////////////////////////////////////////////////////////////////////////

    val stages = Array(tokenizer, remover, cvModel, idf, indexer, indexer2, encoder, assembler, lr)
    val pipeline = new Pipeline().setStages(stages)

    //////////////////////////////////////////////////////////////////////////////
    // 6. Entraînement, test, et sauvegarde du modèle
    //////////////////////////////////////////////////////////////////////////////

    // Split des données en training (90%) et test (10%)

    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = 1984)

    //******************************************************************************
    // Modèle simple
    //******************************************************************************
    val trainModelSimple = pipeline.fit(training)
    val dfWithSimplePredictions = trainModelSimple.transform(test)
    println("Résultats de la prédiction avec le modèle simple : ")
    dfWithSimplePredictions.groupBy("final_status", "predictions").count.show()

    //Pour calculer le f1-score du modèle, on utilise l'évaluateur de la classification multiple
    val eval = new MulticlassClassificationEvaluator().setMetricName("f1")
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    val f1_score_simple = eval.evaluate(dfWithSimplePredictions)
    println(s"Le F1 score du modèle simple ====>  : $f1_score_simple")
    println("\n")


    //******************************************************************************
    // Modèle avec grille
    //******************************************************************************
    // Grid-search l'objectif est de trouver les meilleurs hyperparamètres pour le modèle

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(10e-8, 10e-6, 10e-4, 10e-2))
      .addGrid(cvModel.minDF, Array(55.0, 75.0, 95.0))
      .build()


    // Creation d'un évaluateur pour la classification non binaire

    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("f1")
      .setLabelCol("final_status")
      .setPredictionCol("predictions")

    //  TrainValidationSplit requiert un estimateur, un set d'estimateur ParamMaps, et un Evaluator.
    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7) // 70% des données seront utilisées pour le train et 30% pour la validation

    // Entrainement du modèle avec l'échantillon training
    println("Entrainement du modèle avec grille")

    val validationModel = trainValidationSplit.fit(training)

    // Evaluation du modèle avec l'échantillon de test

    val dfWithPredictions = validationModel
      .transform(test)
      .select("features","final_status","predictions")

    // affichage du f1 score

    println("Voici les prédictions")
    dfWithPredictions.show(5)
    val score=evaluator.evaluate(dfWithPredictions)

    dfWithPredictions.groupBy("final_status","predictions").count.show()

    println("F1 Score du modèle avec grille ====> " + score)

    // Evaluation du modèle via la metric "accuracy"

    val evaluator_acc = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("accuracy")

    // obtention de la mesure de performance

    val accuracy = evaluator_acc.evaluate(dfWithPredictions)
    println("Precision obtenue 'accuracy' : " + accuracy)

    // Sauvegarde du modèle

    validationModel.write.overwrite.save(s"$pathToData/LogisticRegression")

  }
}
