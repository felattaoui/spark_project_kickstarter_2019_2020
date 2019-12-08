package paristech

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession

import org.apache.spark.sql.{ DataFrame, Row, SQLContext }

//import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.functions._

import java.nio.file.{Paths, Files}

object Preprocessor {

  def main(args: Array[String]): Unit = {

    // Des réglages optionnels du job spark. Les réglages par défaut fonctionnent très bien pour ce TP.
    // On vous donne un exemple de setting quand même
    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    // Initialisation du SparkSession qui est le point d'entrée vers Spark SQL (donne accès aux dataframes, aux RDD,
    // création de tables temporaires, etc., et donc aux mécanismes de distribution des calculs)
    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP Spark : Preprocessor")
      .getOrCreate()

    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 2
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    println("/////////////////////////////////////////////////////////////////////////////////////")
    println("//                    TP 2 : Projet de Pre-processing                              //")
    println("/////////////////////////////////////////////////////////////////////////////////////")


    // Chargement des données

    val pathToData= "./data"

    val df: DataFrame = spark
      .read
      .option("header", true) // utilise la première ligne du (des) fichier(s) comme header
      .option("inferSchema", "true") // pour inférer le type de chaque colonne (Int, String, etc.)
      .csv(s"$pathToData/train_clean.csv")


    println("Nombre de lignes et le nombre de colonnes dans le DataFrame :")
    println(s"Nombre de lignes : ${df.count}")
    println(s"Nombre de colonnes : ${df.columns.length}")

    println("\n")
    println("Extrait du DataFrame sous forme de tableau")
    df.show()
    println("Schéma du DataFrame, à savoir le nom de chaque colonne avec son type")
    df.printSchema()
    println("\n")


    // Assignons le type Int aux colonnes qui nous semblent contenir des entiers le dataframe de retour s'appelle dfcasted

    val dfCasted: DataFrame = df
      .withColumn("goal", $"goal".cast("Int"))
      .withColumn("deadline", $"deadline".cast("Int"))
      .withColumn("state_changed_at", $"state_changed_at".cast("Int"))
      .withColumn("created_at", $"created_at".cast("Int"))
      .withColumn("launched_at", $"launched_at".cast("Int"))
      .withColumn("backers_count", $"backers_count".cast("Int"))
      .withColumn("final_status", $"final_status".cast("Int"))


    println("Phase de cleaning des données")
    println("Affichage d'une description statistique des colonnes de type Int :")

    dfCasted
      .select("name", "backers_count", "final_status")
      .describe()
      .show

    // Clean des données
    // (J'ai juste retiré les .show après avoir compris de quoi il s'agissait pour plus de lisibilité sur la console)
    dfCasted.groupBy("disable_communication").count.orderBy($"count".desc)
    dfCasted.groupBy("country").count.orderBy($"count".desc)
    dfCasted.groupBy("currency").count.orderBy($"count".desc)
    dfCasted.select("deadline").dropDuplicates.show()
    dfCasted.groupBy("state_changed_at").count.orderBy($"count".desc)
    dfCasted.groupBy("backers_count").count.orderBy($"count".desc)
    dfCasted.select("goal", "final_status")
    dfCasted.groupBy("country", "currency").count.orderBy($"count".desc)


    // "Suppression de la colonne disable_communication qui ne contient majoritairement que des false, on crée un nouveau dataFrame : df2"
    val df2: DataFrame = dfCasted.drop("disable_communication")
    println("Affichage du dataframe après la phase de cleaning")

    // Pour enlever les données du futur on retire les colonnes backers_count et state_changed_at, on crée un nouveau dataFrame : dfNoFutur
    val dfNoFutur: DataFrame = df2.drop("backers_count", "state_changed_at")

    // Colonnes currency et country
    // Nous avons semble-t-il des inversions entre ces deux colonnes et du nettoyage à faire
    // Nous remarquons en particulier que lorsque country = \"False\" le 'country' à l'air d'être dans 'currency'

    df.filter($"country" === "False")
      .groupBy("currency")
      .count
      .orderBy($"count".desc)

    // Création de deux udfs nommées udf_country et udf_currency telles que :

    def cleanCountry(country: String, currency: String): String = {
      if (country == "False")
        currency
      else if (country.length !=2)
        null
      else
        country
    }

    def cleanCurrency(currency: String): String = {
      if (currency != null && currency.length != 3)
        null
      else
        currency
    }

    val cleanCountryUdf = udf( cleanCountry _ )
    val cleanCurrencyUdf = udf( cleanCurrency _ )

    val dfCountry: DataFrame = dfNoFutur
      .withColumn("country2", cleanCountryUdf($"country", $"currency"))
      .withColumn("currency2", cleanCurrencyUdf($"currency"))
      .drop("country", "currency")

    println("\n")
    println("DataFrame après clean via deux udf")
    dfCountry.show(10)


    // Autre façon de faire : en utilisant sql.functions.when
    // la ligne de code est commentée car nous avons déjà executé l'opération, toutefois ça fonctionne.
    // En utilisant sql.functions.when (en performance les fonctions sql sont meilleurs que les udf en général) :

    /*
        dfNoFutur
          .withColumn("country2", when($"country" === "False", $"currency").otherwise($"country"))
          .withColumn("currency2", when($"country".isNotNull && length($"currency") =!= 3, null).otherwise($"currency"))
          .drop("country", "currency")
        dfNoFutur.show(10)
    */

    // Retrait des valeurs dont le final status n'est pas 0 ou 1 dont le nombre est :
    println(s"Nombre de lignes avant suppression : ${dfCountry.count}")
    //suppression
    val dfCountryFiltered = dfCountry.where("final_status<2")
    println(s"Nombre de lignes apres suppression : ${dfCountryFiltered.count}")

    // Ajouter et manipuler des colonnes"
    // "Ajout d'une colonne days_campaign qui représente la durée de la campagne en jours (le nombre de jours entre launched_at et deadline)
    val dfCountryFiltered2 = dfCountryFiltered.withColumn("days_campaign", ((col("deadline") - col("launched_at"))/86400).cast("Int"))

    // Ajout d'une colonne hours_prepa qui représente le nombre d’heures de préparation de la campagne entre created_at et launched_at
    val dfCountryFiltered3 = dfCountryFiltered2.withColumn("hours_prepa", round(((col("launched_at") - col("created_at"))/3600),3))

    dfCountryFiltered3.drop("launched_at").drop("created_at").drop("deadline")

    // Mettons les colonnes name, desc, et keywords en minuscules

    val dfCountryFiltered4: DataFrame = dfCountryFiltered3
    .withColumn("name", lower(col("name")))
    .withColumn("desc", lower(col("desc")))
    .withColumn("keywords", lower(col("keywords")))

    // "Concaténation des colonnes textuelles en une seule colonne text"

   val dfCountryFiltered5: DataFrame = dfCountryFiltered4
     .withColumn("text", concat($"name", lit(" "), $"desc", lit(" "), $"keywords"))


    // "Valeurs nulles"
    // "Remplaçons les valeurs nulles des colonnes days_campaign, hours_prepa
    //  Remplaçons goal par la valeur -1 et par \"unknown\" pour les colonnes country2 et currency2."

    val finalDataFrame: DataFrame = dfCountryFiltered5
      .na.fill(-1, Seq("days_campaign"))
      .na.fill(-1, Seq("hours_prepa"))
      .na.fill(-1, Seq("goal"))
      .na.fill("unknown", Seq("country2"))
      .na.fill("unknown", Seq("currency2"))

    println("Affichage du dataframe final avant de le transformer au format parquet")
    finalDataFrame.show(10)

    println("Sauvegarde du DataFrame au format parquet")
    finalDataFrame.write.mode("overwrite").parquet(s"$pathToData/FinalDataFrame")

  }
}
