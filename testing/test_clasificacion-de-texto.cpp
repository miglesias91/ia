
#define GTEST_LANG_CXX11 1

// gtest
#include <gtest/gtest.h>

// utiles
#include <utiles/include/FuncionesSistemaArchivos.h>
#include <utiles/include/FuncionesString.h>

// clasificacion-de-texto
#include <clasificacion-de-texto/include/Clasificador.h>
#include <clasificacion-de-texto/include/Vocabulario.h>

TEST(clasificacion_de_texto, DISABLED_generar_vocabulario)
{
    std::string contenido = "";

    herramientas::utiles::FuncionesSistemaArchivos::leer("tweets_curados.txt", contenido);

    std::vector<std::string> corpus = herramientas::utiles::FuncionesString::separar(contenido, "\n");

    ia::clasificacion::Vocabulario vocabulario;

    ia::clasificacion::Vocabulario::config config_vocabulario = { true, 5, "mapeo_utf8.json", "stopwords_espaniol.txt" };

    vocabulario.generar(corpus, config_vocabulario);

    vocabulario.exportar("vocabulario.txt");
}

TEST(clasificacion_de_texto, DISABLED_textos_2_bolsas_de_palabras)
{
    std::string contenido = "";

    herramientas::utiles::FuncionesSistemaArchivos::leer("tweets_curados_con_polaridad.txt", contenido);

    std::vector<std::string> corpus_con_polaridad = herramientas::utiles::FuncionesString::separar(contenido, "\n");

    ia::clasificacion::Vocabulario::config config_vocabulario = { true, 5, "mapeo_utf8.json", "stopwords_espaniol.txt" };

    ia::clasificacion::Vocabulario vocabulario(config_vocabulario);

    std::string contenido_a_escribir = "";
    unsigned int i = 1;
    for (std::string texto_con_polaridad : corpus_con_polaridad)
    {
        std::vector<std::string> texto_y_polaridad = herramientas::utiles::FuncionesString::separar(texto_con_polaridad, ",");

        std::string texto = texto_y_polaridad[0];
        std::string polaridad = texto_y_polaridad[1];

        std::vector<std::string> bolsa_de_palabras;
        vocabulario.depurar(texto, bolsa_de_palabras);

        contenido_a_escribir += herramientas::utiles::FuncionesString::unir(bolsa_de_palabras, " ") + "," + polaridad + "\n";

        if(0 == i % 1000)
        {
            std::cout << "registro analizado: " << i << std::endl;
        }
        i++;
    }

    herramientas::utiles::FuncionesSistemaArchivos::escribir("bolsas_de_palabras_con_polaridad.txt", contenido_a_escribir);
}

TEST(clasificacion_de_texto, DISABLED_bolsas_de_palabras_2_vector_vocabulario)
{
    std::string bolsas_de_palabras = "";

    herramientas::utiles::FuncionesSistemaArchivos::leer("bolsas_de_palabras_2clases_curado.txt", bolsas_de_palabras);
    
    std::vector<std::string> bolsas_de_palabras_con_polaridad = herramientas::utiles::FuncionesString::separar(bolsas_de_palabras, "\n");

    ia::clasificacion::Vocabulario vocabulario;
    vocabulario.importar("vocabulario_curado.txt");

    std::vector<std::pair<std::vector<unsigned int>, std::string>> vectores;
    unsigned int i = 0;
    for (std::string bolsa_de_palabras_y_polaridad : bolsas_de_palabras_con_polaridad)
    {
        std::vector<std::string> bolsa_de_palabras_polaridad = herramientas::utiles::FuncionesString::separar(bolsa_de_palabras_y_polaridad, ",");
        std::vector<std::string> bolsa_de_palabras = herramientas::utiles::FuncionesString::separar(bolsa_de_palabras_polaridad[0]);
        std::string polaridad = bolsa_de_palabras_polaridad[1];

        std::vector<unsigned int> vector_bolsa_de_palabra;

        if (vocabulario.vectorizar(bolsa_de_palabras, vector_bolsa_de_palabra))
        {   // si ninguna de las palabras de la bolsa estaba dentro del vocabulario, se devolvio un vector vacio.
            // entonces no lo agrego.
            vectores.push_back(std::pair<std::vector<unsigned int>, std::string>(vector_bolsa_de_palabra, polaridad));
        }

        if (0 == i % 1000)
        {
            std::cout << "bolsa de palabras vectorizada: " << i << std::endl;
        }
        i++;
    }

    std::string contenido_vectores = "";
    for (std::pair<std::vector<unsigned int>, std::string> vector_y_polaridad : vectores)
    {
        for (unsigned int freq_palabra : vector_y_polaridad.first)
        {
            contenido_vectores += std::to_string(freq_palabra) + ",";
        }
        contenido_vectores += vector_y_polaridad.second + "\n";


    }

    herramientas::utiles::FuncionesSistemaArchivos::escribir("dataset_2clases.csv", contenido_vectores);
}

TEST(clasificacion_de_texto, DISABLED_clasificar_dataset)
{
    std::string contenido;
    herramientas::utiles::FuncionesSistemaArchivos::leer("config_test.txt", contenido);

    std::vector<std::string> config_test = herramientas::utiles::FuncionesString::separar(contenido, "\n");

    //std::string path_dataset = "dataset_vocab1k_reducido_balanceado_mezclado.csv";
    std::string path_dataset = config_test[0];

    std::cout << "cargando " + path_dataset + "." << std::endl;

    //ia::clasificacion::Dataset * dataset = new ia::clasificacion::Dataset("creditcard_equilibrado_mezclado.csv");
    ia::clasificacion::Dataset * dataset = new ia::clasificacion::Dataset(path_dataset);

    std::cout << "termino carga." << std::endl;

    dataset->preparar(); // igualar la cantidad de registros por clase + ordenar aleatoriamente.

    std::cout << "termino preparacion." << std::endl;

    ia::clasificacion::Clasificador::config_entrenamiento config(config_test[1]);
    ia::clasificacion::Clasificador clasificador(dataset);

    clasificador.entrenar(config);

    std::cout << "termino entrenamiento." << std::endl;

    clasificador.evaluar();

    std::cout << "termino evaluacion." << std::endl;
}

TEST(clasificacion_de_texto, guardar_y_cargar_clasificador)
{
    //std::string contenido;
    //herramientas::utiles::FuncionesSistemaArchivos::leer("config_test.txt", contenido);

    //std::vector<std::string> config_test = herramientas::utiles::FuncionesString::separar(contenido, "\n");

    std::string path_dataset = "dataset/creditcard_equilibrado_mezclado.csv";
    //std::string path_dataset = config_test[0];

    std::cout << "cargando " + path_dataset + "." << std::endl;

    //ia::clasificacion::Dataset * dataset = new ia::clasificacion::Dataset("creditcard_equilibrado_mezclado.csv");
    ia::clasificacion::Dataset * dataset = new ia::clasificacion::Dataset(path_dataset);

    std::cout << "termino carga." << std::endl;

    dataset->preparar(); // igualar la cantidad de registros por clase + ordenar aleatoriamente.

    std::cout << "termino preparacion." << std::endl;

    ia::clasificacion::Clasificador::config_entrenamiento config;
    config.optimizador = "gradient_0.01_0.001";
    config.tamanio_batch = 1;
    config.numero_de_ciclos = 5;

    ia::clasificacion::Clasificador clasificador(dataset);

    clasificador.entrenar(config);

    std::cout << "termino entrenamiento." << std::endl;

    clasificador.evaluar();

    std::cout << "termino evaluacion." << std::endl;

    clasificador.guardar("rrnn_clasificador.net", "mapeo_ids_clases.csv");

    // LEVANTO LO QUE GUARDE RECIEN.

    ia::clasificacion::Clasificador clasificador_nuevo;

    clasificador_nuevo.cargar("rrnn_clasificador.net", "mapeo_ids_clases.csv");

    std::string texto_a_predecir = "gracias!!! esto es muy bueno.";

    ia::clasificacion::Vocabulario vocabulario;
    vocabulario.importar("vocabulario_curado_1k.txt");

    std::vector<std::string> bolsa_de_palabras_a_predecir;
    vocabulario.depurar(texto_a_predecir, bolsa_de_palabras_a_predecir);

    std::vector<unsigned int> atributos_a_predecir;
    vocabulario.vectorizar(bolsa_de_palabras_a_predecir, atributos_a_predecir);

    std::vector<float> atributos_float_a_predecir;

    std::for_each(atributos_a_predecir.begin(), atributos_a_predecir.end(), [&atributos_float_a_predecir](unsigned int atributo) { atributos_float_a_predecir.push_back(atributo); });

    std::string clase;
    clasificador_nuevo.predecir(atributos_float_a_predecir, clase);

    std::cout << "prediccion: " << clase << std::endl;
}