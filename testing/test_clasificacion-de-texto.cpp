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

        std::cout << "registro analizado: " << i << std::endl;
        i++;
    }

    herramientas::utiles::FuncionesSistemaArchivos::escribir("bolsas_de_palabras_con_polaridad.txt", contenido_a_escribir);
}

TEST(clasificacion_de_texto, DISABLED_bolsas_de_palabras_2_vector_vocabulario)
{
    std::string bolsas_de_palabras = "";

    herramientas::utiles::FuncionesSistemaArchivos::leer("bolsas_de_palabras_con_polaridad_curado.txt", bolsas_de_palabras);
    
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

        vocabulario.vectorizar(bolsa_de_palabras, vector_bolsa_de_palabra);

        vectores.push_back(std::pair<std::vector<unsigned int>, std::string>(vector_bolsa_de_palabra, polaridad));
        std::cout << "bolsa de palabras vectorizada: " << i << std::endl;
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

    herramientas::utiles::FuncionesSistemaArchivos::escribir("dataset.csv", contenido_vectores);
}

TEST(clasificacion_de_texto, clasificar_dataset)
{
    ia::clasificacion::Dataset * dataset = new ia::clasificacion::Dataset("creditcard_equilibrado_mezclado.csv");

    //dataset.preparar(); // igualar la cantidad de registros por clase + ordenar aleatoriamente.

    ia::clasificacion::Clasificador clasificador(dataset);

    clasificador.entrenar();

    clasificador.evaluar();
}