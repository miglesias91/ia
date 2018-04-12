#pragma once

// stl
#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <bitset>

// scraping
#include <depuracion/include/Depurador.h>

namespace ia
{
namespace clasificacion
{

class Vocabulario
{
public:

    struct config
    {
        bool aplicar_stemming;
        unsigned int cantidad_minima_de_apariciones;
        std::string path_mapa_utf8;
        std::string path_stopwords;
    };

    Vocabulario();
    Vocabulario(const config & configuracion);
    virtual ~Vocabulario();

    // GETTERS

    // SETTERS

    // METODOS

    // genera el vocabulario a partir del corpus (grupo de textos) ingresados por parametro, segun la configuracion seteada.
    void generar(const std::vector<std::string> & corpus, const config & configuracion);

    // devuelve un vector que representa la frecuencia de cada palabra del vocabulario dentro de la bolsa de palabras.
    unsigned int vectorizar(const std::vector<std::string> & bolsa_de_palabras, std::vector<unsigned int> & vector_conteo);

    // depura el texto y lo devuelve como una bolsa de palabras.
    void depurar(std::string texto_a_depurar, std::vector<std::string> & bolsa_de_palabras);

    // agrega una palabra al vocabulario. devuelve la cantidad de apariciones que tiene la palabra.
    unsigned int agregar(const std::string & palabra);

    unsigned int agregar_sincro(const std::string & palabra);

    // agrega una bolsa de palabras al vocabulario.
    void agregar(const std::vector<std::string> & bolsa_de_palabras);

    // importa el vocabulario desde un archivo.
    void importar(const std::string & path);

    // exporta el vocabulario a un archivo.
    void exportar(const std::string & path);

    // CONSULTAS

private:

    // METODOS PRIVADOS

    void analizar_textos(std::vector<std::string> corpus);

    // listado de palabras en el vocabulario. cada palabra tiene asociada su cantidad de apariciones.
    std::unordered_map<std::string, unsigned int> vocabulario;

    std::vector<std::string> vocabulario_palabras;

    scraping::depuracion::Depurador depurador;

    unsigned int cantidad_minima_de_apariciones;

    bool aplicar_stemming;

    std::mutex mutex;
};

};
};


