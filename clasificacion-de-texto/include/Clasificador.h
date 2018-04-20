#pragma once

// stl
#include <string>

// tiny-dnn
#include <tiny_dnn/tiny_dnn.h>

// herramientas
#include <utiles/include/FuncionesSistemaArchivos.h>
#include <utiles/include/FuncionesString.h>

// clasificacion-de-texto
#include<clasificacion-de-texto/include/Dataset.h>

namespace ia
{
namespace clasificacion
{

class Clasificador
{
public:

    // configuracion de la red neuronal
    struct config_red_neuronal
    {
        unsigned long int tamanio_capa_entrada;
        unsigned long int tamanio_capa_salida;
    };

    struct config_entrenamiento
    {
        config_entrenamiento() {};

        config_entrenamiento(std::string path)
        {
            std::string contenido;
            herramientas::utiles::FuncionesSistemaArchivos::leer(path, contenido);
            std::vector<std::string> valores = herramientas::utiles::FuncionesString::separar(contenido, ",");

            this->optimizador = valores[0];
            this->tamanio_batch = std::stoul(valores[1]);
            this->numero_de_ciclos = std::stoul(valores[2]);
        }

        std::string optimizador;
        unsigned long int tamanio_batch;
        unsigned long int numero_de_ciclos;
    };

    // EL DATASET TIENE QUE ESTAR PREVIAMENTE CARGADO Y PREPARADO. EL CLASIFICADOR NO LO MODIFICA.
    Clasificador(Dataset * dataset = nullptr);
    virtual ~Clasificador();

    // GETTERS

    // SETTERS

    // METODOS

    bool entrenar(config_entrenamiento configuracion);

    void evaluar();

    void predecir(const std::vector<float> & valores, std::string & clase);

    bool guardar(const std::string & path_red_neuronal, const std::string & path_mapeo_clases);

    bool cargar(const std::string & path_red_neuronal, const std::string & path_mapeo_clases);

    // CONSULTAS

private:

    // METODOS PRIVADOS

    tiny_dnn::optimizer * crearOptimizador(const config_entrenamiento & configuracion);

    // ATRIBUTOS

    tiny_dnn::network<tiny_dnn::sequential> red_neuronal;
    
    config_red_neuronal config_rn;

    Dataset * dataset;
    std::unordered_map<tiny_dnn::label_t, std::string> mapeo_id_clase;
};

};
};


