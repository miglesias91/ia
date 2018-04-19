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
        config_entrenamiento(std::string path)
        {
            std::string contenido;
            herramientas::utiles::FuncionesSistemaArchivos::leer(path, contenido);
            std::vector<std::string> valores = herramientas::utiles::FuncionesString::separar(contenido, ",");

            this->optimizador = valores[0];
            this->tamanio_batch = std::stoul(valores[1]);
            this->numero_de_ciclos = std::stoul(valores[2]);
        }

        std::string funcion_loss;
        std::string optimizador;
        float tasa_de_aprendizaje;
        float termino_decay;
        unsigned long int tamanio_batch;
        unsigned long int numero_de_ciclos;
    };

    Clasificador(Dataset * dataset);
    virtual ~Clasificador();

    // GETTERS

    // SETTERS

    // METODOS

    bool entrenar(config_entrenamiento configuracion);

    void evaluar();

    // CONSULTAS

private:

    // METODOS PRIVADOS

    tiny_dnn::optimizer * crearOptimizador(const config_entrenamiento & configuracion);

    // ATRIBUTOS

    tiny_dnn::network<tiny_dnn::sequential> red_neuronal;
    
    config_red_neuronal config_rn;

    Dataset * dataset;
};

};
};


