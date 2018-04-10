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
        std::string optimizador;
        float tasa_de_aprendizaje;
        unsigned long int tamanio_batch;
        unsigned long int numero_de_ciclos;
    };

    Clasificador(Dataset * dataset);
    virtual ~Clasificador();

    // GETTERS

    // SETTERS

    // METODOS

    bool entrenar();

    void evaluar();

    // CONSULTAS

private:

    // METODOS PRIVADOS

    // ATRIBUTOS

    tiny_dnn::network<tiny_dnn::sequential> red_neuronal;
    
    config_red_neuronal config_rn;

    Dataset * dataset;
};

//template <typename tipo_de_dato>
//bool Clasificador::cargarDataSet(const std::string & path_dataset)
//{
//    
//
//}

};
};


