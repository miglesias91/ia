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

    struct data
    {
        tiny_dnn::vec_t valores;
        tiny_dnn::label_t clase;
    };

    Clasificador(Dataset * dataset);
    virtual ~Clasificador();

    // GETTERS

    // SETTERS

    // METODOS

    // carga el dataset desde un archivo csv. EL DATASET DEBE ESTAR PREPARADO DE FORMA QUE CUMPLA:
    // 1: la cantidad total de columnas es igual a 'tamanio_capa_entrada + 1'.
    // 2: la cantidad total de clases del dataset es igual a 'tamanio_capa_salida'.
    // 3: la ultima columna del csv indica la clase de cada registro.
    // 4: debe haber una cantidad similar de registros para cada clase.
    // 5: los registros deben estar DESORDENADOS (u ordenados aleatoriamente).
    // 6: los valores de los registros son numeros (todas las columnas excepto la ultima).
    // 7: las clases son strings (la ultima columna de cada registro).
    //template <typename tipo_de_dato>
    //bool cargarDataSet(const std::string & path_dataset);

    bool entrenar();

    void evaluar();

    // CONSULTAS

private:

    // METODOS PRIVADOS

    //tiny_dnn::label_t getIDClase(std::string nombre_clase);

    // ATRIBUTOS

    tiny_dnn::network<tiny_dnn::sequential> red_neuronal;

    //tiny_dnn::label_t contador_ids;
    //std::unordered_map<std::string, tiny_dnn::label_t> mapa_clase_id;
    //
    config_red_neuronal config_rn;

    //std::vector<data> dataset;
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


