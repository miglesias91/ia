#pragma once

// stl
#include <string>

// tiny-dnn
#include <tiny_dnn/tiny_dnn.h>

// herramientas
#include <utiles/include/FuncionesSistemaArchivos.h>
#include <utiles/include/FuncionesString.h>

namespace ia
{
namespace clasificacion
{

class Dataset
{
public:

    Dataset();
    Dataset(std::vector<tiny_dnn::vec_t> valores_de_entrada, std::vector<tiny_dnn::label_t> salida_deseada);
    Dataset(const std::string & path_dataset);
    virtual ~Dataset();

    // GETTERS

    unsigned long int getTamanio();
    unsigned long int getTamanioValores();
    unsigned long int getTamanioClases();

    std::vector<tiny_dnn::vec_t> * getValoresDeEntradaEntrenamiento();
    std::vector<tiny_dnn::label_t> * getSalidaDeseadaEntrenamiento();

    std::vector<tiny_dnn::vec_t> * getValoresDeEntradaEvaluacion();
    std::vector<tiny_dnn::label_t> * getSalidaDeseadaEvaluacion();

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
    bool cargar(const std::string & path_dataset, float porcentaje_de_entrenamiento);

    // CONSULTAS

private:

    // METODOS PRIVADOS

    tiny_dnn::label_t getIDClase(std::string nombre_clase);

    // ATRIBUTOS

    tiny_dnn::label_t contador_ids;
    std::unordered_map<std::string, tiny_dnn::label_t> mapa_clase_id;

    std::vector<tiny_dnn::vec_t> valores_de_entrada_entrenamiento;
    std::vector<tiny_dnn::label_t> salida_deseada_entrenamiento;

    std::vector<tiny_dnn::vec_t> valores_de_entrada_evaluacion;
    std::vector<tiny_dnn::label_t> salida_deseada_evaluacion;

    float porcentaje_de_entrenamiento;
};

};
};


