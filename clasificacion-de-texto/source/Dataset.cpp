#include <clasificacion-de-texto/include/Dataset.h>

using namespace ia::clasificacion;

Dataset::Dataset() : contador_ids(0)
{
}

Dataset::Dataset(std::vector<tiny_dnn::vec_t> valores_de_entrada, std::vector<tiny_dnn::label_t> salida_deseada) : 
    valores_de_entrada(valores_de_entrada), salida_deseada(salida_deseada)
{
}

Dataset::Dataset(const std::string & path_dataset) : contador_ids(0)
{
    this->cargar(path_dataset);
}

Dataset::~Dataset()
{
}

// GETTERS

unsigned long int Dataset::getTamanio()
{
    return this->valores_de_entrada_entrenamiento.size() + this->valores_de_entrada_evaluacion.size();
}

unsigned long int Dataset::getTamanioValores()
{
    return this->valores_de_entrada_entrenamiento.front().size();
}

unsigned long int Dataset::getTamanioClases()
{
    return this->contador_ids;
}

std::vector<tiny_dnn::vec_t> * Dataset::getValoresDeEntradaEntrenamiento()
{
    return &this->valores_de_entrada_entrenamiento;
}

std::vector<tiny_dnn::label_t> * Dataset::getSalidaDeseadaEntrenamiento()
{
    return &this->salida_deseada_entrenamiento;
}

std::vector<tiny_dnn::vec_t> * Dataset::getValoresDeEntradaEvaluacion()
{
    return &this->valores_de_entrada_evaluacion;
}

std::vector<tiny_dnn::label_t> * Dataset::getSalidaDeseadaEvaluacion()
{
    return &this->salida_deseada_evaluacion;
}

// SETTERS

// METODOS

bool Dataset::cargar(const std::string & path_dataset, float porcentaje_de_entrenamiento)
{
    this->porcentaje_de_entrenamiento = porcentaje_de_entrenamiento;
    {
        std::vector<std::string> registros;
        {
            std::string string_dataset = "";
            herramientas::utiles::FuncionesSistemaArchivos::leer(path_dataset, string_dataset);

            registros = herramientas::utiles::FuncionesString::separar(string_dataset, "\n");
        }

        std::vector<std::string> columnas;
        for (const auto & registro : registros)
        {
            columnas = herramientas::utiles::FuncionesString::separar(registro, ",");

            tiny_dnn::vec_t valor;
            for (std::vector<std::string>::iterator it = columnas.begin(); it != (columnas.end() - 1); it++)
            {
               valor.push_back(std::stof(*it));
            }

            if (this->valores_de_entrada_entrenamiento.size() > porcentaje_de_entrenamiento * registros.size())
            {

            }
            this->valores_de_entrada.push_back(valor);
            this->salida_deseada.push_back(this->getIDClase(*(columnas.end() - 1)));

        }
    }

    return true;
}

// CONSULTAS

// METODOS PRIVADOS

tiny_dnn::label_t Dataset::getIDClase(std::string nombre_clase)
{
    const auto & iterador_mapa = this->mapa_clase_id.find(nombre_clase);

    if (this->mapa_clase_id.end() == iterador_mapa)
    {
        this->mapa_clase_id[nombre_clase] = this->contador_ids;
        this->contador_ids++;
    }

    return this->mapa_clase_id[nombre_clase];
}