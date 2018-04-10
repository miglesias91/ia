#include <clasificacion-de-texto/include/Dataset.h>

using namespace ia::clasificacion;

Dataset::Dataset() : contador_ids(0)
{
}

Dataset::Dataset(std::vector<tiny_dnn::vec_t> valores_de_entrada, std::vector<tiny_dnn::label_t> salida_deseada, float porcentaje_de_entrenamiento) :
    contador_ids(0), porcentaje_de_entrenamiento(porcentaje_de_entrenamiento)
{
    for (unsigned int i = 0; i < valores_de_entrada.size(); i++)
    {
        data nueva_data{ valores_de_entrada[i], salida_deseada[i] };

        if(i < valores_de_entrada.size() * porcentaje_de_entrenamiento)
        {
            this->set_entrenamiento.push_back(nueva_data);
        }
        else
        {
            this->set_evaluacion.push_back(nueva_data);
        }
    }
}

Dataset::Dataset(const std::string & path_dataset, float porcentaje_de_entrenamiento) : contador_ids(0), porcentaje_de_entrenamiento(porcentaje_de_entrenamiento)
{
    this->cargar(path_dataset, porcentaje_de_entrenamiento);
}

Dataset::~Dataset()
{
}

// GETTERS

unsigned long int Dataset::getTamanio()
{
    return this->set_entrenamiento.size() + this->set_evaluacion.size();
}

unsigned long int Dataset::getTamanioValores()
{
    return this->set_entrenamiento.front().valores.size();
}

unsigned long int Dataset::getTamanioClases()
{
    return this->contador_ids;
}

unsigned long int Dataset::getValoresDeEntradaEntrenamiento(std::vector<tiny_dnn::vec_t> * valores_de_entrada_entrenamiento)
{
    for (auto data : this->set_entrenamiento)
    {
        valores_de_entrada_entrenamiento->push_back(data.valores);
    }

    return valores_de_entrada_entrenamiento->size();
}

unsigned long int Dataset::getSalidaDeseadaEntrenamiento(std::vector<tiny_dnn::label_t> * salida_deseada_entrenamiento)
{
    for (auto data : this->set_entrenamiento)
    {
        salida_deseada_entrenamiento->push_back(data.clase);
    }

    return salida_deseada_entrenamiento->size();
}

unsigned long int Dataset::getValoresDeEntradaEvaluacion(std::vector<tiny_dnn::vec_t> * valores_de_entrada_evaluacion)
{
    for (auto data : this->set_evaluacion)
    {
        valores_de_entrada_evaluacion->push_back(data.valores);
    }

    return valores_de_entrada_evaluacion->size();
}

unsigned long int Dataset::getSalidaDeseadaEvaluacion(std::vector<tiny_dnn::label_t> * salida_deseada_evaluacion)
{
    for (auto data : this->set_evaluacion)
    {
        salida_deseada_evaluacion->push_back(data.clase);
    }

    return salida_deseada_evaluacion->size();
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

            if (this->set_entrenamiento.size() < porcentaje_de_entrenamiento * registros.size())
            {
                this->set_entrenamiento.push_back(data{ valor, this->getIDClase(*(columnas.end() - 1)) });
            }
            else
            {
                this->set_evaluacion.push_back(data{ valor, this->getIDClase(*(columnas.end() - 1)) });
            }
        }
    }

    return true;
}

void Dataset::mezclar()
{
    std::random_shuffle(this->set_entrenamiento.begin(), this->set_entrenamiento.end());
    std::random_shuffle(this->set_evaluacion.begin(), this->set_evaluacion.end());
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