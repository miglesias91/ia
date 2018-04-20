#include <clasificacion-de-texto/include/Dataset.h>

#include <algorithm>

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

void Dataset::getMapeoClases(std::unordered_map<tiny_dnn::label_t, std::string> & mapa_id_clase)
{
    for (auto & mapeo : this->mapa_clase_id)
    {
        std::string nombre_clase = mapeo.first;
        tiny_dnn::label_t id_clase = mapeo.second.id_clase;

        mapa_id_clase.insert(std::make_pair(id_clase, nombre_clase));
    }
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

void Dataset::preparar()
{
    std::unordered_map<tiny_dnn::label_t, std::vector<data>> mapa_vector_de_clases;

    // obtengo la clase que menos tiene.
    info_clase info_clase_con_menos_instancias = this->mapa_clase_id.begin()->second;
    for (auto & info_clase : this->mapa_clase_id)
    {
        if (info_clase_con_menos_instancias.cantidad_de_instancias > info_clase.second.cantidad_de_instancias)
        {
            info_clase_con_menos_instancias = info_clase.second;
        }

        mapa_vector_de_clases[info_clase.second.id_clase] = std::vector<data>();
    }

    // agrupo las clases en vectores.
    for (auto & data : this->set_entrenamiento)
    {
        (&mapa_vector_de_clases[data.clase])->push_back(data);
    }
    for (auto & data : this->set_evaluacion)
    {
        (&mapa_vector_de_clases[data.clase])->push_back(data);
    }

    // igualo la cantidad de isntancias por clase.
    for (auto & clase : mapa_vector_de_clases)
    {
        if (clase.first == info_clase_con_menos_instancias.id_clase)
        {
            continue;
        }

        std::vector<data> * instancias = &clase.second;
        instancias->erase(instancias->begin() + info_clase_con_menos_instancias.cantidad_de_instancias, instancias->end());
    }

    // vuelvo a guardar las clases en los set de entrenamiento y de evaluacion.
    this->set_entrenamiento.clear();
    this->set_evaluacion.clear();

    for (unsigned long long int i = 0; i < info_clase_con_menos_instancias.cantidad_de_instancias; i++)
    {
        if (i < porcentaje_de_entrenamiento * info_clase_con_menos_instancias.cantidad_de_instancias)
        {
            for (auto & clase : mapa_vector_de_clases)
            {
                this->set_entrenamiento.push_back(clase.second[i]);
            }
        }
        else
        {
            for (auto & clase : mapa_vector_de_clases)
            {
                this->set_evaluacion.push_back(clase.second[i]);
            }
        }
    }
}

void Dataset::mezclar()
{
    //std::random_shuffle(this->set_entrenamiento.begin(), this->set_entrenamiento.end());
    //std::random_shuffle(this->set_evaluacion.begin(), this->set_evaluacion.end());
}

// CONSULTAS

// METODOS PRIVADOS

tiny_dnn::label_t Dataset::getIDClase(std::string nombre_clase)
{
    const auto & iterador_mapa = this->mapa_clase_id.find(nombre_clase);

    if (this->mapa_clase_id.end() == iterador_mapa)
    {
        this->mapa_clase_id[nombre_clase] = info_clase{ 0, this->contador_ids };
        this->contador_ids++;
    }

    (&this->mapa_clase_id[nombre_clase])->cantidad_de_instancias++;

    return this->mapa_clase_id[nombre_clase].id_clase;
}