#include <clasificacion-de-texto/include/Clasificador.h>

using namespace ia::clasificacion;

Clasificador::Clasificador(Dataset * dataset) : dataset(dataset)
{
    this->red_neuronal = tiny_dnn::make_mlp <tiny_dnn::activation::tanh>(
    {
        dataset->getTamanioValores(),
        dataset->getTamanioClases()
    });

    this->config_rn.tamanio_capa_entrada = dataset->getTamanioValores();
    this->config_rn.tamanio_capa_salida = dataset->getTamanioClases();
}

Clasificador::~Clasificador()
{
}

//bool Clasificador::cargarDataSet(const std::string & path_dataset)
//{
//    {
//        std::vector<std::string> registros;
//        {
//            std::string string_dataset = "";
//            herramientas::utiles::FuncionesSistemaArchivos::leer(path_dataset, string_dataset);
//
//            registros = herramientas::utiles::FuncionesString::separar(string_dataset, "\n");
//        }
//
//        std::vector<std::string> columnas;
//        for (const auto & registro : registros)
//        {
//            columnas = herramientas::utiles::FuncionesString::separar(registro, ",");
//
//            data una_data;
//            for (std::vector<std::string>::iterator it = columnas.begin(); it != (columnas.end() - 1); it++)
//            {
//                una_data.valores.push_back(std::stof(*it));
//            }
//            una_data.clase = this->getIDClase(*(columnas.end() - 1));
//
//            this->dataset.push_back(una_data);
//        }
//    }
//
//    return true;
//}

bool Clasificador::entrenar()
{
    tiny_dnn::adagrad opt;

    size_t tamanio_batch = 1;
    size_t numero_de_ciclos = 30;

    bool resultado = this->red_neuronal.train<tiny_dnn::mse>(opt, *this->dataset->getValoresDeEntradaEntrenamiento(), *this->dataset->getSalidaDeseadaEntrenamiento(), tamanio_batch, numero_de_ciclos);

    return resultado;
}

void Clasificador::evaluar()
{

}

//tiny_dnn::label_t Clasificador::getIDClase(std::string nombre_clase)
//{
//    const auto & iterador_mapa = this->mapa_clase_id.find(nombre_clase);
//
//    if (this->mapa_clase_id.end() == iterador_mapa)
//    {
//        this->mapa_clase_id[nombre_clase] = this->contador_ids;
//        this->contador_ids++;
//    }
//
//    return this->mapa_clase_id[nombre_clase];
//}

// GETTERS

// SETTERS

// METODOS

// CONSULTAS