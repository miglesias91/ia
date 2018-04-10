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

bool Clasificador::entrenar()
{
    tiny_dnn::adagrad opt;
    opt.alpha = 0.01;

    size_t tamanio_batch = 50;
    size_t numero_de_ciclos = 100;

    std::vector<tiny_dnn::vec_t> entrada_entrenamiento;
    this->dataset->getValoresDeEntradaEntrenamiento(&entrada_entrenamiento);

    std::vector<tiny_dnn::label_t> salida_deseada_entrenamiento;
    this->dataset->getSalidaDeseadaEntrenamiento(&salida_deseada_entrenamiento);

    bool resultado = this->red_neuronal.train<tiny_dnn::mse>(opt, entrada_entrenamiento, salida_deseada_entrenamiento, tamanio_batch, numero_de_ciclos);

    return resultado;
}

void Clasificador::evaluar()
{
    std::vector<tiny_dnn::vec_t> entrada_evaluacion;
    this->dataset->getValoresDeEntradaEvaluacion(&entrada_evaluacion);

    std::vector<tiny_dnn::label_t> salida_deseada_evaluacion;
    this->dataset->getSalidaDeseadaEvaluacion(&salida_deseada_evaluacion);

    tiny_dnn::result resultado = this->red_neuronal.test(entrada_evaluacion, salida_deseada_evaluacion);
    resultado.print_detail(std::cout);
}

// GETTERS

// SETTERS

// METODOS

// CONSULTAS