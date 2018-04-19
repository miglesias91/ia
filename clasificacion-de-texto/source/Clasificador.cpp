#include <clasificacion-de-texto/include/Clasificador.h>

using namespace ia::clasificacion;

Clasificador::Clasificador(Dataset * dataset) : dataset(dataset)
{
    this->red_neuronal = tiny_dnn::make_mlp<tiny_dnn::activation::tanh>(
    {
        dataset->getTamanioValores(),
        dataset->getTamanioClases()
    });

    this->red_neuronal.weight_init(tiny_dnn::weight_init::lecun());
    this->red_neuronal.bias_init(tiny_dnn::weight_init::xavier(2.0));

    this->config_rn.tamanio_capa_entrada = dataset->getTamanioValores();
    this->config_rn.tamanio_capa_salida = dataset->getTamanioClases();
}

Clasificador::~Clasificador()
{
}

bool Clasificador::entrenar(config_entrenamiento configuracion)
{
    //tiny_dnn::adagrad opt;
    tiny_dnn::optimizer * opt = this->crearOptimizador(configuracion);

    size_t tamanio_batch = configuracion.tamanio_batch;
    size_t numero_de_ciclos = configuracion.numero_de_ciclos;

    std::vector<tiny_dnn::vec_t> entrada_entrenamiento;
    this->dataset->getValoresDeEntradaEntrenamiento(&entrada_entrenamiento);

    std::vector<tiny_dnn::label_t> salida_deseada_entrenamiento;
    this->dataset->getSalidaDeseadaEntrenamiento(&salida_deseada_entrenamiento);

    bool resultado = false;
    try
    {
        resultado = this->red_neuronal.train<tiny_dnn::mse>(*opt, entrada_entrenamiento, salida_deseada_entrenamiento, tamanio_batch, numero_de_ciclos,
            []() {},
            [this]()
        {
            static unsigned long int i = 0;
            std::vector<tiny_dnn::vec_t> entrada_evaluacion;
            this->dataset->getValoresDeEntradaEvaluacion(&entrada_evaluacion);

            std::vector<tiny_dnn::label_t> salida_deseada_evaluacion;
            this->dataset->getSalidaDeseadaEvaluacion(&salida_deseada_evaluacion);

            tiny_dnn::result resultado = this->red_neuronal.test(entrada_evaluacion, salida_deseada_evaluacion);

            std::cout << "iteracion: " << i << " - ";

            resultado.print_detail(std::cout);

            i++;
        });
    }
    catch (const tiny_dnn::nn_error & e)
    {
        std::cout << e.what();
    }

    return resultado;
}

void Clasificador::evaluar()
{
    std::vector<tiny_dnn::vec_t> entrada_evaluacion;
    this->dataset->getValoresDeEntradaEvaluacion(&entrada_evaluacion);

    std::vector<tiny_dnn::label_t> salida_deseada_evaluacion;
    this->dataset->getSalidaDeseadaEvaluacion(&salida_deseada_evaluacion);

    tiny_dnn::result resultado;
    try
    {
        resultado = this->red_neuronal.test(entrada_evaluacion, salida_deseada_evaluacion);
    }
    catch (const tiny_dnn::nn_error & e)
    {
        std::cout << e.what();
    }

    resultado.print_detail(std::cout);
}

tiny_dnn::optimizer * Clasificador::crearOptimizador(const config_entrenamiento & configuracion)
{
    std::vector<std::string> valores = herramientas::utiles::FuncionesString::separar(configuracion.optimizador, "_");

    if (valores[0] == "adagrad") {
        tiny_dnn::adagrad * opt = new tiny_dnn::adagrad();
        opt->alpha = std::stof(valores[1]);
      
        return opt;
    };

    if (valores[0] == "rmsprop") {
        tiny_dnn::RMSprop * opt = new tiny_dnn::RMSprop();
        opt->alpha = std::stof(valores[1]);
        opt->mu = std::stof(valores[2]);
      
        return opt;
    };

    if (valores[0] == "adam") {
        tiny_dnn::adam * opt = new tiny_dnn::adam();
        opt->alpha = std::stof(valores[1]);
        opt->b1 = std::stof(valores[2]);
        opt->b1_t = std::stof(valores[3]);
        opt->b2 = std::stof(valores[4]);
        opt->b2_t = std::stof(valores[5]);
      
        return opt;
    };

    if (valores[0] == "adamax") {
        tiny_dnn::adamax * opt = new tiny_dnn::adamax();
        opt->alpha = std::stof(valores[1]);
        opt->b1 = std::stof(valores[2]);
        opt->b1_t = std::stof(valores[3]);
        opt->b2 = std::stof(valores[4]);

        return opt;
    };

    if (valores[0] == "gradient") {
        tiny_dnn::gradient_descent * opt = new tiny_dnn::gradient_descent();
        opt->alpha = std::stof(valores[1]);
        opt->lambda = std::stof(valores[2]);

        return opt;
    };

    if (valores[0] == "momentum") {
        tiny_dnn::momentum * opt = new tiny_dnn::momentum();
        opt->alpha = std::stof(valores[1]);
        opt->lambda = std::stof(valores[2]);
        opt->mu = std::stof(valores[3]);

        return opt;
    };

    if (valores[0] == "nesterov") {
        tiny_dnn::nesterov_momentum * opt = new tiny_dnn::nesterov_momentum();
        opt->alpha = std::stof(valores[1]);
        opt->lambda = std::stof(valores[2]);
        opt->mu = std::stof(valores[3]);

        return opt;
    };

    throw - 1;
    return nullptr;
}

// GETTERS

// SETTERS

// METODOS

// CONSULTAS