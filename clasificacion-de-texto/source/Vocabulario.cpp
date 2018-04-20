#include <clasificacion-de-texto/include/Vocabulario.h>

// stl
#include <algorithm>

// utiles
#include <utiles/include/Stemming.h>
#include <utiles/include/FuncionesSistemaArchivos.h>
#include <utiles/include/FuncionesString.h>

using namespace ia::clasificacion;

Vocabulario::Vocabulario() : cantidad_minima_de_apariciones(0), aplicar_stemming(true)
{
}

Vocabulario::Vocabulario(const config & configuracion) : cantidad_minima_de_apariciones(0), aplicar_stemming(true)
{
    this->cantidad_minima_de_apariciones = configuracion.cantidad_minima_de_apariciones;

    // cargo mapa utf8
    scraping::depuracion::mapeo::MapaUTF8 * mapa_utf8 = new scraping::depuracion::mapeo::MapaUTF8(configuracion.path_mapa_utf8);
    scraping::depuracion::Depurador::setMapaUTF8(mapa_utf8);

    // cargo stopwords
    std::string contenido = "";
    herramientas::utiles::FuncionesSistemaArchivos::leer(configuracion.path_stopwords, contenido);
    std::vector<std::string> stopwords = herramientas::utiles::FuncionesString::separar(contenido, "\n");
    scraping::depuracion::Depurador::setStopwords(stopwords);

    this->aplicar_stemming = configuracion.aplicar_stemming;
}

Vocabulario::~Vocabulario()
{
}

// GETTERS

// SETTERS

// METODOS

void Vocabulario::analizar_textos(std::vector<std::string> corpus)
{
    static unsigned int i = 0;
    std::unique_lock<std::mutex> lock(this->mutex);
    for (std::string texto : corpus)
    {
        std::vector<std::string> bolsa_de_palabras;
        this->depurar(texto, bolsa_de_palabras);

        this->agregar(bolsa_de_palabras);

        std::cout << "registro analizado: " << i << std::endl;
        i++;
    }
}

void Vocabulario::generar(const std::vector<std::string> & corpus, const config & configuracion)
{
    this->cantidad_minima_de_apariciones = configuracion.cantidad_minima_de_apariciones;
    this->aplicar_stemming = configuracion.aplicar_stemming;

    // cargo mapa utf8
    scraping::depuracion::mapeo::MapaUTF8 * mapa_utf8 = new scraping::depuracion::mapeo::MapaUTF8(configuracion.path_mapa_utf8);
    scraping::depuracion::Depurador::setMapaUTF8(mapa_utf8);

    // cargo stopwords
    std::string contenido = "";
    herramientas::utiles::FuncionesSistemaArchivos::leer(configuracion.path_stopwords, contenido);
    std::vector<std::string> stopwords = herramientas::utiles::FuncionesString::separar(contenido, "\n");
    scraping::depuracion::Depurador::setStopwords(stopwords);

    // divido el corpus en pedazos para analizarlo en varios hilos
    unsigned int i = 0;
    for (std::string texto : corpus)
    {
        std::vector<std::string> bolsa_de_palabras;
        this->depurar(texto, bolsa_de_palabras);

        this->agregar(bolsa_de_palabras);

        std::cout << "registro analizado: " << i << std::endl;
        i++;
    }
}

unsigned int Vocabulario::vectorizar(const std::vector<std::string> & bolsa_de_palabras, std::vector<unsigned int> & vector_conteo)
{
    // inicializo el vector con el tamanio del vocabulario.
    vector_conteo = std::vector<unsigned int>(this->vocabulario_palabras.size(), 0);

    unsigned int cantidad_de_matcheos = 0;
    for (std::string palabra : bolsa_de_palabras)
    {
        std::vector<std::string>::iterator it_palabra = std::lower_bound(this->vocabulario_palabras.begin(), this->vocabulario_palabras.end(), palabra);
        if (palabra == *it_palabra)
        {// si entra aca, entonces encontro la palabra en el vocabulario.
            unsigned long long int posicion_palabra_en_vector = std::distance(this->vocabulario_palabras.begin(), it_palabra);

            unsigned int conteo = vector_conteo[posicion_palabra_en_vector];
            vector_conteo[posicion_palabra_en_vector] = conteo + 1;

            cantidad_de_matcheos++;
        }
    }
    return cantidad_de_matcheos;
}

unsigned int Vocabulario::agregar(const std::string & palabra)
{
    std::unordered_map<std::string, unsigned int>::iterator it_palabra = this->vocabulario.find(palabra);

    if (this->vocabulario.end() != it_palabra)
    {
        unsigned int nuevo_valor = it_palabra->second + 1;
        this->vocabulario[palabra] = nuevo_valor;
    }
    else
    {
        this->vocabulario[palabra] = 1;
    }

    return this->vocabulario[palabra];
}

unsigned int Vocabulario::agregar_sincro(const std::string & palabra)
{
    std::unique_lock<std::mutex> lock(this->mutex);

    std::unordered_map<std::string, unsigned int>::iterator it_palabra = this->vocabulario.find(palabra);

    if (this->vocabulario.end() != it_palabra)
    {
        unsigned int nuevo_valor = it_palabra->second + 1;
        this->vocabulario[palabra] = nuevo_valor;
    }
    else
    {
        this->vocabulario[palabra] = 1;
    }

    return this->vocabulario[palabra];
}

void Vocabulario::agregar(const std::vector<std::string> & bolsa_de_palabras)
{
    for (std::string palabra : bolsa_de_palabras)
    {
        this->agregar(palabra);
    }
}

void Vocabulario::importar(const std::string & path)
{
    std::string contenido_vocabulario;
    herramientas::utiles::FuncionesSistemaArchivos::leer(path, contenido_vocabulario);

    std::vector<std::string> vocabulario = herramientas::utiles::FuncionesString::separar(contenido_vocabulario, "\n");

    for (std::string palabra_y_frecuencia : vocabulario)
    {
        std::vector<std::string> palabra_frecuencia = herramientas::utiles::FuncionesString::separar(palabra_y_frecuencia, ",");
        std::string palabra = palabra_frecuencia[0];
        unsigned int frecuencia = std::stoul(palabra_frecuencia[1]);

        this->vocabulario[palabra] = frecuencia;

        this->vocabulario_palabras.push_back(palabra);
    }

    std::sort(vocabulario_palabras.begin(), vocabulario_palabras.end());
}

void Vocabulario::exportar(const std::string & path)
{
    std::string contenido_vocabulario = "";
    for (std::pair<std::string, unsigned int> palabra : this->vocabulario)
    {
        if (this->cantidad_minima_de_apariciones < palabra.second)
        {
            contenido_vocabulario += palabra.first + "," + std::to_string(palabra.second) + "\n";
        }
    }

    herramientas::utiles::FuncionesSistemaArchivos::escribir(path, contenido_vocabulario);
}

void Vocabulario::depurar(std::string texto_a_depurar, std::vector<std::string> & bolsa_de_palabras)
{
    // logica copiada de scraping::depuracion::Depurador::depurar;

    unsigned int caracteres_especiales_reemplazados = this->depurador.reemplazarTodosLosCaracteresEspecialesExceptoTildes(texto_a_depurar);

    // 2do: reemplazo las mayusculas por minusculas.
    bool pasado_a_minuscula = this->depurador.todoMinuscula(texto_a_depurar);

    // 3ero: reemplazo las mayusculas por minusculas.
    unsigned int urls_eliminadas = this->depurador.eliminarURLs(texto_a_depurar);

    // 4to: elimino los simbolos que no forman palabras.
    unsigned int caracteres_signos_puntuacion_eliminados = this->depurador.eliminarSignosYPuntuacion(texto_a_depurar);

    // 5to: elimino los caracteres de control de texto.
    unsigned int caracteres_de_control_eliminados = this->depurador.eliminarCaracteresDeControl(texto_a_depurar);

    // 6to: paso de un texto con palabras a un vector con tokens.
    bolsa_de_palabras = this->depurador.tokenizarTexto(texto_a_depurar);

    // 7mo: elimino las palabras con menos de 2 letras.
    unsigned int cantidad_palabras_muy_cortas_eliminadas = this->depurador.eliminarPalabrasMuyCortas(bolsa_de_palabras);

    // 8vo: elimino las palabras con mas de 15 letras.
    unsigned int cantidad_palabras_muy_largas_eliminadas = this->depurador.eliminarPalabrasMuyLargas(bolsa_de_palabras);

    unsigned int cantidad_stopwords_eliminadas = this->depurador.eliminarStopwords(bolsa_de_palabras);

    if (this->aplicar_stemming)
    {
        herramientas::utiles::Stemming::stemUTF8(bolsa_de_palabras);
    }
}

// CONSULTAS