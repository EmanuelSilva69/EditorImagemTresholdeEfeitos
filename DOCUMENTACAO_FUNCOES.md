# Documentacao Tecnica das Funcoes

Este documento explica, funcao por funcao, o que cada parte do projeto faz, como funciona e qual o objetivo no fluxo de Processamento Digital de Imagens (PDI).

## Visao Geral do Fluxo

1. Carregar imagem ou video.
2. Aplicar pre-processamento opcional (mediana/gaussiano etc.).
3. Aplicar tecnicas de limiarizacao.
4. Gerar visualizacao comparativa e salvar resultados.
5. Imprimir metadados (limiares e tempo de execucao) para analise de beneficios e limitacoes.
6. Gerar painel analitico com histogramas e graficos de metricas.

## processamento_thresholds.py

### Classe MetadadosThreshold

- Objetivo: armazenar metadados de uma tecnica de limiarizacao.
- Campos principais:
  - nome: nome da tecnica.
  - tempo_execucao_ms: tempo de processamento em milissegundos.
  - limiares: valores de limiar calculados automaticamente.
  - parametros: configuracoes relevantes usadas na execucao.
- Metodo __str__:
  - Gera um relatorio formatado para o console.
  - Esse relatorio e usado na apresentacao para discutir desempenho e comportamento dos metodos.

### calcular_percentual_foreground(img)

- Objetivo: medir quanto da imagem final ficou como foreground (pixel > 0).
- Como funciona:
  - Conta pixels nao-zero.
  - Divide pelo total e converte para porcentagem.
- Uso:
  - Estatistica de apoio para comparar tecnicas na visualizacao lado a lado.

### selecionar_arquivo(titulo, tipos_arquivo)

- Objetivo: abrir seletor local de arquivos via Tkinter.
- Como funciona:
  - Cria janela oculta.
  - Abre dialogo de selecao.
  - Retorna caminho selecionado ou None.

### carregar_imagem(caminho)

- Objetivo: carregar imagem em RGB e escala de cinza.
- Como funciona:
  - Se caminho nao for informado, abre seletor local.
  - Le imagem com OpenCV.
  - Converte BGR -> RGB e BGR -> Gray.
- Retorno:
  - rgb, gray, caminho.

### preprocessar(gray, ...)

- Objetivo: melhorar a qualidade para limiarizacao.
- Etapas disponiveis:
  - Gaussian Blur: reduz ruido de alta frequencia.
  - Median Blur: reduz ruido sal-e-pimenta de forma robusta a outliers.
  - Bilateral: suaviza preservando bordas.
  - Equalizacao global: amplia contraste global.
  - CLAHE: melhora contraste local por blocos.
- Observacao:
  - Essa funcao e opcional e controlada por parametros.

### threshold_adaptativo_local(gray, ...)

- Objetivo: limiarizacao local com `skimage.filters.threshold_local`.
- Base matematica:
  - Calcula um limiar espacialmente variante T(x,y).
  - Decisao: I(x,y) > T(x,y) - offset (ou polaridade invertida).
- Metadados gerados:
  - Limiar reportado: media do mapa local T(x,y).
  - Tempo de execucao.
  - Parametros da janela/metodo.

### threshold_multi_otsu(gray, ...)

- Objetivo: segmentacao multiclasse com `skimage.filters.threshold_multiotsu`.
- Base matematica:
  - Escolhe K-1 limiares para minimizar variancia intra-classe e maximizar variancia inter-classes.
  - Generalizacao do Otsu classico.
- Modos de saida:
  - levels: retorna niveis de cinza por classe.
  - class: retorna mascara binaria da classe alvo.
- Metadados gerados:
  - Lista de limiares.
  - Tempo de execucao.
  - Classes/modo/classe alvo.

### threshold_range(gray, baixo, alto, inverter)

- Objetivo: binarizacao por intervalo customizado.
- Base matematica:
  - Mascara: foreground se baixo <= I(x,y) <= alto.
- Metadados gerados:
  - Limiares inferior e superior.
  - Tempo de execucao.
  - Flag de inversao.

### threshold_estatistico(gray, k, local, janela_local, polaridade)

- Objetivo: limiarizacao baseada em media e desvio padrao.
- Base matematica global:
  - media: mu = (1/N) * soma(I_i)
  - desvio: sigma = sqrt((1/N) * soma((I_i - mu)^2))
  - limiar: T = mu + k*sigma
- Base matematica local:
  - T(x,y) = mu_w(x,y) + k*sigma_w(x,y)
  - mu_w e sigma_w estimados por filtros gaussianos em janela local.
- Metadados gerados:
  - Global: limiar unico T.
  - Local: media do mapa T_local(x,y).
  - Tempo de execucao e parametros.

### preparar_imagem_saida(img, largura_px, altura_px)

- Objetivo: gerar saida fixa 16:9 (por padrao 1920x1080).
- Como funciona:
  - Redimensiona preservando aspecto.
  - Centraliza em canvas com padding.
- Uso:
  - Exportacao consistente para slides.

### salvar_resultados_individuais(...)

- Objetivo: salvar arquivos separados por tecnica.
- Saidas tipicas:
  - original
  - adaptativo
  - multi_otsu
  - range
  - estatistico
- Formato:
  - PNG, dimensao configuravel (padrao 1920x1080).

### mostrar_resultados_lado_a_lado(...)

- Objetivo: plot unico comparando original + 4 tecnicas.
- Como funciona:
  - Usa Matplotlib com 5 subplots.
  - Exibe percentual de foreground em cada resultado.
  - Opcionalmente salva figura comparativa.

  ### _calcular_entropia(img)

  - Objetivo: calcular a entropia de Shannon da distribuicao de intensidades.
  - Interpretacao:
    - Maior entropia indica distribuicao mais espalhada e potencialmente maior riqueza de tons.
    - Menor entropia indica imagem mais concentrada em poucos niveis de cinza.

  ### mostrar_analise_avancada(...)

  - Objetivo: gerar painel complementar de diagnostico visual e quantitativo.
  - Graficos incluidos:
    - Histograma da imagem de entrada.
    - Histogramas das saidas das 4 tecnicas.
    - Barras de foreground (%).
    - Barras de tempo de execucao (ms).
    - Barras de contraste (desvio padrao).
    - Barras de entropia de Shannon.
  - Uso:
    - Serve para comparar tecnicas de forma objetiva durante a apresentacao.

### processar_imagem(args)

- Objetivo: pipeline completo para uma imagem.
- Fluxo:
  1. Carrega imagem.
  2. Pre-processa.
  3. Aplica 4 tecnicas.
  4. Imprime metadados de cada tecnica (limiar e tempo).
  5. Salva resultados individuais e/ou comparativo.
  6. Exibe/salva painel analitico avancado (quando habilitado).

### aplicar_metodo(frame_gray, args)

- Objetivo: selecionar tecnica para cada frame de video.
- Como funciona:
  - Roteia para a tecnica escolhida em `--video-method`.
  - Retorna apenas imagem binaria/segmentada (metadados sao descartados no fluxo de video).

### processar_video(args)

- Objetivo: processar video frame a frame.
- Fluxo:
  1. Abre video.
  2. Converte frame para gray.
  3. Pre-processa.
  4. Aplica tecnica selecionada.
  5. Exibe em tempo real (opcional) e/ou salva video de saida.

### construir_parser()

- Objetivo: definir argumentos de linha de comando.
- Inclui:
  - Selecao de modo (imagem/video).
  - Parametros de pre-processamento.
  - Parametros das 4 tecnicas.
  - Parametros de exportacao e exibicao.

### main()

- Objetivo: ponto de entrada da execucao via terminal.
- Como funciona:
  - Le argumentos.
  - Encaminha para `processar_imagem` ou `processar_video`.

## main.py (GUI Tkinter)

### Classe AppThresholdGUI

- Objetivo: fornecer interface local para selecionar imagens, ajustar parametros e processar sem CLI.
- Recursos:
  - Lista de imagens selecionadas.
  - Preview de miniaturas (original + 4 tecnicas).
  - Painel de parametros.
  - Botao para mostrar/ocultar filtros extras.
  - Processamento de imagem selecionada ou em lote.

### _build_ui()

- Objetivo: montar todos os componentes da janela.
- Inclui:
  - Botoes de selecao/remocao.
  - Lista.
  - Preview.
  - Parametros basicos e extras.
  - Barra de status.

### _ler_parametros()

- Objetivo: validar e converter entradas da interface.
- Regras aplicadas:
  - Tamanhos de janela impares.
  - Minimos validos para kernels/tile.
  - Conversoes numericas seguras.

### _gerar_resultados(caminho_imagem, params)

- Objetivo: executar pipeline para uma imagem na GUI.
- Observacao:
  - As funcoes de threshold retornam (imagem, metadados).
  - A GUI usa apenas as imagens para preview/salvamento.

### _processar_arquivo(caminho_imagem, params)

- Objetivo: salvar resultados e opcionalmente mostrar comparativo.
- Integracao:
  - Reaproveita as funcoes do modulo principal para manter consistencia.

### _atualizar_preview_selecionada()

- Objetivo: atualizar miniaturas da imagem selecionada.
- Exibicao:
  - Original, Adaptativo, Multi-Otsu, Range e Estatistico.

### processar_selecionada() e processar_todas()

- Objetivo: executar processamento sob demanda do usuario.
- Comportamento:
  - Mostra progresso na barra de status.
  - Trata erros com caixas de dialogo.

## Resumo de Cobertura dos Requisitos

- 4 tecnicas de limiarizacao: implementadas.
- Upload/arquivo local (imagem e video): implementado.
- Visualizacao lado a lado: implementada.
- Processamento de video por frame: implementado.
- Pre-processamento com mediana e gaussiano: implementado.
- Metadados por tecnica (limiar + tempo): implementado no modo imagem.
- Comentarios/documentacao matematica no codigo: implementados nos docstrings das tecnicas.
- Markdown explicativo de funcoes: este documento.
- Histogramas e graficos extras para analise: implementados (CLI e GUI).
