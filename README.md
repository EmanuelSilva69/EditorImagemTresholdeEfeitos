# Processamento Digital de Imagens - Thresholding

Script principal: `processamento_thresholds.py`

Interface local (sem Streamlit): `main.py`
Modulo de filtros de troca de cor principal: `filtros_cor.py`

## Bibliotecas

Instale as dependencias:

```bash
pip install -r requirements.txt
```

Observacao: a interface local usa `Pillow` para exibir miniaturas (preview) dentro da janela.

## 1) Modo Imagem (upload local)

Sem passar `--image`, o script abre um seletor de arquivo local.

```bash
python processamento_thresholds.py --mode image --preprocess --blur gaussian --equalize
```

No modo imagem, cada resultado e salvo em arquivo separado (sem textos) em `resultados/` com dimensao 16:9 (1920x1080), pronto para apresentacao.
Tambem e possivel exibir/salvar um painel de analise avancada com histogramas e graficos de metricas.

Organizacao de saida:

- `resultados/img`: imagens filtradas, comparativos e saidas dos metodos
- `resultados/hist`: histogramas e paineis de analise

Arquivos gerados por imagem:

- `nome_original.png`
- `nome_adaptativo.png`
- `nome_multi_otsu.png`
- `nome_range.png`
- `nome_estatistico.png`

Isso exibe lado a lado:

- Original
- Threshold Adaptativo (`threshold_local`)
- Multi-Otsu (`threshold_multiotsu`)
- Range Thresholding
- Threshold Estatistico (media/desvio)

Metodos globais extras adicionados:

- Yen
- Triangle
- Otsu
- Minimum
- Mean (media de cinza)
- ISODATA

## 2) Modo Video (tempo real e/ou salvar output)

Sem passar `--video`, o script abre seletor de arquivo local.

Exibir em tempo real:

```bash
python processamento_thresholds.py --mode video --video-method adaptive --show-video --preprocess --blur median
```

Salvar video processado:

```bash
python processamento_thresholds.py --mode video --video "caminho/do/video.mp4" --video-method range --output "saida_threshold.mp4"
```

## Parametros uteis

- `--block-size`, `--offset`, `--adaptive-method`, `--adaptive-polarity`: adaptativo
- `--classes`, `--multi-mode`, `--multi-target-class`: Multi-Otsu
- `--range-low`, `--range-high`, `--range-invert`: range threshold
- `--k`, `--stat-local`, `--stat-window`, `--stat-polarity`: threshold estatistico
- `--preprocess`, `--blur`, `--blur-kernel`, `--equalize`, `--clahe`: pre-processamento
- `--save-figure` / `--no-save-figure`: ativa ou desativa exportacao automatica da figura comparativa
- `--save-individual` / `--no-save-individual`: ativa ou desativa exportacao separada por tecnica
- `--show-analysis` / `--no-show-analysis`: exibe ou oculta painel analitico (histogramas + metricas)
- `--save-analysis`: salva painel analitico em PNG
- `--save-extra-methods` / `--no-save-extra-methods`: salva/nao salva saidas dos metodos extras
- `--save-histograms` / `--no-save-histograms`: salva/nao salva painel de histogramas dos metodos extras
- `--figure-dir`: pasta de saida da figura para slides (padrao: `resultados`)
- `--figure-width` e `--figure-height`: dimensao da imagem em pixels (padrao: `1920x1080`)
- `--figure-dpi`: DPI usado na exportacao (padrao: `100`)

## 3) Interface local com janela (Tkinter)

Para facilitar o uso, execute:

```bash
python main.py
```

Na interface voce pode:

- Selecionar uma ou mais imagens
- Selecionar uma pasta inteira de imagens (processamento em lote)
- Ajustar parametros de threshold e pre-processamento
- Processar apenas a imagem selecionada ou processar todas
- Ver os resultados na tela e salvar automaticamente em `resultados/`
- Visualizar miniaturas (Original + 4 thresholds) ao clicar em uma imagem na lista
- Clicar em `Mostrar filtros extras` para abrir as opcoes auxiliares de filtros
- Exibir e/ou salvar uma analise avancada com histogramas e graficos de apoio
- Acompanhar barra de progresso durante o processamento

Agora o app possui duas abas:

- `Threshold`: fluxo completo de limiarizacao e analise
- `Troca de Cor`: filtro para trocar a cor principal da imagem (ex.: vermelho -> verde), preservando luminosidade

Na aba `Troca de Cor`:

- Selecione imagem de entrada
- Escolha cor de origem (`auto` ou manual) por combobox
- Escolha cor de destino por combobox
- Opcional: ative `Usar sliders de Hue` para ajustar origem/destino por matiz numerica (0-179)
- Opcional: ative `Usar seletor RGB para cor destino` e use `Escolher cor final (RGB)`
- Ajuste tolerancia e suavizacao
- Aplique e salve em `resultados/img` (ou pasta de saida escolhida)
- Aplique tambem o efeito `Memory Overflow` com intensidade configuravel
- Preview em tempo real para os parametros de troca de cor e overflow

## Observacao

No modo video, pressione `q` para encerrar a visualizacao em tempo real.
