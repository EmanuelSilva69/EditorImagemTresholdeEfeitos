import argparse
import os
import time
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import (
    threshold_isodata,
    threshold_local,
    threshold_minimum,
    threshold_multiotsu,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)


class MetadadosThreshold:
    """Armazena informacoes de auditoria para cada tecnica de limiarizacao."""

    def __init__(self, nome_tecnica: str) -> None:
        self.nome = nome_tecnica
        self.tempo_execucao_ms = 0.0
        self.limiares: list[float] = []
        self.parametros: dict[str, object] = {}

    def __str__(self) -> str:
        linhas = [
            "=" * 68,
            f"RELATORIO: {self.nome}",
            "=" * 68,
            f"Tempo de execucao: {self.tempo_execucao_ms:.4f} ms",
        ]
        if self.limiares:
            limiares_fmt = ", ".join(f"{v:.4f}" for v in self.limiares)
            linhas.append(f"Limiar(es) calculado(s): [{limiares_fmt}]")
        if self.parametros:
            linhas.append("Parametros:")
            for chave, valor in self.parametros.items():
                linhas.append(f"- {chave}: {valor}")
        return "\n".join(linhas)


def calcular_percentual_foreground(img: np.ndarray) -> float:
    """Retorna a porcentagem de pixels classificados como foreground (>0)."""
    total = img.size
    if total == 0:
        return 0.0
    return float(np.count_nonzero(img > 0) * 100.0 / total)


def selecionar_arquivo(titulo: str, tipos_arquivo: Tuple[Tuple[str, str], ...]) -> Optional[str]:
    """Abre um seletor local de arquivos (upload local via interface gráfica)."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        caminho = filedialog.askopenfilename(title=titulo, filetypes=tipos_arquivo)
        root.destroy()
        return caminho if caminho else None
    except Exception as exc:
        print(f"Nao foi possivel abrir a interface grafica de upload: {exc}")
        return None


def carregar_imagem(caminho: Optional[str]) -> Tuple[np.ndarray, np.ndarray, str]:
    """Carrega imagem colorida e em escala de cinza."""
    if not caminho:
        caminho = selecionar_arquivo(
            "Selecione uma imagem",
            (
                ("Imagens", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("Todos os arquivos", "*.*"),
            ),
        )

    if not caminho or not os.path.exists(caminho):
        raise FileNotFoundError("Imagem nao encontrada. Informe um caminho valido.")

    bgr = cv2.imread(caminho, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Falha ao ler a imagem.")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return rgb, gray, caminho


def preprocessar(
    gray: np.ndarray,
    usar_preprocessamento: bool = False,
    blur: str = "gaussian",
    blur_kernel: int = 5,
    equalizar_histograma: bool = False,
    usar_clahe: bool = False,
    clahe_clip: float = 2.0,
    clahe_tile: int = 8,
) -> np.ndarray:
    """
    Etapa opcional de pre-processamento.

    Justificativa tecnica:
    - Blur Gaussiano/Mediana reduzem ruido de alta frequencia e outliers, estabilizando o limiar.
    - Equalizacao de histograma aumenta contraste global, separando melhor fundo e objeto.
    Isso tende a melhorar a binarizacao quando a iluminacao e heterogenea.
    """
    img = gray.copy()

    if not usar_preprocessamento:
        return img

    blur_kernel = max(3, int(blur_kernel))
    if blur_kernel % 2 == 0:
        blur_kernel += 1

    if blur == "gaussian":
        img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)
    elif blur == "median":
        img = cv2.medianBlur(img, blur_kernel)
    elif blur == "bilateral":
        img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    if equalizar_histograma:
        img = cv2.equalizeHist(img)

    if usar_clahe:
        clahe_tile = max(2, int(clahe_tile))
        clahe = cv2.createCLAHE(clipLimit=max(0.1, float(clahe_clip)), tileGridSize=(clahe_tile, clahe_tile))
        img = clahe.apply(img)

    return img


def threshold_adaptativo_local(
    gray: np.ndarray,
    block_size: int = 35,
    offset: float = 10.0,
    metodo: str = "gaussian",
    polaridade: str = "above",
) -> Tuple[np.ndarray, MetadadosThreshold]:
    """
    Threshold Adaptativo (threshold_local).

    Fundamento matematico resumido:
    - Para cada pixel, calcula-se um limiar local T(x, y) em uma vizinhanca (janela).
    - A decisao binaria usa I(x, y) > T(x, y) - offset.
    - Isso lida melhor com variacoes de iluminacao espacial.

    Aplicacao principal:
    - Documentos, placas, cenas com sombra/iluminacao nao uniforme.
    """
    t0 = time.perf_counter()

    if block_size % 2 == 0:
        block_size += 1
    metodo = metodo if metodo in {"gaussian", "mean", "median"} else "gaussian"
    local_thr = threshold_local(gray, block_size=block_size, method=metodo, offset=offset)
    if polaridade == "below":
        binaria = (gray < local_thr).astype(np.uint8) * 255
    else:
        binaria = (gray > local_thr).astype(np.uint8) * 255

    meta = MetadadosThreshold("Threshold Adaptativo Local")
    meta.tempo_execucao_ms = (time.perf_counter() - t0) * 1000.0
    meta.limiares = [float(np.mean(local_thr))]
    meta.parametros = {
        "block_size": block_size,
        "offset": offset,
        "metodo": metodo,
        "polaridade": polaridade,
        "limiar_reportado": "media do mapa T(x,y)",
    }
    return binaria, meta


def threshold_multi_otsu(
    gray: np.ndarray,
    classes: int = 3,
    modo_saida: str = "levels",
    classe_alvo: int = 1,
) -> Tuple[np.ndarray, MetadadosThreshold]:
    """
    Multi-Otsu (threshold_multiotsu).

    Fundamento matematico resumido:
    - Generaliza Otsu para K classes, escolhendo K-1 limiares que maximizam
      a variancia entre classes e minimizam a variancia intra-classe.
    - Resultado: segmentacao multiclasse por niveis de intensidade.

    Aplicacao principal:
    - Separacao de regioes com diferentes materiais/tecidos/texturas em imagens.
    """
    t0 = time.perf_counter()
    classes = max(2, classes)
    limiares = threshold_multiotsu(gray, classes=classes)
    regioes = np.digitize(gray, bins=limiares)

    meta = MetadadosThreshold("Multi-Otsu")
    meta.tempo_execucao_ms = (time.perf_counter() - t0) * 1000.0
    meta.limiares = [float(v) for v in limiares.tolist()]
    meta.parametros = {
        "classes": classes,
        "modo_saida": modo_saida,
        "classe_alvo": classe_alvo,
    }

    if modo_saida == "class":
        classe_alvo = int(np.clip(classe_alvo, 0, classes - 1))
        return (regioes == classe_alvo).astype(np.uint8) * 255, meta

    if classes == 2:
        escala = np.array([0, 255], dtype=np.uint8)
    else:
        escala = np.linspace(0, 255, classes, dtype=np.uint8)
    return escala[regioes], meta


def threshold_range(
    gray: np.ndarray,
    baixo: int = 80,
    alto: int = 170,
    inverter: bool = False,
) -> Tuple[np.ndarray, MetadadosThreshold]:
    """
    Range Thresholding (limiarizacao por faixa).

    Fundamento matematico resumido:
    - Mantem pixel como foreground se baixo <= I(x, y) <= alto.
    - Equivale a uma mascara booleana por intervalo de intensidade.

    Aplicacao principal:
    - Isolamento de objetos com faixa tonal conhecida (ex.: deteccao simples de materiais).
    """
    t0 = time.perf_counter()
    baixo = int(np.clip(baixo, 0, 255))
    alto = int(np.clip(alto, 0, 255))
    if baixo > alto:
        baixo, alto = alto, baixo

    mascara = ((gray >= baixo) & (gray <= alto)).astype(np.uint8) * 255
    if inverter:
        mascara = 255 - mascara

    meta = MetadadosThreshold("Range Thresholding")
    meta.tempo_execucao_ms = (time.perf_counter() - t0) * 1000.0
    meta.limiares = [float(baixo), float(alto)]
    meta.parametros = {"baixo": baixo, "alto": alto, "inverter": inverter}
    return mascara, meta


def threshold_estatistico(
    gray: np.ndarray,
    k: float = 0.5,
    local: bool = False,
    janela_local: int = 31,
    polaridade: str = "above",
) -> Tuple[np.ndarray, MetadadosThreshold]:
    """
    Threshold Estatistico baseado em media e desvio padrao.

    Fundamento matematico resumido:
    - Global: T = mu + k*sigma, onde mu e media global e sigma desvio padrao global.
    - Local: T(x, y) = mu_w(x, y) + k*sigma_w(x, y), calculado por janela local.
    - Pixels sao classificados por I > T (ou outra regra de interesse).

    Aplicacao principal:
    - Cenarios em que estatisticas de intensidade ajudam a separar anomalias/objetos.
    """
    t0 = time.perf_counter()

    if not local:
        mu = float(np.mean(gray))
        sigma = float(np.std(gray))
        t = mu + k * sigma
        if polaridade == "below":
            binaria = (gray < t).astype(np.uint8) * 255
        else:
            binaria = (gray > t).astype(np.uint8) * 255

        meta = MetadadosThreshold("Threshold Estatistico (Global)")
        meta.tempo_execucao_ms = (time.perf_counter() - t0) * 1000.0
        meta.limiares = [float(t)]
        meta.parametros = {
            "mu": f"{mu:.4f}",
            "sigma": f"{sigma:.4f}",
            "k": k,
            "formula": "T = mu + k*sigma",
            "polaridade": polaridade,
        }
        return binaria, meta

    if janela_local % 2 == 0:
        janela_local += 1

    img = gray.astype(np.float32)
    media_local = cv2.GaussianBlur(img, (janela_local, janela_local), 0)
    media_quadratica = cv2.GaussianBlur(img * img, (janela_local, janela_local), 0)
    variancia_local = np.maximum(media_quadratica - media_local * media_local, 0.0)
    desvio_local = np.sqrt(variancia_local)
    t_local = media_local + k * desvio_local

    if polaridade == "below":
        binaria = (img < t_local).astype(np.uint8) * 255
    else:
        binaria = (img > t_local).astype(np.uint8) * 255

    meta = MetadadosThreshold("Threshold Estatistico (Local)")
    meta.tempo_execucao_ms = (time.perf_counter() - t0) * 1000.0
    meta.limiares = [float(np.mean(t_local))]
    meta.parametros = {
        "k": k,
        "janela_local": janela_local,
        "formula": "T(x,y) = mu_w(x,y) + k*sigma_w(x,y)",
        "polaridade": polaridade,
        "limiar_reportado": "media do mapa T_local(x,y)",
    }
    return binaria, meta


def _aplicar_threshold_global(gray: np.ndarray, limiar: float, nome: str) -> Tuple[np.ndarray, MetadadosThreshold]:
    """Aplica um threshold global simples usando I > T e gera metadados."""
    t0 = time.perf_counter()
    binaria = (gray > limiar).astype(np.uint8) * 255
    meta = MetadadosThreshold(nome)
    meta.tempo_execucao_ms = (time.perf_counter() - t0) * 1000.0
    meta.limiares = [float(limiar)]
    meta.parametros = {"regra": "I(x,y) > T"}
    return binaria, meta


def threshold_metodos_globais(
    gray: np.ndarray,
    metodos: Optional[set[str]] = None,
) -> dict[str, Tuple[np.ndarray, MetadadosThreshold]]:
    """Executa metodos globais extras: Yen, Triangle, Otsu, Minimum, Mean e ISODATA."""
    if metodos is None:
        metodos = {"yen", "triangle", "otsu", "minimum", "mean", "isodata"}

    resultados: dict[str, Tuple[np.ndarray, MetadadosThreshold]] = {}

    if "yen" in metodos:
        limiar_yen = float(threshold_yen(gray))
        resultados["yen"] = _aplicar_threshold_global(gray, limiar_yen, "Yen")

    if "triangle" in metodos:
        limiar_triangle = float(threshold_triangle(gray))
        resultados["triangle"] = _aplicar_threshold_global(gray, limiar_triangle, "Triangle")

    if "otsu" in metodos:
        limiar_otsu = float(threshold_otsu(gray))
        resultados["otsu"] = _aplicar_threshold_global(gray, limiar_otsu, "Otsu")

    # threshold_minimum pode falhar em histogramas sem dois picos bem definidos.
    if "minimum" in metodos:
        try:
            limiar_minimum = float(threshold_minimum(gray))
            resultados["minimum"] = _aplicar_threshold_global(gray, limiar_minimum, "Minimum")
        except Exception as exc:
            fallback = float(np.mean(gray))
            binaria, meta = _aplicar_threshold_global(gray, fallback, "Minimum (fallback mean)")
            meta.parametros["erro_minimum"] = str(exc)
            resultados["minimum"] = (binaria, meta)

    if "mean" in metodos:
        limiar_mean = float(np.mean(gray))
        resultados["mean"] = _aplicar_threshold_global(gray, limiar_mean, "Mean")

    if "isodata" in metodos:
        limiar_isodata = float(threshold_isodata(gray))
        resultados["isodata"] = _aplicar_threshold_global(gray, limiar_isodata, "ISODATA")

    return resultados


def salvar_resultados_por_metodo(
    titulo_base: str,
    resultados: dict[str, Tuple[np.ndarray, MetadadosThreshold]],
    pasta_saida: str,
    largura_px: int = 1920,
    altura_px: int = 1080,
) -> None:
    """Salva imagem de cada metodo selecionado em arquivo separado."""
    os.makedirs(pasta_saida, exist_ok=True)
    base = os.path.splitext(os.path.basename(titulo_base))[0]
    for nome_metodo, (img, _) in resultados.items():
        pronta = preparar_imagem_saida(img, largura_px=largura_px, altura_px=altura_px)
        caminho = os.path.join(pasta_saida, f"{base}_{nome_metodo}.png")
        cv2.imwrite(caminho, pronta)


def salvar_histogramas_individuais(
    titulo_base: str,
    gray_base: np.ndarray,
    resultados: dict[str, Tuple[np.ndarray, MetadadosThreshold]],
    pasta_saida: str,
) -> None:
    """Salva um PNG de histograma para cada metodo selecionado."""
    os.makedirs(pasta_saida, exist_ok=True)
    base = os.path.splitext(os.path.basename(titulo_base))[0]

    for nome_metodo, (img, meta) in resultados.items():
        fig, axs = plt.subplots(1, 2, figsize=(12, 4), facecolor="white")
        axs[0].hist(gray_base.ravel(), bins=32, range=(0, 255), color="#1f77b4")
        axs[0].set_title("Entrada")
        axs[0].set_xlabel("Intensidade")
        axs[0].set_ylabel("Pixels")

        axs[1].hist(img.ravel(), bins=32, range=(0, 255), color="#ff7f0e")
        limiar_txt = f"T={meta.limiares[0]:.2f}" if meta.limiares else "T=n/a"
        axs[1].set_title(f"{nome_metodo.upper()} ({limiar_txt})")
        axs[1].set_xlabel("Intensidade")
        axs[1].set_ylabel("Pixels")

        fig.suptitle(f"Histograma Individual - {nome_metodo.upper()} ({base})", fontweight="bold")
        fig.tight_layout(rect=[0, 0.02, 1, 0.92])
        caminho_saida = os.path.join(pasta_saida, f"{base}_hist_{nome_metodo}.png")
        fig.savefig(caminho_saida, dpi=120)
        plt.close(fig)


def mostrar_histogramas_filtros(
    titulo_base: str,
    gray_base: np.ndarray,
    resultados: dict[str, Tuple[np.ndarray, MetadadosThreshold]],
    pasta_saida: str,
    salvar_figura: bool = True,
    exibir: bool = False,
) -> None:
    """Exibe/salva painel com histogramas da entrada e de todos os filtros selecionados."""
    os.makedirs(pasta_saida, exist_ok=True)
    base = os.path.splitext(os.path.basename(titulo_base))[0]
    nomes = list(resultados.keys())
    total = 1 + len(nomes)
    cols = 4
    rows = int(np.ceil(total / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), facecolor="white")
    axs = np.array(axs).reshape(-1)

    axs[0].hist(gray_base.ravel(), bins=32, range=(0, 255), color="#1f77b4")
    axs[0].set_title("Entrada", fontweight="bold")

    for i, nome in enumerate(nomes, start=1):
        img, meta = resultados[nome]
        axs[i].hist(img.ravel(), bins=32, range=(0, 255), color="#ff7f0e")
        limiar_txt = f"T={meta.limiares[0]:.2f}" if meta.limiares else "T=n/a"
        axs[i].set_title(f"{nome.upper()} ({limiar_txt})", fontsize=9)

    for j in range(total, len(axs)):
        axs[j].axis("off")

    fig.suptitle(f"Histogramas de Todos os Filtros - {base}", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    if salvar_figura:
        caminho_saida = os.path.join(pasta_saida, f"{base}_histogramas_todos_filtros.png")
        fig.savefig(caminho_saida, dpi=120)

    if exibir:
        plt.show()
    else:
        plt.close(fig)


def mostrar_comparativo_selecionados(
    original_rgb: np.ndarray,
    resultados: dict[str, Tuple[np.ndarray, MetadadosThreshold]],
    titulo_base: str,
    pasta_saida: str,
    salvar_figura: bool = True,
    exibir: bool = False,
    dpi_saida: int = 100,
    largura_px: int = 1920,
    altura_px: int = 1080,
) -> None:
    """Exibe/salva comparativo para todos os metodos selecionados."""
    os.makedirs(pasta_saida, exist_ok=True)
    nomes = list(resultados.keys())
    total = 1 + len(nomes)
    cols = min(4, total)
    rows = int(np.ceil(total / cols))

    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(largura_px / dpi_saida, altura_px / dpi_saida),
        facecolor="white",
    )
    axs = np.array(axs).reshape(-1)

    axs[0].imshow(original_rgb)
    axs[0].set_title("Original", fontweight="bold")
    axs[0].axis("off")

    for i, nome in enumerate(nomes, start=1):
        img, _ = resultados[nome]
        axs[i].imshow(img, cmap="gray")
        axs[i].set_title(nome.upper(), fontsize=10, fontweight="bold")
        axs[i].axis("off")

    for j in range(total, len(axs)):
        axs[j].axis("off")

    fig.suptitle(f"Comparativo de Filtros Selecionados - {os.path.basename(titulo_base)}", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])

    if salvar_figura:
        nome_base = os.path.splitext(os.path.basename(titulo_base))[0]
        caminho_saida = os.path.join(pasta_saida, f"{nome_base}_comparativo_selecionados.png")
        fig.savefig(caminho_saida, dpi=dpi_saida)

    if exibir:
        plt.show()
    else:
        plt.close(fig)


def salvar_resultados_metodos_globais(
    titulo_base: str,
    resultados: dict[str, Tuple[np.ndarray, MetadadosThreshold]],
    pasta_saida: str,
    largura_px: int = 1920,
    altura_px: int = 1080,
) -> None:
    """Salva saidas dos metodos globais extras em arquivos separados."""
    os.makedirs(pasta_saida, exist_ok=True)
    base = os.path.splitext(os.path.basename(titulo_base))[0]
    for nome_metodo, (img, _) in resultados.items():
        pronta = preparar_imagem_saida(img, largura_px=largura_px, altura_px=altura_px)
        caminho = os.path.join(pasta_saida, f"{base}_{nome_metodo}.png")
        cv2.imwrite(caminho, pronta)


def salvar_histogramas_metodos_globais(
    titulo_base: str,
    gray_base: np.ndarray,
    resultados: dict[str, Tuple[np.ndarray, MetadadosThreshold]],
    pasta_saida: str,
) -> None:
    """Salva painel com histogramas da entrada e das saidas dos metodos globais extras."""
    os.makedirs(pasta_saida, exist_ok=True)
    base = os.path.splitext(os.path.basename(titulo_base))[0]

    fig, axs = plt.subplots(2, 4, figsize=(16, 8), facecolor="white")
    axs = axs.ravel()

    axs[0].hist(gray_base.ravel(), bins=32, range=(0, 255), color="#1f77b4")
    axs[0].set_title("Entrada", fontweight="bold")

    nomes = list(resultados.keys())
    for i, nome in enumerate(nomes, start=1):
        img, meta = resultados[nome]
        axs[i].hist(img.ravel(), bins=32, range=(0, 255), color="#ff7f0e")
        limiar_txt = f"T={meta.limiares[0]:.2f}" if meta.limiares else "T=n/a"
        axs[i].set_title(f"{nome.upper()} ({limiar_txt})", fontsize=9)

    for j in range(len(nomes) + 1, len(axs)):
        axs[j].axis("off")

    fig.suptitle(f"Histogramas - Metodos Globais Extras ({base})", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    caminho_saida = os.path.join(pasta_saida, f"{base}_histogramas_metodos_globais.png")
    fig.savefig(caminho_saida, dpi=120)
    plt.close(fig)


def preparar_imagem_saida(img: np.ndarray, largura_px: int = 1920, altura_px: int = 1080) -> np.ndarray:
    """Redimensiona mantendo preenchimento para gerar saida final exata 1920x1080."""
    largura_px = max(320, int(largura_px))
    altura_px = max(180, int(altura_px))

    if img.ndim == 2:
        h, w = img.shape
        canais = 1
    else:
        h, w = img.shape[:2]
        canais = img.shape[2]

    escala = min(largura_px / w, altura_px / h)
    novo_w = max(1, int(w * escala))
    novo_h = max(1, int(h * escala))

    interp = cv2.INTER_AREA if escala < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (novo_w, novo_h), interpolation=interp)

    if canais == 1:
        canvas = np.zeros((altura_px, largura_px), dtype=np.uint8)
    else:
        canvas = np.zeros((altura_px, largura_px, canais), dtype=np.uint8)

    off_x = (largura_px - novo_w) // 2
    off_y = (altura_px - novo_h) // 2
    canvas[off_y : off_y + novo_h, off_x : off_x + novo_w] = resized
    return canvas


def salvar_resultados_individuais(
    titulo_base: str,
    original_rgb: np.ndarray,
    adaptativo: np.ndarray,
    multi_otsu: np.ndarray,
    faixa: np.ndarray,
    estatistico: np.ndarray,
    pasta_saida: str = "resultados",
    largura_px: int = 1920,
    altura_px: int = 1080,
) -> None:
    """Salva cada resultado em arquivo separado, sem textos, em 1920x1080."""
    os.makedirs(pasta_saida, exist_ok=True)
    base = os.path.splitext(os.path.basename(titulo_base))[0]

    original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    saidas = {
        "original": original_bgr,
        "adaptativo": adaptativo,
        "multi_otsu": multi_otsu,
        "range": faixa,
        "estatistico": estatistico,
    }

    for nome, img in saidas.items():
        pronta = preparar_imagem_saida(img, largura_px=largura_px, altura_px=altura_px)
        caminho = os.path.join(pasta_saida, f"{base}_{nome}.png")
        cv2.imwrite(caminho, pronta)


def mostrar_resultados_lado_a_lado(
    original_rgb: np.ndarray,
    adaptativo: np.ndarray,
    multi_otsu: np.ndarray,
    faixa: np.ndarray,
    estatistico: np.ndarray,
    titulo_base: str,
    salvar_figura: bool = True,
    pasta_saida: str = "resultados",
    dpi_saida: int = 100,
    largura_px: int = 1920,
    altura_px: int = 1080,
) -> None:
    """Exibe e opcionalmente salva uma comparacao pronta para apresentacao."""
    largura_px = max(320, int(largura_px))
    altura_px = max(180, int(altura_px))
    dpi_saida = max(50, int(dpi_saida))
    fig = plt.figure(figsize=(largura_px / dpi_saida, altura_px / dpi_saida), facecolor="white")

    plt.subplot(1, 5, 1)
    plt.imshow(original_rgb)
    plt.title("Original", fontsize=16, fontweight="bold")
    plt.axis("off")

    pct_adaptativo = calcular_percentual_foreground(adaptativo)
    pct_multi = calcular_percentual_foreground(multi_otsu)
    pct_faixa = calcular_percentual_foreground(faixa)
    pct_estat = calcular_percentual_foreground(estatistico)

    plt.subplot(1, 5, 2)
    plt.imshow(adaptativo, cmap="gray")
    plt.title(f"Adaptativo\nFG: {pct_adaptativo:.1f}%", fontsize=14, fontweight="bold")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(multi_otsu, cmap="gray")
    plt.title(f"Multi-Otsu\nFG: {pct_multi:.1f}%", fontsize=14, fontweight="bold")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(faixa, cmap="gray")
    plt.title(f"Range\nFG: {pct_faixa:.1f}%", fontsize=14, fontweight="bold")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(estatistico, cmap="gray")
    plt.title(f"Estatistico\nFG: {pct_estat:.1f}%", fontsize=14, fontweight="bold")
    plt.axis("off")

    plt.suptitle(
        f"Comparacao de Thresholds - {os.path.basename(titulo_base)}",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.02,
        "Metrica exibida: percentual de foreground (FG) em cada resultado.",
        ha="center",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 0.90])

    if salvar_figura:
        os.makedirs(pasta_saida, exist_ok=True)
        nome_base = os.path.splitext(os.path.basename(titulo_base))[0]
        caminho_saida = os.path.join(pasta_saida, f"{nome_base}_comparativo_thresholds.png")
        fig.savefig(caminho_saida, dpi=dpi_saida)
        print(
            f"Figura salva para slides em: {caminho_saida} "
            f"({largura_px}x{altura_px}, aspecto 16:9)"
        )

    plt.show()


def _calcular_entropia(img: np.ndarray) -> float:
    """Calcula a entropia de Shannon da imagem em tons de cinza."""
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel().astype(np.float64)
    total = float(np.sum(hist))
    if total <= 0:
        return 0.0
    p = hist / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def mostrar_analise_avancada(
    gray_base: np.ndarray,
    adaptativo: np.ndarray,
    multi_otsu: np.ndarray,
    faixa: np.ndarray,
    estatistico: np.ndarray,
    meta_adaptativo: MetadadosThreshold,
    meta_multi: MetadadosThreshold,
    meta_faixa: MetadadosThreshold,
    meta_estatistico: MetadadosThreshold,
    titulo_base: str,
    salvar_figura: bool = False,
    exibir: bool = True,
    pasta_saida: str = "resultados",
    dpi_saida: int = 100,
    largura_px: int = 1920,
    altura_px: int = 1080,
) -> None:
    """Gera painel analitico com histogramas e metricas para apoiar a apresentacao."""
    tecnicas = ["Adaptativo", "Multi-Otsu", "Range", "Estatistico"]
    resultados = [adaptativo, multi_otsu, faixa, estatistico]
    metadados = [meta_adaptativo, meta_multi, meta_faixa, meta_estatistico]

    fg_pct = [calcular_percentual_foreground(img) for img in resultados]
    contraste = [float(np.std(img)) for img in resultados]
    entropias = [_calcular_entropia(img) for img in resultados]
    tempos = [m.tempo_execucao_ms for m in metadados]

    largura_px = max(320, int(largura_px))
    altura_px = max(180, int(altura_px))
    dpi_saida = max(50, int(dpi_saida))

    fig, axs = plt.subplots(2, 3, figsize=(largura_px / dpi_saida, altura_px / dpi_saida), facecolor="white")
    axs = axs.ravel()

    axs[0].hist(gray_base.ravel(), bins=32, range=(0, 255), color="#1f77b4", alpha=0.9)
    axs[0].set_title("Histograma da Entrada", fontweight="bold")
    axs[0].set_xlabel("Intensidade")
    axs[0].set_ylabel("Quantidade de pixels")

    for nome, img in zip(tecnicas, resultados):
        axs[1].hist(img.ravel(), bins=32, range=(0, 255), alpha=0.35, label=nome)
    axs[1].set_title("Histogramas das Saidas", fontweight="bold")
    axs[1].set_xlabel("Intensidade")
    axs[1].set_ylabel("Quantidade de pixels")
    axs[1].legend(fontsize=9)

    axs[2].bar(tecnicas, fg_pct, color=["#2ca02c", "#ff7f0e", "#9467bd", "#d62728"])
    axs[2].set_title("Foreground (%)", fontweight="bold")
    axs[2].set_ylabel("% de pixels")
    axs[2].set_ylim(0, 100)

    axs[3].bar(tecnicas, tempos, color=["#17becf", "#bcbd22", "#8c564b", "#7f7f7f"])
    axs[3].set_title("Tempo de Execucao", fontweight="bold")
    axs[3].set_ylabel("ms")

    axs[4].bar(tecnicas, contraste, color=["#1f77b4", "#ff9896", "#98df8a", "#c5b0d5"])
    axs[4].set_title("Contraste (Desvio Padrao)", fontweight="bold")
    axs[4].set_ylabel("std")

    axs[5].bar(tecnicas, entropias, color=["#aec7e8", "#ffbb78", "#98df8a", "#c49c94"])
    axs[5].set_title("Entropia de Shannon", fontweight="bold")
    axs[5].set_ylabel("bits")

    fig.suptitle(f"Analise Avancada - {os.path.basename(titulo_base)}", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    if salvar_figura:
        os.makedirs(pasta_saida, exist_ok=True)
        nome_base = os.path.splitext(os.path.basename(titulo_base))[0]
        caminho_saida = os.path.join(pasta_saida, f"{nome_base}_analise_metricas.png")
        fig.savefig(caminho_saida, dpi=dpi_saida)
        print(f"Painel analitico salvo em: {caminho_saida}")

    if exibir:
        plt.show()
    else:
        plt.close(fig)


def processar_imagem(args: argparse.Namespace) -> None:
    rgb, gray, caminho = carregar_imagem(args.image)
    dir_img = os.path.join(args.figure_dir, "img")
    dir_hist = os.path.join(args.figure_dir, "hist")
    dir_comp = os.path.join(args.figure_dir, "comp")

    gray_proc = preprocessar(
        gray,
        usar_preprocessamento=args.preprocess,
        blur=args.blur,
        blur_kernel=args.blur_kernel,
        equalizar_histograma=args.equalize,
        usar_clahe=args.clahe,
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
    )

    adaptativo, meta_adaptativo = threshold_adaptativo_local(
        gray_proc,
        block_size=args.block_size,
        offset=args.offset,
        metodo=args.adaptive_method,
        polaridade=args.adaptive_polarity,
    )
    multi, meta_multi = threshold_multi_otsu(
        gray_proc,
        classes=args.classes,
        modo_saida=args.multi_mode,
        classe_alvo=args.multi_target_class,
    )
    faixa, meta_faixa = threshold_range(gray_proc, baixo=args.range_low, alto=args.range_high, inverter=args.range_invert)
    estatistico, meta_estatistico = threshold_estatistico(
        gray_proc,
        k=args.k,
        local=args.stat_local,
        janela_local=args.stat_window,
        polaridade=args.stat_polarity,
    )

    print(meta_adaptativo)
    print(meta_multi)
    print(meta_faixa)
    print(meta_estatistico)

    extras = threshold_metodos_globais(gray_proc)
    resultados_todos: dict[str, Tuple[np.ndarray, MetadadosThreshold]] = {
        "adaptativo": (adaptativo, meta_adaptativo),
        "multi_otsu": (multi, meta_multi),
        "range": (faixa, meta_faixa),
        "estatistico": (estatistico, meta_estatistico),
    }
    resultados_todos.update(extras)
    for _nome, (_, meta_extra) in extras.items():
        print(meta_extra)

    if args.save_individual:
        salvar_resultados_individuais(
            titulo_base=caminho,
            original_rgb=rgb,
            adaptativo=adaptativo,
            multi_otsu=multi,
            faixa=faixa,
            estatistico=estatistico,
            pasta_saida=dir_img,
            largura_px=args.figure_width,
            altura_px=args.figure_height,
        )

    mostrar_resultados_lado_a_lado(
        original_rgb=rgb,
        adaptativo=adaptativo,
        multi_otsu=multi,
        faixa=faixa,
        estatistico=estatistico,
        titulo_base=caminho,
        salvar_figura=args.save_figure,
        pasta_saida=dir_comp,
        dpi_saida=args.figure_dpi,
        largura_px=args.figure_width,
        altura_px=args.figure_height,
    )

    if args.show_analysis or args.save_analysis:
        mostrar_analise_avancada(
            gray_base=gray_proc,
            adaptativo=adaptativo,
            multi_otsu=multi,
            faixa=faixa,
            estatistico=estatistico,
            meta_adaptativo=meta_adaptativo,
            meta_multi=meta_multi,
            meta_faixa=meta_faixa,
            meta_estatistico=meta_estatistico,
            titulo_base=caminho,
            salvar_figura=args.save_analysis,
            exibir=args.show_analysis,
            pasta_saida=dir_hist,
            dpi_saida=args.figure_dpi,
            largura_px=args.figure_width,
            altura_px=args.figure_height,
        )

    if args.save_extra_methods:
        salvar_resultados_metodos_globais(
            titulo_base=caminho,
            resultados=extras,
            pasta_saida=dir_img,
            largura_px=args.figure_width,
            altura_px=args.figure_height,
        )

    if args.save_histograms:
        mostrar_histogramas_filtros(
            titulo_base=caminho,
            gray_base=gray_proc,
            resultados=resultados_todos,
            pasta_saida=dir_hist,
            salvar_figura=True,
            exibir=False,
        )
        salvar_histogramas_individuais(
            titulo_base=caminho,
            gray_base=gray_proc,
            resultados=resultados_todos,
            pasta_saida=dir_hist,
        )


def aplicar_metodo(frame_gray: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Aplica o metodo escolhido no frame de video."""
    if args.video_method == "adaptive":
        saida, _ = threshold_adaptativo_local(
            frame_gray,
            block_size=args.block_size,
            offset=args.offset,
            metodo=args.adaptive_method,
            polaridade=args.adaptive_polarity,
        )
        return saida
    if args.video_method == "multiotsu":
        saida, _ = threshold_multi_otsu(
            frame_gray,
            classes=args.classes,
            modo_saida=args.multi_mode,
            classe_alvo=args.multi_target_class,
        )
        return saida
    if args.video_method == "range":
        saida, _ = threshold_range(frame_gray, baixo=args.range_low, alto=args.range_high, inverter=args.range_invert)
        return saida
    saida, _ = threshold_estatistico(
        frame_gray,
        k=args.k,
        local=args.stat_local,
        janela_local=args.stat_window,
        polaridade=args.stat_polarity,
    )
    return saida


def processar_video(args: argparse.Namespace) -> None:
    caminho_video = args.video
    if not caminho_video:
        caminho_video = selecionar_arquivo(
            "Selecione um video",
            (
                ("Videos", "*.mp4;*.avi;*.mov;*.mkv"),
                ("Todos os arquivos", "*.*"),
            ),
        )

    if not caminho_video or not os.path.exists(caminho_video):
        raise FileNotFoundError("Video nao encontrado. Informe um caminho valido.")

    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        raise RuntimeError("Nao foi possivel abrir o video.")

    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, fps, (largura, altura), True)

    print("Pressione 'q' para encerrar a exibicao em tempo real.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_proc = preprocessar(
            gray,
            usar_preprocessamento=args.preprocess,
            blur=args.blur,
            blur_kernel=args.blur_kernel,
            equalizar_histograma=args.equalize,
            usar_clahe=args.clahe,
            clahe_clip=args.clahe_clip,
            clahe_tile=args.clahe_tile,
        )

        saida = aplicar_metodo(gray_proc, args)
        saida_bgr = cv2.cvtColor(saida, cv2.COLOR_GRAY2BGR)

        if writer is not None:
            writer.write(saida_bgr)

        if args.show_video:
            cv2.imshow("Threshold em Video", saida)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    if args.show_video:
        cv2.destroyAllWindows()


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Processamento Digital de Imagens: Thresholding com skimage, OpenCV e Matplotlib"
    )

    parser.add_argument("--mode", choices=["image", "video"], default="image", help="Modo de execucao")

    # Upload/entrada
    parser.add_argument("--image", type=str, default=None, help="Caminho da imagem")
    parser.add_argument("--video", type=str, default=None, help="Caminho do video")

    # Pre-processamento opcional
    parser.add_argument("--preprocess", action="store_true", help="Ativa pre-processamento")
    parser.add_argument(
        "--blur",
        choices=["none", "gaussian", "median", "bilateral"],
        default="gaussian",
        help="Tipo de blur no pre-processamento",
    )
    parser.add_argument("--blur-kernel", dest="blur_kernel", type=int, default=5, help="Kernel dos blurs gaussian/median")
    parser.add_argument("--equalize", action="store_true", help="Ativa equalizacao de histograma")
    parser.add_argument("--clahe", action="store_true", help="Ativa equalizacao local CLAHE")
    parser.add_argument("--clahe-clip", dest="clahe_clip", type=float, default=2.0, help="Clip limit do CLAHE")
    parser.add_argument("--clahe-tile", dest="clahe_tile", type=int, default=8, help="Tile size do CLAHE")

    # Parametros de threshold
    parser.add_argument("--block-size", dest="block_size", type=int, default=35, help="Janela do adaptativo")
    parser.add_argument("--offset", type=float, default=10.0, help="Offset do adaptativo")
    parser.add_argument(
        "--adaptive-method",
        dest="adaptive_method",
        choices=["gaussian", "mean", "median"],
        default="gaussian",
        help="Metodo do threshold adaptativo",
    )
    parser.add_argument(
        "--adaptive-polarity",
        dest="adaptive_polarity",
        choices=["above", "below"],
        default="above",
        help="Polaridade do threshold adaptativo",
    )

    parser.add_argument("--classes", type=int, default=3, help="Numero de classes no Multi-Otsu")
    parser.add_argument(
        "--multi-mode",
        dest="multi_mode",
        choices=["levels", "class"],
        default="levels",
        help="Saida do Multi-Otsu: niveis de cinza ou mascara de classe",
    )
    parser.add_argument(
        "--multi-target-class",
        dest="multi_target_class",
        type=int,
        default=1,
        help="Classe alvo quando --multi-mode class",
    )

    parser.add_argument("--range-low", dest="range_low", type=int, default=80, help="Limite inferior range")
    parser.add_argument("--range-high", dest="range_high", type=int, default=170, help="Limite superior range")
    parser.add_argument("--range-invert", dest="range_invert", action="store_true", help="Inverte mascara do range")

    parser.add_argument("--k", type=float, default=0.5, help="Fator k do threshold estatistico")
    parser.add_argument("--stat-local", dest="stat_local", action="store_true", help="Usa estatistico local")
    parser.add_argument("--stat-window", dest="stat_window", type=int, default=31, help="Janela estatistico local")
    parser.add_argument(
        "--stat-polarity",
        dest="stat_polarity",
        choices=["above", "below"],
        default="above",
        help="Polaridade do threshold estatistico",
    )

    # Video
    parser.add_argument(
        "--video-method",
        choices=["adaptive", "multiotsu", "range", "statistical"],
        default="adaptive",
        help="Metodo de threshold aplicado no video",
    )
    parser.add_argument("--show-video", action="store_true", help="Exibe video em tempo real")
    parser.add_argument("--output", type=str, default=None, help="Salva o video processado")

    # Saida de apresentacao (imagem)
    parser.add_argument(
        "--save-figure",
        dest="save_figure",
        action="store_true",
        default=False,
        help="Salva automaticamente a figura comparativa para slides",
    )
    parser.add_argument(
        "--no-save-figure",
        dest="save_figure",
        action="store_false",
        help="Desativa o salvamento automatico da figura comparativa",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default="resultados",
        help="Diretorio para salvar a figura comparativa",
    )
    parser.add_argument(
        "--figure-dpi",
        type=int,
        default=100,
        help="DPI da figura salva (usado para obter dimensao exata em pixels)",
    )
    parser.add_argument(
        "--figure-width",
        type=int,
        default=1920,
        help="Largura da figura salva em pixels",
    )
    parser.add_argument(
        "--figure-height",
        type=int,
        default=1080,
        help="Altura da figura salva em pixels",
    )
    parser.add_argument(
        "--save-individual",
        dest="save_individual",
        action="store_true",
        default=True,
        help="Salva cada resultado em arquivo separado, sem textos",
    )
    parser.add_argument(
        "--no-save-individual",
        dest="save_individual",
        action="store_false",
        help="Desativa salvamento individual",
    )
    parser.add_argument(
        "--show-analysis",
        dest="show_analysis",
        action="store_true",
        default=True,
        help="Exibe painel de analise avancada (histogramas e metricas)",
    )
    parser.add_argument(
        "--no-show-analysis",
        dest="show_analysis",
        action="store_false",
        help="Desativa exibicao do painel de analise avancada",
    )
    parser.add_argument(
        "--save-analysis",
        dest="save_analysis",
        action="store_true",
        default=False,
        help="Salva painel de analise avancada em arquivo PNG",
    )
    parser.add_argument(
        "--save-extra-methods",
        dest="save_extra_methods",
        action="store_true",
        default=True,
        help="Salva saidas dos metodos extras: Yen, Triangle, Otsu, Minimum, Mean, ISODATA",
    )
    parser.add_argument(
        "--no-save-extra-methods",
        dest="save_extra_methods",
        action="store_false",
        help="Desativa salvamento dos metodos extras",
    )
    parser.add_argument(
        "--save-histograms",
        dest="save_histograms",
        action="store_true",
        default=True,
        help="Salva painel de histogramas dos metodos globais extras",
    )
    parser.add_argument(
        "--no-save-histograms",
        dest="save_histograms",
        action="store_false",
        help="Desativa salvamento do painel de histogramas",
    )

    return parser


def main() -> None:
    parser = construir_parser()
    args = parser.parse_args()

    if args.blur == "none":
        args.blur = ""

    if args.mode == "image":
        processar_imagem(args)
    else:
        processar_video(args)


if __name__ == "__main__":
    main()
