import os
from typing import Optional, Tuple, Dict, Set

import cv2
import numpy as np
from skimage import color, util


COR_PARA_HUE = {
    "vermelho": 0,
    "laranja": 15,
    "amarelo": 30,
    "verde": 60,
    "ciano": 90,
    "azul": 120,
    "magenta": 150,
}


def _to_float_rgb(image_bgr: np.ndarray) -> np.ndarray:
    """Converte BGR uint8 para RGB float em [0, 1]."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return util.img_as_float(rgb)


def _to_bgr_uint8(image_rgb_float: np.ndarray) -> np.ndarray:
    """Converte RGB float em [0, 1] para BGR uint8."""
    rgb_u8 = util.img_as_ubyte(np.clip(image_rgb_float, 0.0, 1.0))
    return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)


def shift_color(
    image_bgr: np.ndarray,
    source_hue: float,
    target_hue: float,
    tolerance: float = 18.0,
    feather: float = 10.0,
) -> Tuple[np.ndarray, dict]:
    """
    Rotaciona o hue de pixels proximos da cor de origem para a cor de destino.

    A operacao preserva Saturation (S) e Value (V), alterando apenas Hue (H)
    no espaco HSV da skimage (h em [0, 1]).
    """
    rgb_f = _to_float_rgb(image_bgr)
    hsv = color.rgb2hsv(rgb_f)

    h = hsv[:, :, 0]
    source = float(source_hue % 180.0) / 180.0
    target = float(target_hue % 180.0) / 180.0
    tol = max(0.0, float(tolerance) / 180.0)
    fea = max(0.0, float(feather) / 180.0)

    d = np.abs(h - source)
    d = np.minimum(d, 1.0 - d)

    if fea > 0:
        w = np.clip((tol + fea - d) / fea, 0.0, 1.0)
        w[d <= tol] = 1.0
    else:
        w = (d <= tol).astype(np.float32)

    delta = ((target - source + 0.5) % 1.0) - 0.5
    h_new = (h + delta * w) % 1.0

    hsv_new = hsv.copy()
    hsv_new[:, :, 0] = h_new
    out_rgb = color.hsv2rgb(hsv_new)

    meta = {
        "source_hue": float(source_hue),
        "target_hue": float(target_hue),
        "tolerance": float(tolerance),
        "feather": float(feather),
    }
    return _to_bgr_uint8(out_rgb), meta


def _aplicar_blur_progressivo(img: np.ndarray, kernel_size: int) -> np.ndarray:
    """Aplica blur Gaussiano progressivo."""
    if kernel_size < 3:
        return img
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def _aplicar_rotacao_suave(img: np.ndarray, angle: float) -> np.ndarray:
    """Aplica rotação suave ao redor do centro."""
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _aplicar_distorcao_onda(img: np.ndarray, amplitude: float = 5.0) -> np.ndarray:
    """Aplica distorção tipo onda ao sinal."""
    h, w = img.shape[:2]
    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x, y)
    
    # Cria ondas senoidais
    offset_x = (amplitude * np.sin(2 * np.pi * yy / h)).astype(np.float32)
    offset_y = (amplitude * np.cos(2 * np.pi * xx / w)).astype(np.float32)
    
    xx_new = (xx + offset_x).astype(np.float32)
    yy_new = (yy + offset_y).astype(np.float32)
    
    xx_new = np.clip(xx_new, 0, w - 1)
    yy_new = np.clip(yy_new, 0, h - 1)
    
    return cv2.remap(img, xx_new, yy_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _aplicar_brilho_contraste(img: np.ndarray, brilho: float, contraste: float) -> np.ndarray:
    """Ajusta brilho e contraste."""
    img_f = img.astype(np.float32) / 255.0
    img_f = img_f * contraste + brilho
    img_f = np.clip(img_f, 0, 1) * 255.0
    return img_f.astype(np.uint8)


def _aplicar_saturacao(img: np.ndarray, saturacao: float) -> np.ndarray:
    """Ajusta saturação da imagem."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturacao, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _aplicar_negativo(img: np.ndarray, intensidade: float) -> np.ndarray:
    """Mistura a imagem com sua inversão cromática."""
    alpha = float(np.clip(intensidade, 0.0, 1.0))
    base = img.astype(np.float32)
    invertida = 255.0 - base
    return np.clip(base * (1.0 - alpha) + invertida * alpha, 0, 255).astype(np.uint8)


def _aplicar_polarizador(img: np.ndarray, intensidade: float) -> np.ndarray:
    """Cria um efeito de polarização com mais contraste e saturação local."""
    alpha = float(np.clip(intensidade, 0.0, 1.0))
    if alpha <= 0.0:
        return img

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + 0.9 * alpha), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1.0 - 0.2 * alpha), 0, 255)
    polar = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    h, w = img.shape[:2]
    grad_x = np.linspace(0.88, 1.12, w, dtype=np.float32)[None, :]
    grad_y = (0.94 + 0.06 * np.cos(np.linspace(0.0, np.pi * 2.0, h, dtype=np.float32)))[:, None]
    mascara = np.clip(grad_x * grad_y, 0.75, 1.15)
    polar *= mascara[..., None]
    return np.clip(polar, 0, 255).astype(np.uint8)


def _aplicar_sepia(img: np.ndarray, intensidade: float) -> np.ndarray:
    """Aplica um tom sépia com mistura gradual."""
    alpha = float(np.clip(intensidade, 0.0, 1.0))
    if alpha <= 0.0:
        return img

    kernel = np.array(
        [
            [0.131, 0.534, 0.272],
            [0.168, 0.686, 0.349],
            [0.189, 0.769, 0.393],
        ],
        dtype=np.float32,
    )
    sepia = cv2.transform(img.astype(np.float32), kernel)
    return np.clip(img.astype(np.float32) * (1.0 - alpha) + sepia * alpha, 0, 255).astype(np.uint8)


def _aplicar_nitidez(img: np.ndarray, intensidade: float) -> np.ndarray:
    """Realça bordas e microcontraste."""
    alpha = float(np.clip(intensidade, 0.0, 1.0))
    if alpha <= 0.0:
        return img

    kernel = np.array(
        [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    sharp = cv2.filter2D(img, -1, kernel)
    return np.clip(img.astype(np.float32) * (1.0 - alpha) + sharp.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def _aplicar_posterizacao(img: np.ndarray, intensidade: float) -> np.ndarray:
    """Reduz a quantidade de níveis de cor para um visual mais bruto."""
    alpha = float(np.clip(intensidade, 0.0, 1.0))
    if alpha <= 0.0:
        return img

    niveis = int(np.clip(round(32 - 26 * alpha), 2, 32))
    passo = 256.0 / float(niveis)
    base = img.astype(np.float32)
    poster = np.floor(base / passo) * passo + (passo / 2.0)
    return np.clip(base * (1.0 - alpha) + poster * alpha, 0, 255).astype(np.uint8)


def _escala_efeito(percentual: float) -> float:
    """Converte um slider de 0 a 1000% em multiplicador prático."""
    return max(0.0, float(percentual) / 100.0)


def _intensidade_mistura(percentual: float, divisor: float = 4.0) -> float:
    """Normaliza um slider percentual para efeitos de mistura entre 0 e 1."""
    return float(np.clip(_escala_efeito(percentual) / float(divisor), 0.0, 1.0))


def _blend_mascarado(base: np.ndarray, efeito: np.ndarray, mascara: np.ndarray, alpha: float) -> np.ndarray:
    """Mistura o efeito apenas onde a mascara está ativa."""
    mascara_f = np.clip(mascara.astype(np.float32), 0.0, 1.0)[..., None]
    alpha_f = np.clip(float(alpha), 0.0, 1.0)
    return base * (1.0 - mascara_f * alpha_f) + efeito * (mascara_f * alpha_f)


def recortar_centro(image_bgr: np.ndarray, largura: int, altura: int) -> np.ndarray:
    """Recorta a imagem pelo centro sem preencher bordas."""
    h, w = image_bgr.shape[:2]
    largura = min(max(1, int(largura)), w)
    altura = min(max(1, int(altura)), h)
    x0 = max(0, (w - largura) // 2)
    y0 = max(0, (h - altura) // 2)
    return image_bgr[y0:y0 + altura, x0:x0 + largura].copy()


def aplicar_filtros_selecionados(
    image_bgr: np.ndarray,
    filtros: Optional[Set[str]] = None,
    intensidades: Optional[Dict[str, float]] = None,
    blur_kernel: int = 5,
    rotacao_angle: float = 15.0,
    onda_amplitude: float = 5.0,
    brilho: float = 0.0,
    contraste: float = 1.0,
    saturacao: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Aplica filtros selecionados à imagem.
    
    Filtros disponíveis: 'blur', 'rotacao', 'onda', 'brilho_contraste', 'saturacao',
    'negativo', 'polarizador', 'sepia', 'nitidez', 'posterizar'
    """
    if filtros is None:
        filtros = {"blur", "rotacao", "onda", "brilho_contraste", "saturacao", "negativo", "polarizador", "sepia", "nitidez", "posterizar"}
    if intensidades is None:
        intensidades = {}
    
    rng = np.random.default_rng(seed)
    result = image_bgr.copy()

    intensidade_blur = _escala_efeito(intensidades.get("blur", 100.0))
    intensidade_rotacao = _escala_efeito(intensidades.get("rotacao", 100.0))
    intensidade_onda = _escala_efeito(intensidades.get("onda", 100.0))
    intensidade_bc = _escala_efeito(intensidades.get("brilho_contraste", 100.0))
    intensidade_sat = _escala_efeito(intensidades.get("saturacao", 100.0))
    intensidade_negativo = _intensidade_mistura(intensidades.get("negativo", 100.0))
    intensidade_polarizador = _intensidade_mistura(intensidades.get("polarizador", 100.0))
    intensidade_sepia = _intensidade_mistura(intensidades.get("sepia", 100.0))
    intensidade_nitidez = _intensidade_mistura(intensidades.get("nitidez", 100.0))
    intensidade_posterizar = _intensidade_mistura(intensidades.get("posterizar", 100.0))
    
    if "blur" in filtros:
        kernel = int(max(1, blur_kernel * intensidade_blur))
        if kernel >= 3:
            result = _aplicar_blur_progressivo(result, kernel)
    
    if "rotacao" in filtros:
        angle = rng.uniform(-rotacao_angle, rotacao_angle) * intensidade_rotacao
        result = _aplicar_rotacao_suave(result, angle)
    
    if "onda" in filtros:
        result = _aplicar_distorcao_onda(result, amplitude=onda_amplitude * intensidade_onda)
    
    if "brilho_contraste" in filtros:
        result = _aplicar_brilho_contraste(
            result,
            brilho * intensidade_bc,
            1.0 + (contraste - 1.0) * intensidade_bc,
        )
    
    if "saturacao" in filtros:
        result = _aplicar_saturacao(result, 1.0 + (saturacao - 1.0) * intensidade_sat)

    if "polarizador" in filtros:
        result = _aplicar_polarizador(result, intensidade_polarizador)

    if "sepia" in filtros:
        result = _aplicar_sepia(result, intensidade_sepia)

    if "nitidez" in filtros:
        result = _aplicar_nitidez(result, intensidade_nitidez)

    if "posterizar" in filtros:
        result = _aplicar_posterizacao(result, intensidade_posterizar)

    if "negativo" in filtros:
        result = _aplicar_negativo(result, intensidade_negativo)
    
    return result


def memory_overflow_glitch(
    image_bgr: np.ndarray,
    intensity: int = 8,
    scale_decay: float = 0.82,
    jitter: int = 28,
    filtros: Optional[Set[str]] = None,
    intensidades: Optional[Dict[str, float]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simula efeito de estouro de memoria com triângulo invertido no eixo Y.
    
    Estrutura vertical:
    - Topo: Filtros distorcidos cada vez menores (mais efeitos)
    - Meio: Miniaturas coloridas reduzidas
    - Base: Imagem recortada (sem quadrado branco)
    
    Parâmetros:
    - filtros: set de strings com filtros a aplicar
              {'blur', 'rotacao', 'onda', 'brilho_contraste', 'saturacao'}
    """
    if filtros is None:
        filtros = {"blur", "rotacao", "onda", "brilho_contraste", "saturacao", "negativo", "polarizador", "sepia", "nitidez", "posterizar"}
    if intensidades is None:
        intensidades = {}
    
    rng = np.random.default_rng(seed)
    intensity = max(2, int(intensity))
    scale_decay = float(np.clip(scale_decay, 0.55, 0.95))
    jitter = max(0, int(jitter))
    intensidade_blur = _escala_efeito(intensidades.get("blur", 100.0))
    intensidade_rotacao = _escala_efeito(intensidades.get("rotacao", 100.0))
    intensidade_onda = _escala_efeito(intensidades.get("onda", 100.0))
    intensidade_bc = _escala_efeito(intensidades.get("brilho_contraste", 100.0))
    intensidade_sat = _escala_efeito(intensidades.get("saturacao", 100.0))

    h, w = image_bgr.shape[:2]
    base = image_bgr.copy()

    base_crop = recortar_centro(base, max(24, int(w * 0.94)), max(24, int(h * 0.88)))
    canvas = cv2.resize(base_crop, (w, h), interpolation=cv2.INTER_CUBIC).astype(np.float32)

    overlay_mask = np.zeros((h, w), dtype=np.float32)
    top_limit = max(1, int(h * 0.72))
    overlay_mask[:top_limit, :] = np.linspace(1.0, 0.0, top_limit, dtype=np.float32)[:, None]

    # Cria camadas em triângulo invertido (menores acima)
    for i in range(1, intensity):
        scale = scale_decay ** i
        new_w = max(24, int(w * scale))
        new_h = max(24, int(h * scale))
        
        if new_w < 8 or new_h < 8:
            break
            
        resized = cv2.resize(base, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Efeitos aumentam a cada camada
        effect_intensity = i / max(1, intensity - 1)

        # 1. Mudança de cor (mais radical nas camadas superiores)
        hue_src = float((i * 27 + rng.integers(0, 30)) % 180)
        hue_dst = float((hue_src + rng.integers(40, 140)) % 180)
        recolored, _ = shift_color(
            resized,
            source_hue=hue_src,
            target_hue=hue_dst,
            tolerance=50,
            feather=20,
        )

        # 2. Aplica filtros selecionados com parâmetros ajustados
        blur_k = max(3, int(5 - i * 0.8))
        if blur_k % 2 == 0:
            blur_k += 1
        
        filtered = aplicar_filtros_selecionados(
            recolored,
            filtros=filtros,
            blur_kernel=blur_k,
            rotacao_angle=25 * effect_intensity * intensidade_rotacao,
            onda_amplitude=(3.0 + i * 4.5) * intensidade_onda,
            brilho=rng.uniform(-0.25 * effect_intensity, 0.30 * effect_intensity) * intensidade_bc,
            contraste=1.0 + (
                (rng.uniform(0.5 + effect_intensity * 0.5, 1.5 + effect_intensity * 0.3) - 1.0)
                * intensidade_bc
            ),
            saturacao=1.0 + ((rng.uniform(0.4 + effect_intensity * 0.5, 1.8) - 1.0) * intensidade_sat),
            seed=rng.integers(0, 2**31 - 1) if seed else None,
        )

        # Posiciona verticalmente (eixo Y) - fica acima da base e vai subindo
        cx = w // 2 + int(rng.integers(-jitter // 2, jitter // 2 + 1))
        x0 = int(np.clip(cx - new_w // 2, 0, max(0, w - new_w)))

        y0 = int(np.clip(h - new_h - int((i + 1) * (h / (intensity + 1))), 0, max(0, h - new_h)))

        alpha = max(0.18, 0.92 - (i * 0.07))
        roi = canvas[y0:y0 + new_h, x0:x0 + new_w]
        mask = overlay_mask[y0:y0 + new_h, x0:x0 + new_w]
        canvas[y0:y0 + new_h, x0:x0 + new_w] = _blend_mascarado(roi, filtered.astype(np.float32), mask, alpha)

    return np.clip(canvas, 0, 255).astype(np.uint8)


def detectar_hue_principal(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> float:
    """Detecta a matiz dominante, priorizando pixels com boa saturacao e brilho."""
    mascara = (s > 35) & (v > 25)
    if not np.any(mascara):
        return float(np.mean(h))

    h_valid = h[mascara].astype(np.uint8)
    pesos = ((s[mascara].astype(np.float32) + 1.0) * (v[mascara].astype(np.float32) + 1.0))
    hist = np.bincount(h_valid, weights=pesos, minlength=180)
    return float(np.argmax(hist))


def _dist_circular_hue(h: np.ndarray, h0: float) -> np.ndarray:
    d = np.abs(h - h0)
    return np.minimum(d, 180.0 - d)


def trocar_cor_principal(
    imagem_bgr: np.ndarray,
    cor_destino: str,
    cor_origem: str = "auto",
    tolerancia: int = 18,
    suavizacao: int = 12,
    proteger_baixa_saturacao: bool = True,
) -> Tuple[np.ndarray, dict]:
    """
    Troca a cor principal mantendo luminosidade (canal V no HSV).

    Estrategia:
    - Detecta (ou usa) hue de origem.
    - Calcula distancia circular no eixo de hue.
    - Aplica deslocamento gradual para hue destino com feather (suavizacao).
    - Mantem S e V para preservar estrutura visual e luminosidade.
    """
    if cor_destino not in COR_PARA_HUE:
        raise ValueError(f"Cor destino invalida: {cor_destino}")

    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    if cor_origem == "auto":
        hue_origem = detectar_hue_principal(h, s, v)
    else:
        if cor_origem not in COR_PARA_HUE:
            raise ValueError(f"Cor origem invalida: {cor_origem}")
        hue_origem = float(COR_PARA_HUE[cor_origem])

    hue_destino = float(COR_PARA_HUE[cor_destino])

    tolerancia = max(0, int(tolerancia))
    suavizacao = max(0, int(suavizacao))
    dist = _dist_circular_hue(h, hue_origem)

    if suavizacao > 0:
        w = np.clip((tolerancia + suavizacao - dist) / float(suavizacao), 0.0, 1.0)
        w[dist <= tolerancia] = 1.0
    else:
        w = (dist <= tolerancia).astype(np.float32)

    if proteger_baixa_saturacao:
        w *= np.clip((s - 20.0) / 100.0, 0.0, 1.0)

    delta = ((hue_destino - hue_origem + 90.0) % 180.0) - 90.0
    h_novo = (h + delta * w) % 180.0

    hsv_novo = hsv.copy()
    hsv_novo[:, :, 0] = h_novo
    imagem_saida = cv2.cvtColor(hsv_novo.astype(np.uint8), cv2.COLOR_HSV2BGR)

    metadados = {
        "hue_origem": float(hue_origem),
        "hue_destino": float(hue_destino),
        "tolerancia": tolerancia,
        "suavizacao": suavizacao,
        "cor_origem": cor_origem,
        "cor_destino": cor_destino,
    }
    return imagem_saida, metadados


def salvar_troca_cor(
    caminho_entrada: str,
    imagem_saida_bgr: np.ndarray,
    pasta_saida: str,
    sufixo: str = "troca_cor_principal",
) -> str:
    os.makedirs(pasta_saida, exist_ok=True)
    base = os.path.splitext(os.path.basename(caminho_entrada))[0]
    caminho_saida = os.path.join(pasta_saida, f"{base}_{sufixo}.png")
    if not cv2.imwrite(caminho_saida, imagem_saida_bgr):
        raise IOError(f"Nao foi possivel salvar a imagem em: {caminho_saida}")
    return caminho_saida


def salvar_troca_cor_com_filtros(
    caminho_entrada: str,
    imagem_saida_bgr: np.ndarray,
    pasta_saida: str,
    filtros: Optional[Set[str]] = None,
    sufixo: str = "troca_cor_com_filtros",
    blur_kernel: int = 5,
    rotacao_angle: float = 10.0,
    onda_amplitude: float = 3.0,
    brilho: float = 0.0,
    contraste: float = 1.0,
    saturacao: float = 1.0,
) -> str:
    """
    Salva imagem de troca de cor com filtros selecionados aplicados.
    
    Parâmetros:
    - filtros: set de strings {'blur', 'rotacao', 'onda', 'brilho_contraste', 'saturacao'}
    - sufixo: sufixo do arquivo para diferenciar versões
    """
    os.makedirs(pasta_saida, exist_ok=True)
    
    # Aplica filtros selecionados
    imagem_filtrada = aplicar_filtros_selecionados(
        imagem_saida_bgr,
        filtros=filtros,
        blur_kernel=blur_kernel,
        rotacao_angle=rotacao_angle,
        onda_amplitude=onda_amplitude,
        brilho=brilho,
        contraste=contraste,
        saturacao=saturacao,
    )
    
    base = os.path.splitext(os.path.basename(caminho_entrada))[0]
    
    # Adiciona detalhes sobre filtros aplicados ao nome do arquivo
    filtros_str = "_".join(sorted(filtros)) if filtros else "nenhum"
    caminho_saida = os.path.join(pasta_saida, f"{base}_{sufixo}_{filtros_str}.png")
    
    if not cv2.imwrite(caminho_saida, imagem_filtrada):
        raise IOError(f"Nao foi possivel salvar a imagem em: {caminho_saida}")
    return caminho_saida
