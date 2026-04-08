import os
from typing import Optional, Tuple

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


def memory_overflow_glitch(
    image_bgr: np.ndarray,
    intensity: int = 8,
    scale_decay: float = 0.82,
    jitter: int = 28,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simula efeito de estouro de memoria com pirâmide invertida e múltiplos efeitos.
    
    Cria uma composição em pirâmide com a imagem menor em cima, aplicando:
    - Mudanças de cor progressivas
    - Blur e rotação
    - Distorção tipo onda
    - Variações de brilho, contraste e saturação
    """
    rng = np.random.default_rng(seed)
    intensity = max(1, int(intensity))
    scale_decay = float(np.clip(scale_decay, 0.55, 0.95))
    jitter = max(0, int(jitter))

    h, w = image_bgr.shape[:2]
    canvas = image_bgr.astype(np.float32).copy()
    base = image_bgr.copy()

    # Cria pirâmide invertida (menor em cima, maior embaixo)
    layers = []
    for i in range(intensity):
        scale = scale_decay ** i
        new_w = max(24, int(w * scale))
        new_h = max(24, int(h * scale))
        resized = cv2.resize(base, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Aplica múltiplos efeitos
        # 1. Mudança de cor
        hue_src = float((i * 23) % 180)
        hue_dst = float((hue_src + rng.integers(30, 110)) % 180)
        recolored, _ = shift_color(
            resized,
            source_hue=hue_src,
            target_hue=hue_dst,
            tolerance=42,
            feather=16,
        )

        # 2. Blur progressivo aumentando a cada camada
        blur_kernel = 3 + i * 2
        blurred = _aplicar_blur_progressivo(recolored, blur_kernel)

        # 3. Rotação suave
        angle = rng.uniform(-12, 12)
        rotated = _aplicar_rotacao_suave(blurred, angle)

        # 4. Distorção tipo onda
        distorted = _aplicar_distorcao_onda(rotated, amplitude=2.0 + i * 1.5)

        # 5. Brilho, contraste e saturação
        brightness = rng.uniform(-0.1, 0.15)
        contrast = rng.uniform(0.7, 1.3)
        saturation = rng.uniform(0.6, 1.4)
        adjusted = _aplicar_brilho_contraste(distorted, brightness, contrast)
        adjusted = _aplicar_saturacao(adjusted, saturation)

        layers.append({
            "image": adjusted,
            "width": new_w,
            "height": new_h,
            "layer_index": i,
        })

    # Composição em pirâmide invertida (menores em cima/centro)
    for layer_data in layers:
        layer_img = layer_data["image"]
        lw = layer_data["width"]
        lh = layer_data["height"]
        layer_idx = layer_data["layer_index"]

        # Posiciona em torno do centro com leve jitter
        cx = w // 2 + int(rng.integers(-jitter, jitter + 1))
        cy = h // 2 + int(rng.integers(-jitter, jitter + 1))
        x0 = int(np.clip(cx - lw // 2, 0, max(0, w - lw)))
        y0 = int(np.clip(cy - lh // 2, 0, max(0, h - lh)))

        # Transparência aumenta conforme desce a pirâmide
        alpha = max(0.10, 0.70 - layer_idx * 0.065)
        
        roi = canvas[y0 : y0 + lh, x0 : x0 + lw]
        layer = layer_img.astype(np.float32)
        canvas[y0 : y0 + lh, x0 : x0 + lw] = roi * (1.0 - alpha) + layer * (alpha)

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
    cv2.imwrite(caminho_saida, imagem_saida_bgr)
    return caminho_saida
