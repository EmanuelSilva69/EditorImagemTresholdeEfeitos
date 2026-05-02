"""Processamento de cores avançado com efeitos Onyx e Perola.

Este modulo implementa efeitos tonais para objetos segmentados sobre fundo claro,
com fallback CPU e uso opcional de CUDA quando disponivel.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EffectResult:
    """Resultado de um efeito aplicado.

    Attributes:
        image_bgr: Imagem de saida em BGR uint8.
        object_mask: Mascara binaria do objeto (255 objeto, 0 fundo).
        used_gpu: Indica se houve aceleracao GPU no pipeline principal.
    """

    image_bgr: np.ndarray
    object_mask: np.ndarray
    used_gpu: bool


class ColorEffectsProcessor:
    """Processador de efeitos tonais com deteccao de GPU e segmentacao robusta."""

    def __init__(self) -> None:
        self.cuda_devices = self._cuda_device_count()
        self.cuda_enabled = self.cuda_devices > 0

    @staticmethod
    def _cuda_device_count() -> int:
        """Retorna o numero de GPUs CUDA disponiveis para OpenCV."""
        try:
            cuda_api = getattr(cv2, "cuda", None)
            if cuda_api is None:
                return 0
            if hasattr(cuda_api, "getDeviceCount"):
                return int(cuda_api.getDeviceCount())
            if hasattr(cuda_api, "getCudaEnabledDeviceCount"):
                return int(cuda_api.getCudaEnabledDeviceCount())
            return 0
        except Exception:
            return 0

    def build_backend_message(self) -> str:
        if self.cuda_enabled:
            return f"GPU CUDA ativa ({self.cuda_devices} dispositivo(s))"
        return "Execucao em CPU (CUDA indisponivel no OpenCV)"

    @staticmethod
    def _ensure_u8(image: np.ndarray) -> np.ndarray:
        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def segment_object_mask(frame_bgr: np.ndarray) -> np.ndarray:
        """Segmenta o objeto usando luminancia (canal L em LAB).

        Estrategia:
        1. Converte para LAB e usa o canal L para separar fundo claro de objeto.
        2. Aplica Otsu invertido para priorizar objeto mais escuro que o fundo.
        3. Refina com morfologia (open + close) para reduzir ruido e preencher buracos.
        4. Mantem apenas o maior componente conexo para robustez.
        """
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        blur = cv2.GaussianBlur(l_channel, (5, 5), 0)
        _thr, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((5, 5), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count <= 1:
            return mask

        largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        largest_mask = np.zeros_like(mask)
        largest_mask[labels == largest_idx] = 255
        return largest_mask

    def _apply_lut(self, gray_u8: np.ndarray, lut: np.ndarray) -> tuple[np.ndarray, bool]:
        """Aplica LUT com tentativa de aceleracao GPU; fallback para CPU."""
        if self.cuda_enabled and hasattr(cv2, "cuda_GpuMat") and hasattr(cv2.cuda, "LUT"):
            try:
                gpu_src = cv2.cuda_GpuMat()
                gpu_src.upload(gray_u8)
                gpu_lut = cv2.cuda_GpuMat()
                gpu_lut.upload(lut.reshape(1, 256))
                gpu_out = cv2.cuda.LUT(gpu_src, gpu_lut)
                return gpu_out.download(), True
            except Exception:
                pass
        return cv2.LUT(gray_u8, lut), False

    def apply_onyx_effect(self, frame_bgr: np.ndarray, gamma: float = 3.5) -> EffectResult:
        r"""Aplica o Efeito Onyx (preto profundo) no objeto segmentado.

        Formula tonal (via LUT):
        $$y = \left(\frac{x}{255}\right)^\gamma \cdot 255$$

        Com $\gamma > 1$, tons medios sao comprimidos para perto do preto, enquanto
        altas luzes permanecem mais visiveis, reforcando reflexos especulares.
        """
        gamma = max(1.0, float(gamma))
        mask = self.segment_object_mask(frame_bgr)

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        x = np.arange(256, dtype=np.float32)
        lut = np.power(x / 255.0, gamma) * 255.0
        lut_u8 = self._ensure_u8(lut)
        onyx_gray, used_gpu = self._apply_lut(gray, lut_u8)

        # Preserva um pouco de brilho alto para aspecto de pedra polida.
        highlight = cv2.GaussianBlur(gray, (0, 0), sigmaX=3.0)
        onyx_gray = self._ensure_u8(0.9 * onyx_gray + 0.1 * highlight)

        onyx_bgr = cv2.cvtColor(onyx_gray, cv2.COLOR_GRAY2BGR)
        out = frame_bgr.copy()
        obj = mask > 0
        out[obj] = onyx_bgr[obj]
        return EffectResult(image_bgr=out, object_mask=mask, used_gpu=used_gpu)

    def apply_pearl_effect(
        self,
        frame_bgr: np.ndarray,
        contrast: float = 1.2,
        brightness: float = 30.0,
        background_gray: int = 200,
    ) -> EffectResult:
        r"""Aplica o Efeito Perola (branco de alto contraste) com fundo cinza neutro.

        Mapeamento tonal do objeto em escala de cinza:
        $$y = \alpha \cdot x + \beta$$

        onde $\alpha$ e o contraste e $\beta$ o deslocamento de brilho.
        """
        alpha = float(max(0.1, contrast))
        beta = float(brightness)
        bg = int(np.clip(background_gray, 0, 255))

        mask = self.segment_object_mask(frame_bgr)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        pearl_gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        pearl_gray = self._ensure_u8(pearl_gray)
        pearl_bgr = cv2.cvtColor(pearl_gray, cv2.COLOR_GRAY2BGR)

        out = np.full_like(frame_bgr, bg, dtype=np.uint8)
        obj = mask > 0
        out[obj] = pearl_bgr[obj]
        return EffectResult(image_bgr=out, object_mask=mask, used_gpu=False)


def plot_effect_transfer_curves(
    gamma: float = 3.5,
    contrast: float = 1.2,
    brightness: float = 30.0,
    save_path: str | None = None,
    show: bool = True,
) -> str | None:
    """Gera um grafico comparando as curvas de transferencia de Onyx e Perola.

    Args:
        gamma: Parametro de curva de potencia do efeito Onyx.
        contrast: Coeficiente linear alpha do efeito Perola.
        brightness: Deslocamento linear beta do efeito Perola.
        save_path: Caminho opcional para salvar o grafico em PNG.
        show: Se True, exibe a janela interativa do matplotlib.

    Returns:
        O caminho salvo quando save_path e fornecido; caso contrario None.
    """
    import matplotlib.pyplot as plt

    x = np.arange(256, dtype=np.float32)
    onyx = np.power(x / 255.0, max(1.0, float(gamma))) * 255.0
    pearl = np.clip(float(contrast) * x + float(brightness), 0, 255)

    plt.figure(figsize=(8, 5))
    plt.plot(x, onyx, label=f"Onyx gamma={gamma}", linewidth=2)
    plt.plot(x, pearl, label=f"Perola alpha={contrast}, beta={brightness}", linewidth=2)
    plt.plot(x, x, "--", label="Identidade", alpha=0.6)
    plt.xlabel("Input (0..255)")
    plt.ylabel("Output (0..255)")
    plt.title("Curvas de Transferencia - Onyx vs Perola")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    saved = None
    if save_path:
        plt.savefig(save_path, dpi=150)
        saved = save_path

    if show:
        plt.show()
    else:
        plt.close()

    return saved


if __name__ == "__main__":
    plot_effect_transfer_curves()
