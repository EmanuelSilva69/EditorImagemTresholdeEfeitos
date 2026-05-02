import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity as ssim


@dataclass
class ResultadoProcessamento:
    imagem_bgr: np.ndarray
    ssim_value: Optional[float] = None
    modo: str = ""


class AdvancedVisionProcessor:
    def __init__(self) -> None:
        self.backend = self._detect_backend()
        self.cuda_supported = self._build_supports_cuda()
        self.backend_message = self._build_backend_message()

    @staticmethod
    def _detect_backend() -> str:
        cuda_api = getattr(cv2, "cuda", None)
        if cuda_api is None:
            return "cpu"

        try:
            if hasattr(cuda_api, "getDeviceCount"):
                count = int(cuda_api.getDeviceCount())
            elif hasattr(cuda_api, "getCudaEnabledDeviceCount"):
                count = int(cuda_api.getCudaEnabledDeviceCount())
            else:
                count = 0
        except Exception:
            count = 0

        return "gpu" if count > 0 else "cpu"

    @staticmethod
    def _build_supports_cuda() -> bool:
        build_info = cv2.getBuildInformation()
        markers = ("CUDA: YES", "NVIDIA CUDA: YES", "Use CUDA: YES")
        return any(marker in build_info for marker in markers)

    def _build_backend_message(self) -> str:
        if self.backend == "gpu":
            return "GPU CUDA ativa no OpenCV"
        if self.cuda_supported:
            return "CUDA compilado no OpenCV, mas sem dispositivo detectado"
        return "OpenCV atual nao foi compilado com suporte CUDA"

    def _can_use_cuda_color(self) -> bool:
        cuda_api = getattr(cv2, "cuda", None)
        return bool(
            self.backend == "gpu"
            and cuda_api is not None
            and hasattr(cv2, "cuda_GpuMat")
            and hasattr(cuda_api, "cvtColor")
        )

    @staticmethod
    def _ensure_uint8(image: np.ndarray) -> np.ndarray:
        return np.clip(image, 0, 255).astype(np.uint8)

    @staticmethod
    def _resize_to_match(source: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        height, width = target_shape
        if source.shape[:2] == (height, width):
            return source
        return cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)

    def rgb_to_hsv(self, image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("A imagem RGB deve ter 3 canais.")

        if self._can_use_cuda_color():
            try:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(image_rgb)
                gpu_hsv = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_RGB2HSV)
                return gpu_hsv.download()
            except Exception:
                pass

        return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    def bgr_to_hsv(self, image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self.rgb_to_hsv(image_rgb)

    def hsv_preview_bgr(self, image_bgr: np.ndarray) -> np.ndarray:
        hsv = self.bgr_to_hsv(image_bgr)
        h, s, v = cv2.split(hsv)

        h_vis = cv2.applyColorMap(cv2.convertScaleAbs(h, alpha=255.0 / 179.0), cv2.COLORMAP_TURBO)
        s_vis = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
        v_vis = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
        preview = np.hstack([h_vis, s_vis, v_vis])
        return preview

    def histogram_match_bgr(self, source_bgr: np.ndarray, reference_bgr: np.ndarray) -> np.ndarray:
        source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
        reference_rgb = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
        matched_rgb = match_histograms(source_rgb, reference_rgb, channel_axis=-1)
        matched_rgb = self._ensure_uint8(matched_rgb)
        return cv2.cvtColor(matched_rgb, cv2.COLOR_RGB2BGR)

    def histogram_match_rgb(self, source_rgb: np.ndarray, reference_rgb: np.ndarray) -> np.ndarray:
        matched_rgb = match_histograms(source_rgb, reference_rgb, channel_axis=-1)
        return self._ensure_uint8(matched_rgb)

    def compute_ssim_bgr(self, original_bgr: np.ndarray, processed_bgr: np.ndarray) -> float:
        original = original_bgr
        processed = self._resize_to_match(processed_bgr, original.shape[:2])

        if original.ndim == 3 and processed.ndim == 3:
            original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
            processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            return float(ssim(original_rgb, processed_rgb, channel_axis=2, data_range=255))

        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        return float(ssim(original_gray, processed_gray, data_range=255))

    def process_image(self, source_bgr: np.ndarray, reference_bgr: Optional[np.ndarray], mode: str) -> ResultadoProcessamento:
        mode = mode.lower().strip()
        if mode == "hsv":
            hsv_preview = self.hsv_preview_bgr(source_bgr)
            hsv_roundtrip = cv2.cvtColor(self.bgr_to_hsv(source_bgr), cv2.COLOR_HSV2BGR)
            score = self.compute_ssim_bgr(source_bgr, hsv_roundtrip)
            return ResultadoProcessamento(imagem_bgr=hsv_preview, ssim_value=score, modo="HSV")

        if reference_bgr is None:
            raise ValueError("Selecione uma imagem de referência para Histogram Matching.")

        matched = self.histogram_match_bgr(source_bgr, reference_bgr)
        score = self.compute_ssim_bgr(source_bgr, matched)
        return ResultadoProcessamento(imagem_bgr=matched, ssim_value=score, modo="Matching")

    def _overlay_info(self, frame_bgr: np.ndarray, text: str) -> np.ndarray:
        out = frame_bgr.copy()
        cv2.rectangle(out, (8, 8), (860, 52), (0, 0, 0), -1)
        cv2.putText(out, text, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return out

    def process_video(
        self,
        source_video_path: str,
        output_video_path: str,
        reference_path: Optional[str] = None,
        preview: bool = True,
        initial_mode: str = "hsv",
    ) -> dict:
        if source_video_path.isdigit():
            cap = cv2.VideoCapture(int(source_video_path))
        else:
            if not os.path.exists(source_video_path):
                raise FileNotFoundError("Video de entrada nao encontrado.")
            cap = cv2.VideoCapture(source_video_path)
        if not cap.isOpened():
            raise ValueError("Nao foi possivel abrir o video de entrada.")

        reference_bgr = None
        if reference_path:
            reference_bgr = cv2.imread(reference_path, cv2.IMREAD_COLOR)
            if reference_bgr is None:
                raise ValueError("Nao foi possivel abrir a imagem de referencia.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), True)
        if not writer.isOpened():
            cap.release()
            raise ValueError("Nao foi possivel criar o arquivo de saida do video.")

        mode = initial_mode.lower().strip()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_idx = 0
        ssim_values: list[float] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            try:
                if mode == "hsv":
                    output_frame = cv2.cvtColor(self.bgr_to_hsv(frame), cv2.COLOR_HSV2BGR)
                    display_frame = self.hsv_preview_bgr(frame)
                    ssim_value = self.compute_ssim_bgr(frame, output_frame)
                else:
                    if reference_bgr is None:
                        raise ValueError("Selecione uma imagem de referência para Histogram Matching.")
                    matched = self.histogram_match_bgr(frame, reference_bgr)
                    output_frame = matched
                    display_frame = matched
                    ssim_value = self.compute_ssim_bgr(frame, output_frame)

                ssim_values.append(ssim_value)
                overlay = f"Modo: {mode.upper()} | Backend: {self.backend.upper()} | SSIM: {ssim_value:.4f}"
                output_frame = cv2.resize(output_frame, (width, height), interpolation=cv2.INTER_AREA)
                display_frame = cv2.resize(display_frame, (width, height), interpolation=cv2.INTER_AREA)
                output_frame = self._overlay_info(output_frame, overlay)
                display_frame = self._overlay_info(display_frame, overlay)
                writer.write(output_frame)

                if preview:
                    cv2.imshow("Processamento Avancado", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("h"):
                        mode = "hsv"
                    elif key == ord("m"):
                        mode = "matching"
            except Exception:
                break

        cap.release()
        writer.release()
        if preview:
            cv2.destroyAllWindows()

        return {
            "output_path": output_video_path,
            "frames": frame_idx,
            "total_frames": total_frames,
            "mode": mode,
            "backend": self.backend,
            "ssim_mean": float(np.mean(ssim_values)) if ssim_values else None,
        }
