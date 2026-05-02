"""Modulo T2 - Processamento de Cores, Histogram Matching, SSIM e Recolorizacao HSV."""

from __future__ import annotations

import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import colorchooser, filedialog, messagebox, simpledialog, ttk

from color_processor import ColorEffectsProcessor, plot_effect_transfer_curves
from processamento_avancado import AdvancedVisionProcessor

if hasattr(Image, "Resampling"):
    RESAMPLING = Image.Resampling.LANCZOS
else:
    RESAMPLING = Image.LANCZOS


@dataclass
class FrameResult:
    original_bgr: np.ndarray
    processed_bgr: np.ndarray
    ssim_value: float
    mode_name: str


class ColorAppGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("T2 - Processamento de Cores")
        self.root.geometry("1520x900")
        self.root.minsize(1280, 780)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.processor = AdvancedVisionProcessor()
        self.effects = ColorEffectsProcessor()
        self._after_id: str | None = None
        self._capture: cv2.VideoCapture | None = None
        self._streaming = False
        self._current_frame_bgr: np.ndarray | None = None
        self._original_photo: ImageTk.PhotoImage | None = None
        self._processed_photo: ImageTk.PhotoImage | None = None

        self.image_path = tk.StringVar(value="")
        self.video_path = tk.StringVar(value="")
        self.reference_path = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value=os.path.join(os.path.dirname(__file__), "..", "resultados", "t2"))
        self.source_mode = tk.StringVar(value="webcam")
        self.processing_mode = tk.StringVar(value="recolor")
        self.webcam_index = tk.IntVar(value=0)
        self.h_var = tk.IntVar(value=0)
        self.s_var = tk.IntVar(value=255)
        self.v_var = tk.IntVar(value=255)
        self.hue_shift = tk.IntVar(value=0)
        self.achromatic_sat_keep = tk.DoubleVar(value=0.0)
        self.achromatic_lum_scale = tk.DoubleVar(value=1.25)
        self.onyx_gamma = tk.DoubleVar(value=3.5)
        self.pearl_contrast = tk.DoubleVar(value=1.2)
        self.pearl_brightness = tk.DoubleVar(value=30.0)
        self.ssim_var = tk.StringVar(value="SSIM: -")
        self.gpu_var = tk.StringVar(value=f"{self.processor.backend_message} | {self.effects.build_backend_message()}")
        self.status_var = tk.StringVar(value="Selecione uma fonte e pressione Iniciar Preview.")
        self.color_hex_var = tk.StringVar(value="#ffffff")
        self.color_label_var = tk.StringVar(value="Cor HSV: H=0 S=255 V=255 (colorida)")

        self._build_ui()
        self._source_mode_changed()
        self._processing_mode_changed()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        controls = ttk.Frame(self.root, padding=(14, 14, 10, 14))
        controls.grid(row=0, column=0, sticky="nsew")
        controls.columnconfigure(0, weight=1)

        previews = ttk.Frame(self.root, padding=(10, 14, 14, 14))
        previews.grid(row=0, column=1, sticky="nsew")
        previews.rowconfigure(1, weight=1)
        previews.columnconfigure(0, weight=1)
        previews.columnconfigure(1, weight=1)

        header = ttk.Label(
            controls,
            text="T2 - Processamento de Cores",
            font=("Segoe UI", 18, "bold"),
        )
        header.grid(row=0, column=0, sticky="w", pady=(0, 10))

        ttk.Label(
            controls,
            text=self.processor.backend_message,
            foreground="#4f4f4f",
            wraplength=380,
        ).grid(row=1, column=0, sticky="w", pady=(0, 14))

        source_box = ttk.LabelFrame(controls, text="Fonte", padding=10)
        source_box.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        source_box.columnconfigure(1, weight=1)

        ttk.Label(source_box, text="Modo de entrada").grid(row=0, column=0, sticky="w")
        mode_row = ttk.Frame(source_box)
        mode_row.grid(row=0, column=1, sticky="w")
        for label in ("imagem", "video", "webcam"):
            ttk.Radiobutton(
                mode_row,
                text=label.title(),
                value=label,
                variable=self.source_mode,
                command=self._source_mode_changed,
            ).pack(side="left", padx=(0, 8))

        self._path_row(source_box, 1, "Imagem", self.image_path, self._browse_image)
        self._path_row(source_box, 2, "Video", self.video_path, self._browse_video)
        self._path_row(source_box, 3, "Referencia", self.reference_path, self._browse_reference)
        self._path_row(source_box, 4, "Saida", self.output_dir, self._browse_output_dir)

        webcam_row = ttk.Frame(source_box)
        webcam_row.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(webcam_row, text="Webcam").pack(side="left")
        ttk.Spinbox(webcam_row, from_=0, to=10, textvariable=self.webcam_index, width=6).pack(side="left", padx=(8, 0))

        mode_box = ttk.LabelFrame(controls, text="Processamento", padding=10)
        mode_box.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(mode_box, text="Modo de cor").grid(row=0, column=0, sticky="w")
        self.mode_combo = ttk.Combobox(
            mode_box,
            textvariable=self.processing_mode,
            values=["recolor", "hsv", "matching", "hsv_match", "onyx", "pearl"],
            state="readonly",
            width=16,
        )
        self.mode_combo.grid(row=0, column=1, sticky="w", padx=(10, 0))
        self.mode_combo.bind("<<ComboboxSelected>>", lambda _event: self._processing_mode_changed())

        color_row = ttk.Frame(mode_box)
        color_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Button(color_row, text="Escolher cor HSV", command=self.choose_hsv_color).pack(side="left")
        self.color_swatch = tk.Label(color_row, text="", width=5, height=2, relief="solid", bd=1, bg="#ffffff")
        self.color_swatch.pack(side="left", padx=10)
        ttk.Label(color_row, textvariable=self.color_label_var, wraplength=260).pack(side="left", fill="x", expand=True)

        slider_box = ttk.LabelFrame(controls, text="Ajustes HSV", padding=10)
        slider_box.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        slider_box.columnconfigure(1, weight=1)
        self._add_slider(slider_box, 0, "Hue alvo", self.h_var, 0, 179)
        self._add_slider(slider_box, 1, "Sat", self.s_var, 0, 255)
        self._add_slider(slider_box, 2, "Val", self.v_var, 0, 255)
        self._add_slider(slider_box, 3, "Hue shift", self.hue_shift, -179, 179)

        refine_box = ttk.LabelFrame(controls, text="Refino Branco/Preto", padding=10)
        refine_box.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        refine_box.columnconfigure(1, weight=1)
        self._add_slider(refine_box, 0, "Sat keep", self.achromatic_sat_keep, 0.0, 1.0)
        self._add_slider(refine_box, 1, "Lum scale", self.achromatic_lum_scale, 0.1, 2.2)

        studio_box = ttk.LabelFrame(controls, text="Efeitos de Estudio", padding=10)
        studio_box.grid(row=6, column=0, sticky="ew", pady=(0, 10))
        studio_box.columnconfigure(1, weight=1)
        self._add_slider(studio_box, 0, "Gamma Onyx", self.onyx_gamma, 1.0, 6.0)
        self._add_slider(studio_box, 1, "Contraste", self.pearl_contrast, 0.5, 2.5)
        self._add_slider(studio_box, 2, "Brilho", self.pearl_brightness, -20.0, 90.0)

        action_box = ttk.LabelFrame(controls, text="Acoes", padding=10)
        action_box.grid(row=7, column=0, sticky="ew")
        for text, command, column in (
            ("Iniciar Preview", self.start_preview, 0),
            ("Parar Preview", self.stop_preview, 1),
            ("Salvar imagem", self.save_image, 0),
            ("Salvar video", self.save_video, 1),
        ):
            pass

        top_actions = ttk.Frame(action_box)
        top_actions.grid(row=0, column=0, columnspan=2, sticky="ew")
        top_actions.columnconfigure(0, weight=1)
        top_actions.columnconfigure(1, weight=1)
        top_actions.columnconfigure(2, weight=1)
        top_actions.columnconfigure(3, weight=1)
        top_actions.columnconfigure(4, weight=1)
        top_actions.columnconfigure(5, weight=1)
        ttk.Button(top_actions, text="Iniciar Preview", command=self.start_preview).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(top_actions, text="Parar Preview", command=self.stop_preview).grid(row=0, column=1, sticky="ew", padx=(0, 6))
        ttk.Button(top_actions, text="Salvar imagem", command=self.save_image).grid(row=0, column=2, sticky="ew", padx=(0, 6))
        ttk.Button(top_actions, text="Salvar video", command=self.save_video).grid(row=0, column=3, sticky="ew")
        ttk.Button(top_actions, text="Plotar Curvas", command=self.show_transfer_curves).grid(row=0, column=4, sticky="ew", padx=(6, 0))
        ttk.Button(top_actions, text="Salvar Curvas PNG", command=self.save_transfer_curves_png).grid(row=0, column=5, sticky="ew", padx=(6, 0))

        footer = ttk.Frame(controls)
        footer.grid(row=8, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(footer, textvariable=self.gpu_var, foreground="#444444", wraplength=380).pack(anchor="w")
        ttk.Label(footer, textvariable=self.ssim_var, foreground="#444444").pack(anchor="w", pady=(4, 0))
        ttk.Label(footer, textvariable=self.status_var, foreground="#222222", wraplength=380).pack(anchor="w", pady=(4, 0))

        preview_left = ttk.LabelFrame(previews, text="Original", padding=8)
        preview_left.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        preview_left.rowconfigure(0, weight=1)
        preview_left.columnconfigure(0, weight=1)

        preview_right = ttk.LabelFrame(previews, text="Processado", padding=8)
        preview_right.grid(row=0, column=1, sticky="nsew", pady=(0, 8))
        preview_right.rowconfigure(0, weight=1)
        preview_right.columnconfigure(0, weight=1)

        self.original_label = ttk.Label(preview_left, anchor="center")
        self.original_label.grid(row=0, column=0, sticky="nsew")
        self.processed_label = ttk.Label(preview_right, anchor="center")
        self.processed_label.grid(row=0, column=0, sticky="nsew")

        preview_footer = ttk.Frame(previews)
        preview_footer.grid(row=1, column=0, columnspan=2, sticky="ew")
        ttk.Label(
            preview_footer,
            text="Preview embutido em Tkinter. Use os controles do lado esquerdo para trocar fonte, modo e cor em tempo real.",
            wraplength=920,
            foreground="#4c4c4c",
        ).pack(anchor="w")

    def _path_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        row_frame = ttk.Frame(parent)
        row_frame.grid(row=row, column=1, sticky="ew", pady=3)
        row_frame.columnconfigure(0, weight=1)
        ttk.Entry(row_frame, textvariable=variable).grid(row=0, column=0, sticky="ew")
        ttk.Button(row_frame, text="...", command=command, width=4).grid(row=0, column=1, padx=(6, 0))

    def _add_slider(self, parent: ttk.Frame, row: int, label: str, variable: tk.Variable, minimum: float, maximum: float) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        slider = ttk.Scale(parent, from_=minimum, to=maximum, variable=variable, command=lambda _value: self._on_slider_change())
        slider.grid(row=row, column=1, sticky="ew", padx=(10, 10))
        ttk.Label(parent, textvariable=variable, width=8).grid(row=row, column=2, sticky="e")

    def _browse_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if path:
            self.image_path.set(path)
            self.source_mode.set("imagem")
            self._source_mode_changed()
            self.refresh_preview()

    def _browse_video(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            self.video_path.set(path)
            self.source_mode.set("video")
            self._source_mode_changed()

    def _browse_reference(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        if path:
            self.reference_path.set(path)
            self.refresh_preview()

    def _browse_output_dir(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.output_dir.set(path)

    def _source_mode_changed(self) -> None:
        mode = self.source_mode.get().strip().lower()
        is_video = mode in {"video", "webcam"}
        state_video = "normal" if mode == "video" else ("disabled" if mode == "imagem" else "normal")
        state_image = "normal" if mode == "imagem" else "disabled"
        self._set_entry_state(self.image_path, state_image)
        self._set_entry_state(self.video_path, state_video)
        if is_video:
            self.status_var.set("Modo de video/webcam pronto. Clique em Iniciar Preview.")
        else:
            self.status_var.set("Modo imagem pronto. Clique em Iniciar Preview ou altere os sliders.")
        if mode == "imagem":
            self.stop_preview()
            self.refresh_preview()

    def _processing_mode_changed(self) -> None:
        mode = self.processing_mode.get().strip().lower()
        if mode == "recolor":
            target_type = self._target_color_type(self.s_var.get(), self.v_var.get())
            self.color_label_var.set(f"Cor HSV: H={self.h_var.get()} S={self.s_var.get()} V={self.v_var.get()} ({target_type})")
        self.refresh_preview()

    def _on_slider_change(self) -> None:
        self._update_color_swatch()
        if self.processing_mode.get().strip().lower() == "recolor":
            target_type = self._target_color_type(self.s_var.get(), self.v_var.get())
            self.color_label_var.set(f"Cor HSV: H={self.h_var.get()} S={self.s_var.get()} V={self.v_var.get()} ({target_type})")
        self.refresh_preview()

    def _set_entry_state(self, variable: tk.StringVar, state: str) -> None:
        # placeholder to keep the layout logic compact; state handling is done in refresh methods
        _ = variable, state

    @staticmethod
    def _load_bgr(path: str) -> np.ndarray:
        if not path:
            raise ValueError("Selecione um arquivo valido.")
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Nao foi possivel abrir: {path}")
        return image

    def _resolve_output_dir(self) -> str:
        path = self.output_dir.get().strip()
        if not path:
            return os.path.join(os.path.dirname(__file__), "..", "resultados", "t2")
        if os.path.isabs(path):
            return os.path.normpath(path)
        return os.path.normpath(os.path.join(os.path.dirname(__file__), path))

    @staticmethod
    def _resize_keep_ratio(image_bgr: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
        height, width = image_bgr.shape[:2]
        if width <= 0 or height <= 0:
            return image_bgr
        scale = min(max_width / width, max_height / height, 1.0)
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        return cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _bgr_to_photoimage(image_bgr: np.ndarray, max_width: int = 660, max_height: int = 480) -> ImageTk.PhotoImage:
        resized = ColorAppGUI._resize_keep_ratio(image_bgr, max_width, max_height)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return ImageTk.PhotoImage(pil)

    def _update_color_swatch(self) -> None:
        bgr = self._hsv_to_bgr(self.h_var.get(), self.s_var.get(), self.v_var.get())
        hex_value = f"#{int(bgr[2]):02x}{int(bgr[1]):02x}{int(bgr[0]):02x}"
        self.color_hex_var.set(hex_value)
        target_type = self._target_color_type(self.s_var.get(), self.v_var.get())
        self.color_label_var.set(f"Cor HSV: H={self.h_var.get()} S={self.s_var.get()} V={self.v_var.get()} ({target_type})")
        self.color_swatch.configure(bg=hex_value)

    @staticmethod
    def _target_color_type(s_value: int, v_value: int) -> str:
        if v_value <= 35:
            return "preta"
        if s_value <= 25 and v_value >= 220:
            return "branca"
        return "colorida"

    @staticmethod
    def _hsv_to_bgr(h: int, s: int, v: int) -> np.ndarray:
        sample = np.uint8([[[int(h) % 180, int(np.clip(s, 0, 255)), int(np.clip(v, 0, 255))]]])
        return cv2.cvtColor(sample, cv2.COLOR_HSV2BGR)[0, 0]

    def choose_hsv_color(self) -> None:
        selected = colorchooser.askcolor(title="Escolher cor do filtro")
        if not selected or selected[1] is None or selected[0] is None:
            return

        r, g, b = (int(round(v)) for v in selected[0])
        hsv = cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV)[0, 0]
        self.h_var.set(int(hsv[0]))
        self.s_var.set(int(hsv[1]))
        self.v_var.set(int(hsv[2]))
        self.processing_mode.set("recolor")
        target_type = self._target_color_type(self.s_var.get(), self.v_var.get())
        if target_type == "branca":
            self.achromatic_sat_keep.set(0.0)
            self.achromatic_lum_scale.set(1.35)
        elif target_type == "preta":
            self.achromatic_sat_keep.set(0.0)
            self.achromatic_lum_scale.set(0.35)
        else:
            self.achromatic_sat_keep.set(1.0)
            self.achromatic_lum_scale.set(1.0)
        self._update_color_swatch()
        self.refresh_preview()

    def _reference_bgr(self) -> np.ndarray:
        return self._load_bgr(self.reference_path.get().strip())

    def _apply_hsv_recolor(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV).astype(np.uint8)
        target_hue = int(self.h_var.get()) % 180
        shift = int(self.hue_shift.get())
        final_hue = int((target_hue + shift) % 180)

        target_type = self._target_color_type(self.s_var.get(), self.v_var.get())
        hsv_new = hsv.copy()

        # Recolor only chromatic pixels to preserve neutral backgrounds.
        mask_chromatic = hsv[:, :, 1] > 18

        if target_type == "colorida":
            hsv_new[:, :, 0][mask_chromatic] = final_hue
            return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

        sat_keep = float(np.clip(self.achromatic_sat_keep.get(), 0.0, 1.0))
        lum_scale = float(np.clip(self.achromatic_lum_scale.get(), 0.1, 2.2))

        s_channel = hsv_new[:, :, 1].astype(np.float32)
        v_channel = hsv_new[:, :, 2].astype(np.float32)
        s_channel[mask_chromatic] = np.clip(s_channel[mask_chromatic] * sat_keep, 0, 255)

        if target_type == "branca":
            # Lift value while preserving relative shading structure.
            v_channel[mask_chromatic] = np.clip(v_channel[mask_chromatic] * lum_scale, 0, 255)
        else:
            # Darken value for black while preserving texture contrast.
            v_channel[mask_chromatic] = np.clip(v_channel[mask_chromatic] * lum_scale, 0, 255)

        hsv_new[:, :, 1] = s_channel.astype(np.uint8)
        hsv_new[:, :, 2] = v_channel.astype(np.uint8)
        return cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)

    def _process_frame(self, frame_bgr: np.ndarray) -> FrameResult:
        mode = self.processing_mode.get().strip().lower()
        if mode == "matching":
            reference = self._reference_bgr()
            processed = self.processor.histogram_match_bgr(frame_bgr, reference)
            ssim_value = self.processor.compute_ssim_bgr(frame_bgr, processed)
            return FrameResult(frame_bgr, processed, ssim_value, "MATCHING")

        if mode == "hsv_match":
            reference = self._reference_bgr()
            matched = self.processor.histogram_match_bgr(frame_bgr, reference)
            processed = self._apply_hsv_recolor(matched)
            ssim_value = self.processor.compute_ssim_bgr(frame_bgr, processed)
            return FrameResult(frame_bgr, processed, ssim_value, "HSV_MATCH")

        if mode == "hsv":
            hsv_preview = self.processor.hsv_preview_bgr(frame_bgr)
            roundtrip = cv2.cvtColor(self.processor.bgr_to_hsv(frame_bgr), cv2.COLOR_HSV2BGR)
            ssim_value = self.processor.compute_ssim_bgr(frame_bgr, roundtrip)
            return FrameResult(frame_bgr, hsv_preview, ssim_value, "HSV")

        if mode == "onyx":
            effect = self.effects.apply_onyx_effect(frame_bgr, gamma=float(self.onyx_gamma.get()))
            ssim_value = self.processor.compute_ssim_bgr(frame_bgr, effect.image_bgr)
            return FrameResult(frame_bgr, effect.image_bgr, ssim_value, "ONYX")

        if mode == "pearl":
            effect = self.effects.apply_pearl_effect(
                frame_bgr,
                contrast=float(self.pearl_contrast.get()),
                brightness=float(self.pearl_brightness.get()),
            )
            ssim_value = self.processor.compute_ssim_bgr(frame_bgr, effect.image_bgr)
            return FrameResult(frame_bgr, effect.image_bgr, ssim_value, "PEARL")

        processed = self._apply_hsv_recolor(frame_bgr)
        ssim_value = self.processor.compute_ssim_bgr(frame_bgr, processed)
        return FrameResult(frame_bgr, processed, ssim_value, "RECOLOR")

    def _read_current_source(self) -> tuple[str, Optional[cv2.VideoCapture], Optional[np.ndarray]]:
        mode = self.source_mode.get().strip().lower()
        if mode == "imagem":
            path = self.image_path.get().strip()
            if not path:
                raise ValueError("Selecione uma imagem valida.")
            return mode, None, self._load_bgr(path)

        if mode == "webcam":
            capture = cv2.VideoCapture(int(self.webcam_index.get()))
            if not capture.isOpened():
                raise ValueError("Nao foi possivel abrir a webcam.")
            return mode, capture, None

        path = self.video_path.get().strip()
        if not path:
            raise ValueError("Selecione um video valido.")
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            raise ValueError("Nao foi possivel abrir o video.")
        return mode, capture, None

    def start_preview(self) -> None:
        self.stop_preview()
        try:
            mode, capture, static_frame = self._read_current_source()
            if mode == "imagem":
                self._current_frame_bgr = static_frame
                self._render_current_frame(static_frame)
                self.status_var.set("Preview de imagem atualizado.")
                return

            self._capture = capture
            self._streaming = True
            self.status_var.set(f"Preview iniciado em {mode}.")
            self._stream_step()
        except Exception as exc:
            messagebox.showerror("Erro", str(exc))
            self.status_var.set("Falha ao iniciar preview.")

    def _stream_step(self) -> None:
        if not self._streaming or self._capture is None:
            return

        ok, frame = self._capture.read()
        if not ok:
            if self.source_mode.get().strip().lower() == "video":
                self.status_var.set("Fim do video alcançado.")
            self.stop_preview()
            return

        self._current_frame_bgr = frame
        self._render_current_frame(frame)
        self._after_id = self.root.after(33, self._stream_step)

    def _render_current_frame(self, frame_bgr: np.ndarray) -> None:
        result = self._process_frame(frame_bgr)
        original_photo = self._bgr_to_photoimage(result.original_bgr)
        processed_photo = self._bgr_to_photoimage(result.processed_bgr)
        self._original_photo = original_photo
        self._processed_photo = processed_photo
        self.original_label.configure(image=original_photo)
        self.processed_label.configure(image=processed_photo)
        self.ssim_var.set(f"SSIM: {result.ssim_value:.4f}")
        self.gpu_var.set(f"{self.processor.backend_message} | {self.effects.build_backend_message()}")
        self.status_var.set(f"Modo {result.mode_name} atualizado.")

    def refresh_preview(self) -> None:
        if self.source_mode.get().strip().lower() == "imagem":
            try:
                path = self.image_path.get().strip()
                if not path:
                    return
                self._current_frame_bgr = self._load_bgr(path)
                self._render_current_frame(self._current_frame_bgr)
            except Exception as exc:
                self.status_var.set(str(exc))
        elif self._streaming and self._capture is not None:
            if self._current_frame_bgr is not None:
                self._render_current_frame(self._current_frame_bgr)

    def stop_preview(self) -> None:
        self._streaming = False
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self._capture is not None:
            try:
                self._capture.release()
            except Exception:
                pass
            self._capture = None

    def save_image(self) -> None:
        try:
            if self._current_frame_bgr is None:
                self.start_preview()
                if self._current_frame_bgr is None:
                    raise ValueError("Nenhum frame disponivel para salvar.")

            result = self._process_frame(self._current_frame_bgr)
            output_dir = self._resolve_output_dir()
            os.makedirs(output_dir, exist_ok=True)

            base_name = self.image_path.get().strip() or self.video_path.get().strip() or f"webcam_{self.webcam_index.get()}"
            source_name = os.path.splitext(os.path.basename(base_name))[0] or "frame"
            output_path = os.path.join(output_dir, f"{source_name}_{result.mode_name.lower()}.png")
            if not cv2.imwrite(output_path, result.processed_bgr):
                raise IOError(f"Nao foi possivel salvar em: {output_path}")

            self.ssim_var.set(f"SSIM: {result.ssim_value:.4f}")
            self.status_var.set(f"Imagem salva em {output_path}")
            messagebox.showinfo("Sucesso", f"Imagem salva em:\n{output_path}\n\nSSIM: {result.ssim_value:.4f}")
        except Exception as exc:
            messagebox.showerror("Erro", str(exc))
            self.status_var.set("Falha ao salvar imagem.")

    def save_video(self) -> None:
        try:
            mode = self.source_mode.get().strip().lower()
            if mode == "imagem":
                raise ValueError("Selecione video ou webcam para salvar um video.")

            if self._capture is None:
                self.start_preview()
            capture = self._capture
            if capture is None:
                raise ValueError("Nao foi possivel acessar a fonte de video.")

            current_frame = self._current_frame_bgr
            if current_frame is None:
                raise ValueError("Nenhum frame disponivel para processamento.")

            output_dir = self._resolve_output_dir()
            os.makedirs(output_dir, exist_ok=True)
            source_name = os.path.splitext(os.path.basename(self.video_path.get().strip() or f"webcam_{self.webcam_index.get()}"))[0] or "video"
            output_path = os.path.join(output_dir, f"{source_name}_{self.processing_mode.get().strip().lower()}.mp4")

            fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or current_frame.shape[1])
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or current_frame.shape[0])
            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True)
            if not writer.isOpened():
                raise ValueError("Nao foi possivel criar o arquivo de saida do video.")

            self.status_var.set("Salvando video atual...")
            temp_capture = cv2.VideoCapture(int(self.webcam_index.get())) if mode == "webcam" else cv2.VideoCapture(self.video_path.get().strip())
            if not temp_capture.isOpened():
                writer.release()
                raise ValueError("Nao foi possivel reabrir a fonte para exportacao.")

            ssim_values: list[float] = []
            while True:
                ok, frame = temp_capture.read()
                if not ok:
                    break
                result = self._process_frame(frame)
                ssim_values.append(result.ssim_value)
                writer.write(cv2.resize(result.processed_bgr, (width, height), interpolation=cv2.INTER_AREA))

            temp_capture.release()
            writer.release()
            self.status_var.set(f"Video salvo em {output_path}")
            if ssim_values:
                self.ssim_var.set(f"SSIM: {float(np.mean(ssim_values)):.4f}")
            messagebox.showinfo("Sucesso", f"Video salvo em:\n{output_path}")
        except Exception as exc:
            messagebox.showerror("Erro", str(exc))
            self.status_var.set("Falha ao salvar video.")

    def show_transfer_curves(self) -> None:
        try:
            plot_effect_transfer_curves(
                gamma=float(self.onyx_gamma.get()),
                contrast=float(self.pearl_contrast.get()),
                brightness=float(self.pearl_brightness.get()),
            )
            self.status_var.set("Grafico de curvas exibido com sucesso.")
        except Exception as exc:
            messagebox.showerror("Erro", str(exc))
            self.status_var.set("Falha ao plotar curvas.")

    def save_transfer_curves_png(self) -> None:
        try:
            output_dir = self._resolve_output_dir()
            os.makedirs(output_dir, exist_ok=True)
            default_name = f"curvas_onyx_pearl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            chosen_name = simpledialog.askstring(
                "Nome do PNG",
                "Digite o nome do arquivo PNG para salvar as curvas:",
                initialvalue=default_name,
                parent=self.root,
            )
            if chosen_name is None:
                return
            chosen_name = chosen_name.strip() or default_name
            safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_", " ") else "_" for ch in chosen_name).strip()
            if not safe_name:
                safe_name = default_name
            filename = safe_name if safe_name.lower().endswith(".png") else f"{safe_name}.png"
            save_path = os.path.join(output_dir, filename)
            written = plot_effect_transfer_curves(
                gamma=float(self.onyx_gamma.get()),
                contrast=float(self.pearl_contrast.get()),
                brightness=float(self.pearl_brightness.get()),
                save_path=save_path,
                show=False,
            )
            if not written:
                raise IOError("Nao foi possivel salvar o grafico de curvas.")
            self.status_var.set(f"Curvas salvas em {written}")
            messagebox.showinfo("Sucesso", f"Curvas salvas em:\n{written}")
        except Exception as exc:
            messagebox.showerror("Erro", str(exc))
            self.status_var.set("Falha ao salvar curvas em PNG.")

    def _on_close(self) -> None:
        self.stop_preview()
        self.root.destroy()


def run_color_app() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    ColorAppGUI(root)
    root.mainloop()
