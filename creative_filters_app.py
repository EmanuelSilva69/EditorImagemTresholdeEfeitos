"""
Editor de Filtros Criativos.

Fundamentação resumida para relatório:
- Filtros de cor: negativo e sépia são transformações pontuais que alteram a distribuição cromática.
- Suavização/realce: blur reduz alta frequência; sharpen aumenta contraste local e resposta a bordas.
- Aplicações: estilização, prototipação visual e experimentação de efeitos em imagens e vídeos.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

if hasattr(Image, "Resampling"):
    RESAMPLING = Image.Resampling.LANCZOS
else:
    RESAMPLING = Image.LANCZOS


class CreativeFiltersApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Editor de Filtros Criativos")
        self.root.geometry("1180x760")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.image_path = tk.StringVar(value="")
        self.video_path = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value=self._default_output_dir())
        self.mode = tk.StringVar(value="imagem")
        self.filter_mode = tk.StringVar(value="none")
        self.hsv_h = tk.IntVar(value=0)
        self.hsv_s = tk.IntVar(value=255)
        self.hsv_v = tk.IntVar(value=255)
        self.selected_color_hex = tk.StringVar(value="#ffffff")
        self.selected_color_text = tk.StringVar(value="Nenhuma cor HSV selecionada")
        self.blur_kernel = tk.IntVar(value=7)
        self.sharpen_strength = tk.DoubleVar(value=0.8)
        self.status = tk.StringVar(value="Selecione uma imagem ou video para começar.")

        self._last_processed: np.ndarray | None = None
        self._build_ui()

    def _default_output_dir(self) -> str:
        return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resultados", "creative"))

    def _resolve_output_dir(self) -> str:
        pasta = self.output_dir.get().strip()
        if not pasta:
            return self._default_output_dir()
        if os.path.isabs(pasta):
            return os.path.normpath(pasta)
        return os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), pasta))

    def _build_ui(self) -> None:
        topo = ttk.Frame(self.root, padding=10)
        topo.pack(fill="x")

        ttk.Button(topo, text="Selecionar imagem", command=self.select_image).pack(side="left")
        ttk.Button(topo, text="Selecionar video", command=self.select_video).pack(side="left", padx=8)
        ttk.Button(topo, text="Selecionar saida", command=self.select_output_dir).pack(side="left", padx=8)
        ttk.Button(topo, text="Escolher cor HSV", command=self.open_hsv_picker).pack(side="left", padx=8)

        params = ttk.LabelFrame(self.root, text="Filtros Criativos", padding=10)
        params.pack(fill="x", padx=10, pady=(0, 10))

        l1 = ttk.Frame(params)
        l1.pack(fill="x", pady=2)
        ttk.Label(l1, text="Modo:").pack(side="left")
        ttk.Combobox(l1, textvariable=self.mode, values=["imagem", "video"], state="readonly", width=12).pack(side="left", padx=5)
        ttk.Label(l1, text="Filtro:").pack(side="left", padx=(20, 0))
        ttk.Combobox(l1, textvariable=self.filter_mode, values=["none", "negative", "sepia", "blur", "sharpen", "hsv_tint"], state="readonly", width=12).pack(side="left", padx=5)

        l2 = ttk.Frame(params)
        l2.pack(fill="x", pady=2)
        ttk.Label(l2, text="Kernel blur:").pack(side="left")
        ttk.Scale(l2, from_=3, to=31, variable=self.blur_kernel, orient="horizontal", length=240).pack(side="left", padx=5)
        ttk.Label(l2, textvariable=self.blur_kernel, width=4).pack(side="left")
        ttk.Label(l2, text="Sharpen:").pack(side="left", padx=(20, 0))
        ttk.Scale(l2, from_=0.0, to=2.0, variable=self.sharpen_strength, orient="horizontal", length=240).pack(side="left", padx=5)
        ttk.Label(l2, textvariable=self.sharpen_strength, width=4).pack(side="left")

        l3 = ttk.Frame(params)
        l3.pack(fill="x", pady=2)
        ttk.Label(l3, text="Imagem:").pack(side="left")
        ttk.Entry(l3, textvariable=self.image_path, width=90).pack(side="left", padx=5, fill="x", expand=True)
        ttk.Label(l3, text="Video:").pack(side="left", padx=(20, 0))
        ttk.Entry(l3, textvariable=self.video_path, width=90).pack(side="left", padx=5, fill="x", expand=True)

        l4 = ttk.Frame(params)
        l4.pack(fill="x", pady=2)
        ttk.Label(l4, text="Saida:").pack(side="left")
        ttk.Entry(l4, textvariable=self.output_dir, width=90).pack(side="left", padx=5, fill="x", expand=True)
        ttk.Label(l4, textvariable=self.status).pack(side="left", padx=10)

        l5 = ttk.Frame(params)
        l5.pack(fill="x", pady=2)
        ttk.Label(l5, textvariable=self.selected_color_text).pack(side="left")
        ttk.Label(l5, textvariable=self.selected_color_hex, width=10).pack(side="left", padx=8)

        acao = ttk.Frame(self.root, padding=10)
        acao.pack(fill="x")
        ttk.Button(acao, text="Visualizar imagem", command=self.preview_image).pack(side="left")
        ttk.Button(acao, text="Salvar imagem", command=self.save_image).pack(side="left", padx=8)
        ttk.Button(acao, text="Processar video", command=self.process_video).pack(side="left", padx=8)

    def _on_close(self) -> None:
        self._close_all_windows()
        self.root.destroy()

    def _close_all_windows(self) -> None:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    @staticmethod
    def _hsv_to_bgr(h: int, s: int, v: int) -> np.ndarray:
        sample = np.uint8([[[h, s, v]]])
        return cv2.cvtColor(sample, cv2.COLOR_HSV2BGR)[0, 0]

    def _update_selected_color(self, preview_label: tk.Label) -> None:
        bgr = self._hsv_to_bgr(self.hsv_h.get(), self.hsv_s.get(), self.hsv_v.get())
        hex_value = f"#{int(bgr[2]):02x}{int(bgr[1]):02x}{int(bgr[0]):02x}"
        self.selected_color_hex.set(hex_value)
        self.selected_color_text.set(f"HSV selecionado: H={self.hsv_h.get()} S={self.hsv_s.get()} V={self.hsv_v.get()}")
        preview_label.configure(bg=hex_value)

    def open_hsv_picker(self) -> None:
        picker = tk.Toplevel(self.root)
        picker.title("Selecionar cor HSV")
        picker.geometry("420x320")
        picker.resizable(False, False)
        picker.transient(self.root)
        picker.grab_set()

        container = ttk.Frame(picker, padding=12)
        container.pack(fill="both", expand=True)

        preview = tk.Label(container, text="", width=18, height=4, relief="solid", bd=1)
        preview.pack(fill="x", pady=(0, 12))

        def add_slider(label: str, variable: tk.IntVar, maximum: int) -> None:
            row = ttk.Frame(container)
            row.pack(fill="x", pady=4)
            ttk.Label(row, text=label, width=6).pack(side="left")
            scale = tk.Scale(
                row,
                from_=0,
                to=maximum,
                orient="horizontal",
                variable=variable,
                showvalue=True,
                length=260,
                command=lambda _value: self._update_selected_color(preview),
            )
            scale.pack(side="left", fill="x", expand=True)

        add_slider("Hue", self.hsv_h, 179)
        add_slider("Sat", self.hsv_s, 255)
        add_slider("Val", self.hsv_v, 255)

        self._update_selected_color(preview)

        buttons = ttk.Frame(container)
        buttons.pack(fill="x", pady=(12, 0))

        def confirm() -> None:
            self.filter_mode.set("hsv_tint")
            self.status.set("Cor HSV selecionada para o filtro.")
            picker.destroy()

        ttk.Button(buttons, text="Aplicar", command=confirm).pack(side="left")
        ttk.Button(buttons, text="Cancelar", command=picker.destroy).pack(side="left", padx=8)

        picker.protocol("WM_DELETE_WINDOW", picker.destroy)

    def select_image(self) -> None:
        path = filedialog.askopenfilename(title="Selecione imagem", filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp"), ("Todos os arquivos", "*.*")])
        if path:
            self.image_path.set(path)
            self.status.set(f"Imagem: {os.path.basename(path)}")

    def select_video(self) -> None:
        path = filedialog.askopenfilename(title="Selecione video", filetypes=[("Videos", "*.mp4;*.avi;*.mov;*.mkv"), ("Todos os arquivos", "*.*")])
        if path:
            self.video_path.set(path)
            self.status.set(f"Video: {os.path.basename(path)}")

    def select_output_dir(self) -> None:
        path = filedialog.askdirectory(title="Selecione pasta de saida")
        if path:
            self.output_dir.set(os.path.normpath(path))

    @staticmethod
    def _ensure_odd(value: int) -> int:
        value = max(3, int(value))
        return value if value % 2 == 1 else value + 1

    @staticmethod
    def _apply_filter(frame: np.ndarray, filter_mode: str, blur_kernel: int, sharpen_strength: float) -> np.ndarray:
        if filter_mode == "negative":
            return 255 - frame
        if filter_mode == "sepia":
            kernel = np.array([[0.131, 0.534, 0.272], [0.168, 0.686, 0.349], [0.189, 0.769, 0.393]], dtype=np.float32)
            return np.clip(cv2.transform(frame.astype(np.float32), kernel), 0, 255).astype(np.uint8)
        if filter_mode == "blur":
            k = CreativeFiltersApp._ensure_odd(blur_kernel)
            return cv2.GaussianBlur(frame, (k, k), 0)
        if filter_mode == "sharpen":
            strength = float(sharpen_strength)
            kernel = np.array([[0, -1, 0], [-1, 5 + strength, -1], [0, -1, 0]], dtype=np.float32)
            return np.clip(cv2.filter2D(frame, -1, kernel), 0, 255).astype(np.uint8)
        return frame

    def _apply_hsv_tint(self, frame: np.ndarray) -> np.ndarray:
        tint_bgr = self._hsv_to_bgr(self.hsv_h.get(), self.hsv_s.get(), self.hsv_v.get()).astype(np.float32)
        base = frame.astype(np.float32)
        overlay = np.full_like(base, tint_bgr, dtype=np.float32)
        return np.clip(base * 0.72 + overlay * 0.28, 0, 255).astype(np.uint8)

    def _read_image(self) -> np.ndarray:
        path = self.image_path.get().strip()
        if not path or not os.path.exists(path):
            raise ValueError("Selecione uma imagem valida.")
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Nao foi possivel ler a imagem.")
        return frame

    def preview_image(self) -> None:
        try:
            frame = self._read_image()
            processed = self._apply_hsv_tint(frame) if self.filter_mode.get() == "hsv_tint" else self._apply_filter(frame, self.filter_mode.get(), self.blur_kernel.get(), self.sharpen_strength.get())
            side_by_side = np.hstack([frame, processed])
            window_name = "Editor de Filtros Criativos"
            cv2.imshow(window_name, side_by_side)
            while True:
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        except Exception as exc:
            messagebox.showerror("Erro", str(exc))
        finally:
            self._close_all_windows()

    def save_image(self) -> None:
        try:
            frame = self._read_image()
            processed = self._apply_hsv_tint(frame) if self.filter_mode.get() == "hsv_tint" else self._apply_filter(frame, self.filter_mode.get(), self.blur_kernel.get(), self.sharpen_strength.get())
            output_dir = self._resolve_output_dir()
            os.makedirs(output_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(self.image_path.get().strip()))[0]
            output_path = os.path.join(output_dir, f"{base}_{self.filter_mode.get()}.png")
            if not cv2.imwrite(output_path, processed):
                raise IOError(f"Falha ao salvar em: {output_path}")
            self.status.set(f"Salvo em {output_path}")
            messagebox.showinfo("Sucesso", f"Imagem salva em:\n{output_path}")
        except Exception as exc:
            messagebox.showerror("Erro", str(exc))

    def process_video(self) -> None:
        try:
            path = self.video_path.get().strip()
            if not path:
                raise ValueError("Selecione um video valido.")
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise ValueError("Nao foi possivel abrir o video.")
            out_dir = self._resolve_output_dir()
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(out_dir, f"{base}_{self.filter_mode.get()}.mp4")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True)
            if not writer.isOpened():
                cap.release()
                raise ValueError("Nao foi possivel criar o arquivo de saida.")
            self.status.set("Processando video...")
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                processed = self._apply_hsv_tint(frame) if self.filter_mode.get() == "hsv_tint" else self._apply_filter(frame, self.filter_mode.get(), self.blur_kernel.get(), self.sharpen_strength.get())
                writer.write(processed)
                window_name = "Editor de Filtros Criativos"
                cv2.imshow(window_name, np.hstack([frame, processed]))
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            cap.release()
            writer.release()
            self._close_all_windows()
            self.status.set(f"Video salvo em {out_path}")
        except Exception as exc:
            self._close_all_windows()
            messagebox.showerror("Erro", str(exc))


def run_creative_filters_app() -> None:
    root = tk.Tk()
    app = CreativeFiltersApp(root)
    _ = app
    root.mainloop()
