import os
import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


if hasattr(Image, "Resampling"):
    RESAMPLING = Image.Resampling.LANCZOS
else:
    RESAMPLING = Image.LANCZOS

from processamento_thresholds import (
    mostrar_comparativo_selecionados,
    mostrar_analise_avancada,
    mostrar_histogramas_filtros,
    salvar_histogramas_individuais,
    salvar_resultados_por_metodo,
    mostrar_resultados_lado_a_lado,
    preprocessar,
    threshold_metodos_globais,
    threshold_adaptativo_local,
    threshold_estatistico,
    threshold_multi_otsu,
    threshold_range,
)
from filtros_cor import COR_PARA_HUE, aplicar_filtros_selecionados, memory_overflow_glitch, salvar_troca_cor
from filtros_cor import detectar_hue_principal, shift_color


class AppThresholdGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PDI - Thresholding Local (Tkinter)")
        self.root.geometry("1200x760")
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        self.imagens: list[str] = []
        self.preview_images: list[ImageTk.PhotoImage] = []
        self.preview_color_images: list[ImageTk.PhotoImage] = []
        self.caminho_cor: str = ""
        self.modo_preview_cor: str = "troca"
        self._preview_cor_after_id = None
        self._ultima_preview_cor_bgr: np.ndarray | None = None

        self._build_ui()

    def _create_scrollable_tab(self, parent: ttk.Notebook) -> ttk.Frame:
        """Cria uma área rolável para conteúdos longos dentro de uma aba."""
        container = ttk.Frame(parent)
        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas)

        content_window = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_configure(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(content_window, width=event.width)

        content.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        container.canvas = canvas  # type: ignore[attr-defined]
        container.content = content  # type: ignore[attr-defined]
        return container

    def _toggle_section(self, frame: ttk.Widget, button: ttk.Button, shown_text: str, hidden_text: str, visible: bool, pack_kwargs: dict) -> bool:
        """Mostra ou esconde uma seção mantendo o botão como gatilho."""
        if visible:
            frame.pack_forget()
            button.configure(text=hidden_text)
            return False

        frame.pack(**pack_kwargs)
        button.configure(text=shown_text)
        return True

    def _video_output_dir_padrao(self) -> str:
        return os.path.normpath(os.path.join(self.base_dir, "..", "resultados", "video"))

    def _resolver_video_output_dir(self) -> str:
        pasta = self.var_video_output_dir.get().strip()
        if not pasta:
            return self._video_output_dir_padrao()
        if os.path.isabs(pasta):
            return os.path.normpath(pasta)
        return os.path.normpath(os.path.join(self.base_dir, pasta))

    def _schedule_preview_cor_refresh(self, *_args: object) -> None:
        if self._preview_cor_after_id is not None:
            self.root.after_cancel(self._preview_cor_after_id)
        self._preview_cor_after_id = self.root.after(90, self._refresh_preview_cor_now)

    def _refresh_preview_cor_now(self) -> None:
        self._preview_cor_after_id = None
        self._atualizar_preview_cor()

    def _build_ui(self) -> None:
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True)

        self.tab_threshold = ttk.Frame(self.tabs)
        self.tab_cor = ttk.Frame(self.tabs)
        self.tab_video = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_threshold, text="Threshold")
        self.tabs.add(self.tab_cor, text="Troca de Cor")
        self.tabs.add(self.tab_video, text="Video")

        self.threshold_shell = self._create_scrollable_tab(self.tab_threshold)
        self.threshold_shell.pack(fill="both", expand=True)
        self.threshold_content = self.threshold_shell.content  # type: ignore[attr-defined]

        self.cor_shell = self._create_scrollable_tab(self.tab_cor)
        self.cor_shell.pack(fill="both", expand=True)
        self.cor_content = self.cor_shell.content  # type: ignore[attr-defined]

        self.video_shell = self._create_scrollable_tab(self.tab_video)
        self.video_shell.pack(fill="both", expand=True)
        self.video_content = self.video_shell.content  # type: ignore[attr-defined]

        frame_top = ttk.Frame(self.threshold_content, padding=10)
        frame_top.pack(fill="x")

        ttk.Button(frame_top, text="Selecionar imagens", command=self.selecionar_imagens).pack(side="left")
        ttk.Button(frame_top, text="Selecionar pasta", command=self.selecionar_pasta).pack(side="left", padx=8)
        ttk.Button(frame_top, text="Remover selecionada", command=self.remover_imagem).pack(
            side="left", padx=8
        )
        ttk.Button(frame_top, text="Limpar lista", command=self.limpar_lista).pack(side="left")

        frame_lista = ttk.LabelFrame(self.threshold_content, text="Imagens selecionadas", padding=10)
        frame_lista.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.lista = tk.Listbox(frame_lista, height=8)
        self.lista.pack(fill="both", expand=True)
        self.lista.bind("<<ListboxSelect>>", self._on_selecao_lista)

        frame_preview = ttk.LabelFrame(self.threshold_content, text="Preview (imagem selecionada)", padding=10)
        frame_preview.pack(fill="x", padx=10, pady=(0, 10))

        self.preview_labels = []
        preview_titulos = ["Original", "Adaptativo", "Multi-Otsu", "Range", "Estatistico"]
        for col, titulo in enumerate(preview_titulos):
            slot = ttk.Frame(frame_preview)
            slot.grid(row=0, column=col, padx=4, sticky="n")
            ttk.Label(slot, text=titulo).pack()
            lbl = ttk.Label(slot)
            lbl.pack()
            self.preview_labels.append(lbl)

        for col in range(5):
            frame_preview.columnconfigure(col, weight=1)

        frame_params = ttk.LabelFrame(self.threshold_content, text="Parametros", padding=10)
        frame_params.pack(fill="x", padx=10, pady=(0, 10))

        self.var_preprocess = tk.BooleanVar(value=False)
        self.var_equalize = tk.BooleanVar(value=False)
        self.var_clahe = tk.BooleanVar(value=False)
        self.var_stat_local = tk.BooleanVar(value=False)
        self.var_save_figure = tk.BooleanVar(value=False)
        self.var_save_individual = tk.BooleanVar(value=True)
        self.var_show_comparison = tk.BooleanVar(value=False)
        self.var_show_analysis = tk.BooleanVar(value=True)
        self.var_save_analysis = tk.BooleanVar(value=False)
        self.var_save_extra_methods = tk.BooleanVar(value=True)
        self.var_save_histograms = tk.BooleanVar(value=True)
        self.var_range_invert = tk.BooleanVar(value=False)

        self.var_blur = tk.StringVar(value="gaussian")
        self.var_adaptive_method = tk.StringVar(value="gaussian")
        self.var_adaptive_polarity = tk.StringVar(value="above")
        self.var_multi_mode = tk.StringVar(value="levels")
        self.var_stat_polarity = tk.StringVar(value="above")
        self.var_figure_dir = tk.StringVar(value="resultados")
        self.var_filtros_visiveis = tk.BooleanVar(value=False)

        self.method_vars: dict[str, tk.BooleanVar] = {
            "adaptativo": tk.BooleanVar(value=True),
            "multi_otsu": tk.BooleanVar(value=True),
            "range": tk.BooleanVar(value=True),
            "estatistico": tk.BooleanVar(value=True),
            "yen": tk.BooleanVar(value=True),
            "triangle": tk.BooleanVar(value=True),
            "otsu": tk.BooleanVar(value=True),
            "minimum": tk.BooleanVar(value=True),
            "mean": tk.BooleanVar(value=True),
            "isodata": tk.BooleanVar(value=True),
        }

        self.entries = {}
        
        # Video tab variables
        self.video_entries: dict[str, tk.StringVar] = {}
        self.video_method_vars: dict[str, tk.BooleanVar] = {
            "adaptativo": tk.BooleanVar(value=True),
            "multi_otsu": tk.BooleanVar(value=True),
            "range": tk.BooleanVar(value=True),
            "estatistico": tk.BooleanVar(value=True),
            "yen": tk.BooleanVar(value=True),
            "triangle": tk.BooleanVar(value=True),
            "otsu": tk.BooleanVar(value=True),
            "minimum": tk.BooleanVar(value=True),
            "mean": tk.BooleanVar(value=True),
            "isodata": tk.BooleanVar(value=True),
        }
        self.var_video_path = tk.StringVar(value="")
        self.var_video_output_dir = tk.StringVar(value=self._video_output_dir_padrao())
        self.var_video_preprocess = tk.BooleanVar(value=False)
        self.var_video_equalize = tk.BooleanVar(value=False)
        self.var_video_clahe = tk.BooleanVar(value=False)
        self.var_video_stat_local = tk.BooleanVar(value=False)
        self.var_video_range_invert = tk.BooleanVar(value=False)
        self.var_video_blur = tk.StringVar(value="gaussian")
        self.var_video_adaptive_method = tk.StringVar(value="gaussian")
        self.var_video_adaptive_polarity = tk.StringVar(value="above")
        self.var_video_multi_mode = tk.StringVar(value="levels")
        self.var_video_stat_polarity = tk.StringVar(value="above")
        self.var_video_show_preview = tk.BooleanVar(value=True)

        linha1 = ttk.Frame(frame_params)
        linha1.pack(fill="x", pady=2)
        ttk.Checkbutton(linha1, text="Estatistico local", variable=self.var_stat_local).pack(side="left", padx=10)
        ttk.Checkbutton(linha1, text="Inverter Range", variable=self.var_range_invert).pack(side="left", padx=10)
        ttk.Checkbutton(linha1, text="Salvar cada filtro separado", variable=self.var_save_individual).pack(
            side="left", padx=10
        )
        ttk.Checkbutton(linha1, text="Mostrar comparativo", variable=self.var_show_comparison).pack(side="left", padx=10)
        ttk.Checkbutton(linha1, text="Mostrar analise avancada", variable=self.var_show_analysis).pack(side="left", padx=10)
        ttk.Checkbutton(linha1, text="Salvar analise avancada", variable=self.var_save_analysis).pack(side="left", padx=10)
        ttk.Checkbutton(linha1, text="Salvar metodos extras", variable=self.var_save_extra_methods).pack(side="left", padx=10)
        ttk.Checkbutton(linha1, text="Salvar histogramas", variable=self.var_save_histograms).pack(side="left", padx=10)

        linha2 = ttk.Frame(frame_params)
        linha2.pack(fill="x", pady=2)
        ttk.Label(linha2, text="Metodo adaptativo:").pack(side="left")
        ttk.Combobox(
            linha2,
            textvariable=self.var_adaptive_method,
            values=["gaussian", "mean", "median"],
            state="readonly",
            width=10,
        ).pack(side="left", padx=(5, 16))

        ttk.Label(linha2, text="Polaridade adaptativo:").pack(side="left")
        ttk.Combobox(
            linha2,
            textvariable=self.var_adaptive_polarity,
            values=["above", "below"],
            state="readonly",
            width=8,
        ).pack(side="left", padx=(5, 16))

        ttk.Label(linha2, text="Saida Multi-Otsu:").pack(side="left")
        ttk.Combobox(
            linha2,
            textvariable=self.var_multi_mode,
            values=["levels", "class"],
            state="readonly",
            width=8,
        ).pack(side="left", padx=(5, 16))

        ttk.Label(linha2, text="Polaridade estatistico:").pack(side="left")
        ttk.Combobox(
            linha2,
            textvariable=self.var_stat_polarity,
            values=["above", "below"],
            state="readonly",
            width=8,
        ).pack(side="left", padx=(5, 12))

        linha3 = ttk.Frame(frame_params)
        linha3.pack(fill="x", pady=2)

        ttk.Label(linha3, text="Pasta resultados:").pack(side="left")
        ttk.Entry(linha3, textvariable=self.var_figure_dir, width=25).pack(side="left", padx=5)
        ttk.Button(linha3, text="Selecionar output folder", command=self.selecionar_pasta_saida).pack(side="left", padx=6)
        ttk.Checkbutton(linha3, text="Salvar comparativo", variable=self.var_save_figure).pack(side="left", padx=10)
        ttk.Button(linha3, text="Atualizar preview", command=self._atualizar_preview_selecionada).pack(side="left", padx=10)
        self.btn_toggle_filtros = ttk.Button(
            linha3,
            text="Mostrar filtros extras",
            command=self._toggle_filtros_extras,
        )
        self.btn_toggle_filtros.pack(side="left", padx=10)

        frame_metodos = ttk.LabelFrame(frame_params, text="Thresholds a aplicar", padding=8)
        frame_metodos.pack(fill="x", pady=(6, 2))

        botoes_metodos = ttk.Frame(frame_metodos)
        botoes_metodos.pack(fill="x", pady=(0, 4))
        ttk.Button(botoes_metodos, text="Selecionar todos", command=self._selecionar_todos_metodos).pack(side="left")
        ttk.Button(botoes_metodos, text="Limpar selecao", command=self._limpar_selecao_metodos).pack(side="left", padx=8)

        grid_metodos = ttk.Frame(frame_metodos)
        grid_metodos.pack(fill="x")
        labels = [
            ("adaptativo", "Adaptativo"),
            ("multi_otsu", "Multi-Otsu"),
            ("range", "Range"),
            ("estatistico", "Estatistico"),
            ("yen", "Yen"),
            ("triangle", "Triangle"),
            ("otsu", "Otsu"),
            ("minimum", "Minimum"),
            ("mean", "Mean"),
            ("isodata", "ISODATA"),
        ]
        for i, (key, label) in enumerate(labels):
            ttk.Checkbutton(grid_metodos, text=label, variable=self.method_vars[key]).grid(row=i // 5, column=i % 5, sticky="w", padx=6, pady=2)

        linha4 = ttk.Frame(frame_params)
        linha4.pack(fill="x", pady=2)
        self._add_entry(linha4, "block_size", "Block size", "35")
        self._add_entry(linha4, "offset", "Offset", "10.0")
        self._add_entry(linha4, "classes", "Classes", "3")
        self._add_entry(linha4, "multi_target_class", "Classe alvo", "1")
        self._add_entry(linha4, "range_low", "Range low", "80")
        self._add_entry(linha4, "range_high", "Range high", "170")

        linha5 = ttk.Frame(frame_params)
        linha5.pack(fill="x", pady=2)
        self._add_entry(linha5, "k", "k estatistico", "0.5")
        self._add_entry(linha5, "stat_window", "Janela estat", "31")
        self._add_entry(linha5, "figure_width", "Largura px", "1920")
        self._add_entry(linha5, "figure_height", "Altura px", "1080")
        self._add_entry(linha5, "figure_dpi", "DPI", "100")

        self.btn_toggle_threshold_filters = ttk.Button(
            frame_params,
            text="Mostrar filtros extras",
            command=self._toggle_threshold_filters,
        )
        self.btn_toggle_threshold_filters.pack(fill="x", pady=(8, 4))

        self.frame_filtros_extras = ttk.LabelFrame(frame_params, text="Filtros extras (opcional)", padding=8)

        linha_fx1 = ttk.Frame(self.frame_filtros_extras)
        linha_fx1.pack(fill="x", pady=2)
        ttk.Checkbutton(linha_fx1, text="Pre-processar", variable=self.var_preprocess).pack(side="left")
        ttk.Checkbutton(linha_fx1, text="Equalizar histograma", variable=self.var_equalize).pack(side="left", padx=10)
        ttk.Checkbutton(linha_fx1, text="CLAHE", variable=self.var_clahe).pack(side="left", padx=10)

        linha_fx2 = ttk.Frame(self.frame_filtros_extras)
        linha_fx2.pack(fill="x", pady=2)
        ttk.Label(linha_fx2, text="Blur:").pack(side="left")
        ttk.Combobox(
            linha_fx2,
            textvariable=self.var_blur,
            values=["none", "gaussian", "median", "bilateral"],
            state="readonly",
            width=10,
        ).pack(side="left", padx=(5, 18))
        self._add_entry(linha_fx2, "blur_kernel", "Kernel blur", "5")
        self._add_entry(linha_fx2, "clahe_clip", "CLAHE clip", "2.0")
        self._add_entry(linha_fx2, "clahe_tile", "CLAHE tile", "8")

        self._threshold_filters_visible = False

        frame_acoes = ttk.Frame(self.threshold_content, padding=10)
        frame_acoes.pack(fill="x")

        ttk.Button(frame_acoes, text="Processar selecionada", command=self.processar_selecionada).pack(side="left")
        ttk.Button(frame_acoes, text="Processar todas", command=self.processar_todas).pack(side="left", padx=8)

        self.progress = ttk.Progressbar(frame_acoes, orient="horizontal", mode="determinate", length=280)
        self.progress.pack(side="left", padx=10)

        self.status = tk.StringVar(value="Pronto.")
        ttk.Label(self.threshold_content, textvariable=self.status, relief="sunken", anchor="w").pack(
            fill="x", side="bottom", ipady=4
        )

        self._build_color_tab()
        self._build_video_tab()

    def _build_video_tab(self) -> None:
        topo = ttk.Frame(self.video_content, padding=10)
        topo.pack(fill="x")

        ttk.Button(topo, text="Selecionar video", command=self.selecionar_video).pack(side="left")
        ttk.Button(topo, text="Selecionar pasta saida", command=self.selecionar_video_output_dir).pack(side="left", padx=8)
        ttk.Button(topo, text="Processar video", command=self.processar_video_gui).pack(side="left", padx=8)

        params = ttk.LabelFrame(self.video_content, text="Arquivo e Output", padding=10)
        params.pack(fill="x", padx=10, pady=(0, 10))

        l1 = ttk.Frame(params)
        l1.pack(fill="x", pady=2)
        ttk.Label(l1, text="Video:").pack(side="left")
        ttk.Entry(l1, textvariable=self.var_video_path, width=90).pack(side="left", padx=5, fill="x", expand=True)

        l2 = ttk.Frame(params)
        l2.pack(fill="x", pady=2)
        ttk.Label(l2, text="Pasta saida:").pack(side="left")
        ttk.Entry(l2, textvariable=self.var_video_output_dir, width=90).pack(side="left", padx=5, fill="x", expand=True)
        ttk.Checkbutton(l2, text="Mostrar preview (q para sair)", variable=self.var_video_show_preview).pack(side="left", padx=10)

        # Metodos threshold
        frame_metodos = ttk.LabelFrame(self.video_content, text="Thresholds a aplicar", padding=8)
        frame_metodos.pack(fill="x", padx=10, pady=(0, 10))

        botoes_metodos = ttk.Frame(frame_metodos)
        botoes_metodos.pack(fill="x", pady=(0, 4))
        ttk.Button(botoes_metodos, text="Selecionar todos", command=self._video_selecionar_todos).pack(side="left")
        ttk.Button(botoes_metodos, text="Limpar selecao", command=self._video_limpar_selecao).pack(side="left", padx=8)

        grid_metodos = ttk.Frame(frame_metodos)
        grid_metodos.pack(fill="x")
        labels = [
            ("adaptativo", "Adaptativo"),
            ("multi_otsu", "Multi-Otsu"),
            ("range", "Range"),
            ("estatistico", "Estatistico"),
            ("yen", "Yen"),
            ("triangle", "Triangle"),
            ("otsu", "Otsu"),
            ("minimum", "Minimum"),
            ("mean", "Mean"),
            ("isodata", "ISODATA"),
        ]
        for i, (key, label) in enumerate(labels):
            ttk.Checkbutton(grid_metodos, text=label, variable=self.video_method_vars[key]).grid(row=i // 5, column=i % 5, sticky="w", padx=6, pady=2)

        # Parametros
        frame_params = ttk.LabelFrame(self.video_content, text="Parametros de Threshold", padding=10)
        frame_params.pack(fill="x", padx=10, pady=(0, 10))

        linha1 = ttk.Frame(frame_params)
        linha1.pack(fill="x", pady=2)
        ttk.Checkbutton(linha1, text="Estatistico local", variable=self.var_video_stat_local).pack(side="left", padx=10)
        ttk.Checkbutton(linha1, text="Inverter Range", variable=self.var_video_range_invert).pack(side="left", padx=10)

        linha2 = ttk.Frame(frame_params)
        linha2.pack(fill="x", pady=2)
        ttk.Label(linha2, text="Metodo adaptativo:").pack(side="left")
        ttk.Combobox(
            linha2,
            textvariable=self.var_video_adaptive_method,
            values=["gaussian", "mean", "median"],
            state="readonly",
            width=10,
        ).pack(side="left", padx=(5, 16))

        ttk.Label(linha2, text="Polaridade adaptativo:").pack(side="left")
        ttk.Combobox(
            linha2,
            textvariable=self.var_video_adaptive_polarity,
            values=["above", "below"],
            state="readonly",
            width=8,
        ).pack(side="left", padx=(5, 16))

        ttk.Label(linha2, text="Saida Multi-Otsu:").pack(side="left")
        ttk.Combobox(
            linha2,
            textvariable=self.var_video_multi_mode,
            values=["levels", "class"],
            state="readonly",
            width=8,
        ).pack(side="left", padx=(5, 16))

        ttk.Label(linha2, text="Polaridade estatistico:").pack(side="left")
        ttk.Combobox(
            linha2,
            textvariable=self.var_video_stat_polarity,
            values=["above", "below"],
            state="readonly",
            width=8,
        ).pack(side="left", padx=(5, 12))

        linha3 = ttk.Frame(frame_params)
        linha3.pack(fill="x", pady=2)
        self._add_video_entry(linha3, "block_size", "Block size", "35")
        self._add_video_entry(linha3, "offset", "Offset", "10.0")
        self._add_video_entry(linha3, "classes", "Classes", "3")
        self._add_video_entry(linha3, "range_low", "Range low", "80")
        self._add_video_entry(linha3, "range_high", "Range high", "170")

        linha4 = ttk.Frame(frame_params)
        linha4.pack(fill="x", pady=2)
        self._add_video_entry(linha4, "multi_target_class", "Classe alvo", "1")
        self._add_video_entry(linha4, "k", "k estatistico", "0.5")
        self._add_video_entry(linha4, "stat_window", "Janela estat", "31")

        self.btn_toggle_video_filters = ttk.Button(
            frame_params,
            text="Mostrar filtros extras do vídeo",
            command=self._toggle_video_filters,
        )
        self.btn_toggle_video_filters.pack(fill="x", pady=(8, 4))

        # Filtros extras
        self.frame_video_filters = ttk.Frame(frame_params)
        linha_fx1 = ttk.Frame(self.frame_video_filters)
        linha_fx1.pack(fill="x", pady=2)
        ttk.Checkbutton(linha_fx1, text="Pre-processar", variable=self.var_video_preprocess).pack(side="left")
        ttk.Checkbutton(linha_fx1, text="Equalizar histograma", variable=self.var_video_equalize).pack(side="left", padx=10)
        ttk.Checkbutton(linha_fx1, text="CLAHE", variable=self.var_video_clahe).pack(side="left", padx=10)

        linha_fx2 = ttk.Frame(self.frame_video_filters)
        linha_fx2.pack(fill="x", pady=2)
        ttk.Label(linha_fx2, text="Blur:").pack(side="left")
        ttk.Combobox(
            linha_fx2,
            textvariable=self.var_video_blur,
            values=["none", "gaussian", "median", "bilateral"],
            state="readonly",
            width=10,
        ).pack(side="left", padx=(5, 18))
        self._add_video_entry(linha_fx2, "blur_kernel", "Kernel blur", "5")
        self._add_video_entry(linha_fx2, "clahe_clip", "CLAHE clip", "2.0")
        self._add_video_entry(linha_fx2, "clahe_tile", "CLAHE tile", "8")

        self._video_filters_visible = False

        self.progress_video = ttk.Progressbar(self.video_content, orient="horizontal", mode="determinate")
        self.progress_video.pack(fill="x", padx=10, pady=(0, 10))

        self.status_video = tk.StringVar(value="Selecione um video para iniciar.")
        ttk.Label(self.video_content, textvariable=self.status_video, relief="sunken", anchor="w").pack(fill="x", side="bottom", ipady=4)

    def selecionar_video(self) -> None:
        caminho = filedialog.askopenfilename(
            title="Selecione video",
            filetypes=[("Videos", "*.mp4;*.avi;*.mov;*.mkv"), ("Todos os arquivos", "*.*")],
        )
        if not caminho:
            return
        self.var_video_path.set(caminho)
        self.status_video.set(f"Video selecionado: {os.path.basename(caminho)}")

    def selecionar_video_output_dir(self) -> None:
        pasta = filedialog.askdirectory(title="Selecione pasta para salvar videos processados")
        if not pasta:
            return
        self.var_video_output_dir.set(os.path.normpath(pasta))

    def _aplicar_metodo_video(self, frame_gray: np.ndarray, metodo: str, params: dict) -> np.ndarray:
        """Aplica um método de threshold específico ao frame."""
        if metodo == "adaptativo":
            out, _ = threshold_adaptativo_local(
                frame_gray,
                block_size=params["block_size"],
                offset=params["offset"],
                metodo=params["adaptive_method"],
                polaridade=params["adaptive_polarity"],
            )
            return out
        elif metodo == "multi_otsu":
            out, _ = threshold_multi_otsu(
                frame_gray,
                classes=params["classes"],
                modo_saida=params["multi_mode"],
                classe_alvo=params["multi_target_class"],
            )
            return out
        elif metodo == "range":
            out, _ = threshold_range(
                frame_gray,
                baixo=params["range_low"],
                alto=params["range_high"],
                inverter=params["range_invert"],
            )
            return out
        elif metodo == "estatistico":
            out, _ = threshold_estatistico(
                frame_gray,
                k=params["k"],
                local=params["stat_local"],
                janela_local=params["stat_window"],
                polaridade=params["stat_polarity"],
            )
            return out
        else:
            # Métodos globais
            extras = threshold_metodos_globais(frame_gray, {metodo})
            if metodo in extras:
                return extras[metodo][0]
        return frame_gray

    def processar_video_gui(self) -> None:
        """Processa vídeo com múltiplos métodos de threshold."""
        caminho = self.var_video_path.get().strip()
        if not caminho:
            messagebox.showwarning("Aviso", "Selecione um video primeiro.")
            return
        if not os.path.exists(caminho):
            messagebox.showerror("Erro", "Video nao encontrado.")
            return

        try:
            params = self._ler_parametros_video()
        except Exception as exc:
            messagebox.showerror("Erro", str(exc))
            return

        metodos_selecionados = self._video_metodos_selecionados()
        if not metodos_selecionados:
            messagebox.showwarning("Aviso", "Selecione pelo menos um metodo de threshold.")
            return

        cap = cv2.VideoCapture(caminho)
        if not cap.isOpened():
            messagebox.showerror("Erro", "Nao foi possivel abrir o video.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Criar pasta de saida
        output_dir = self._resolver_video_output_dir()
        os.makedirs(output_dir, exist_ok=True)

        # Writers para cada metodo
        writers = {}
        for metodo in metodos_selecionados:
            output_path = os.path.join(output_dir, f"video_{metodo}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (largura, altura), True)
            writers[metodo] = writer

        self.progress_video.configure(maximum=max(1, total_frames), value=0)
        self.status_video.set("Processando video com multiplos metodos...")
        self.root.update_idletasks()

        frame_idx = 0
        show_preview = self.var_video_show_preview.get()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # Preprocessar frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_proc = preprocessar(
                gray,
                usar_preprocessamento=params["preprocess"],
                blur=params["blur"],
                blur_kernel=params["blur_kernel"],
                equalizar_histograma=params["equalize"],
                usar_clahe=params["clahe"],
                clahe_clip=params["clahe_clip"],
                clahe_tile=params["clahe_tile"],
            )

            # Aplicar cada metodo
            for metodo in metodos_selecionados:
                result = self._aplicar_metodo_video(gray_proc, metodo, params)
                result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
                writers[metodo].write(result_bgr)

            # Preview do primeiro metodo
            if show_preview:
                first_metodo = list(metodos_selecionados)[0]
                preview = self._aplicar_metodo_video(gray_proc, first_metodo, params)
                cv2.imshow(f"Preview - {first_metodo}", preview)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

            self.progress_video.configure(value=min(frame_idx, max(1, total_frames)))
            if frame_idx % 10 == 0:
                self.status_video.set(f"Processando frame {frame_idx}/{total_frames if total_frames > 0 else '?'}")
                self.root.update_idletasks()

        cap.release()
        for writer in writers.values():
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()

        self.status_video.set(f"Videos processados e salvos em: {output_dir}")
        messagebox.showinfo("Sucesso", f"Videos foram salvos em:\n{output_dir}\n\nMetodos: {', '.join(metodos_selecionados)}")

    def _add_video_entry(self, parent: ttk.Frame, key: str, label: str, default: str) -> None:
        """Adiciona um campo de entrada para parâmetro de vídeo."""
        ttk.Label(parent, text=label + ":").pack(side="left", padx=(0, 4))
        var = tk.StringVar(value=default)
        self.video_entries[key] = var
        ttk.Entry(parent, textvariable=var, width=8).pack(side="left", padx=(0, 12))

    def _video_selecionar_todos(self) -> None:
        """Seleciona todos os métodos de vídeo."""
        for var in self.video_method_vars.values():
            var.set(True)

    def _video_limpar_selecao(self) -> None:
        """Limpa seleção de todos os métodos de vídeo."""
        for var in self.video_method_vars.values():
            var.set(False)

    def _video_metodos_selecionados(self) -> set[str]:
        """Retorna conjunto de métodos de vídeo selecionados."""
        return {nome for nome, var in self.video_method_vars.items() if var.get()}

    def _toggle_threshold_filters(self) -> None:
        self._threshold_filters_visible = self._toggle_section(
            self.frame_filtros_extras,
            self.btn_toggle_threshold_filters,
            "Mostrar filtros extras",
            "Ocultar filtros extras",
            self._threshold_filters_visible,
            {"fill": "x", "pady": (8, 4)},
        )

    def _toggle_color_filters(self) -> None:
        self._color_filters_visible = self._toggle_section(
            self.frame_filtros_cor,
            self.btn_toggle_color_filters,
            "Mostrar filtros da troca de cor",
            "Ocultar filtros da troca de cor",
            self._color_filters_visible,
            {"fill": "x", "pady": (8, 2)},
        )

    def _toggle_overflow_filters(self) -> None:
        self._overflow_filters_visible = self._toggle_section(
            self.frame_filtros_overflow,
            self.btn_toggle_overflow_filters,
            "Mostrar filtros do Memory Overflow",
            "Ocultar filtros do Memory Overflow",
            self._overflow_filters_visible,
            {"fill": "x", "pady": (2, 8)},
        )

    def _toggle_video_filters(self) -> None:
        self._video_filters_visible = self._toggle_section(
            self.frame_video_filters,
            self.btn_toggle_video_filters,
            "Mostrar filtros extras do vídeo",
            "Ocultar filtros extras do vídeo",
            self._video_filters_visible,
            {"fill": "x", "pady": (2, 10)},
        )

    def _ler_parametros_video(self) -> dict:
        """Lê e valida parâmetros do vídeo."""
        try:
            params = {
                "preprocess": self.var_video_preprocess.get(),
                "equalize": self.var_video_equalize.get(),
                "clahe": self.var_video_clahe.get(),
                "stat_local": self.var_video_stat_local.get(),
                "blur": self.var_video_blur.get(),
                "adaptive_method": self.var_video_adaptive_method.get(),
                "adaptive_polarity": self.var_video_adaptive_polarity.get(),
                "multi_mode": self.var_video_multi_mode.get(),
                "stat_polarity": self.var_video_stat_polarity.get(),
                "range_invert": self.var_video_range_invert.get(),
                "block_size": int(self.video_entries["block_size"].get()),
                "offset": float(self.video_entries["offset"].get()),
                "classes": int(self.video_entries["classes"].get()),
                "multi_target_class": int(self.video_entries["multi_target_class"].get()),
                "range_low": int(self.video_entries["range_low"].get()),
                "range_high": int(self.video_entries["range_high"].get()),
                "k": float(self.video_entries["k"].get()),
                "stat_window": int(self.video_entries["stat_window"].get()),
                "blur_kernel": int(self.video_entries["blur_kernel"].get()),
                "clahe_clip": float(self.video_entries["clahe_clip"].get()),
                "clahe_tile": int(self.video_entries["clahe_tile"].get()),
            }
        except ValueError as exc:
            raise ValueError("Verifique os parametros numericos.") from exc

        # Validações
        if params["block_size"] < 3:
            params["block_size"] = 3
        if params["block_size"] % 2 == 0:
            params["block_size"] += 1

        if params["stat_window"] < 3:
            params["stat_window"] = 3
        if params["stat_window"] % 2 == 0:
            params["stat_window"] += 1

        if params["blur"] == "none":
            params["blur_kernel"] = 0
        elif params["blur_kernel"] % 2 == 0:
            params["blur_kernel"] += 1

        return params

    def _build_color_tab(self) -> None:
        topo = ttk.Frame(self.cor_content, padding=10)
        topo.pack(fill="x")

        ttk.Button(topo, text="Selecionar imagem", command=self.selecionar_imagem_cor).pack(side="left")
        ttk.Button(topo, text="Selecionar output folder", command=self.selecionar_pasta_saida_cor).pack(side="left", padx=8)
        ttk.Button(topo, text="Salvar imagem atual", command=self.salvar_preview_cor).pack(side="left", padx=8)
        ttk.Button(topo, text="Aplicar troca de cor", command=self.aplicar_troca_cor_principal).pack(side="left", padx=8)
        ttk.Button(topo, text="Aplicar Memory Overflow", command=self.aplicar_memory_overflow).pack(side="left", padx=8)

        self.var_cor_output_dir = tk.StringVar(value="../resultados/img_colorida")
        self.var_cor_origem = tk.StringVar(value="auto")
        self.var_cor_destino = tk.StringVar(value="verde")
        self.var_cor_tolerancia = tk.IntVar(value=18)
        self.var_cor_suavizacao = tk.IntVar(value=12)
        self.var_overflow_intensidade = tk.IntVar(value=8)
        self.var_use_hue_sliders = tk.BooleanVar(value=False)
        self.var_use_rgb_picker = tk.BooleanVar(value=False)
        self.var_hue_origem = tk.IntVar(value=0)
        self.var_hue_destino = tk.IntVar(value=60)
        self.rgb_destino = (0, 255, 0)
        self.hue_destino_rgb = 60

        # Filtros para troca de cor e memory overflow
        self.filtros_cor_vars = {
            "blur": tk.BooleanVar(value=False),
            "rotacao": tk.BooleanVar(value=False),
            "onda": tk.BooleanVar(value=False),
            "brilho_contraste": tk.BooleanVar(value=False),
            "saturacao": tk.BooleanVar(value=True),
            "negativo": tk.BooleanVar(value=False),
            "polarizador": tk.BooleanVar(value=False),
            "sepia": tk.BooleanVar(value=False),
            "nitidez": tk.BooleanVar(value=False),
            "posterizar": tk.BooleanVar(value=False),
        }
        self.filtros_cor_intensidades = {
            "blur": tk.IntVar(value=250),
            "rotacao": tk.IntVar(value=250),
            "onda": tk.IntVar(value=250),
            "brilho_contraste": tk.IntVar(value=250),
            "saturacao": tk.IntVar(value=250),
            "negativo": tk.IntVar(value=220),
            "polarizador": tk.IntVar(value=220),
            "sepia": tk.IntVar(value=220),
            "nitidez": tk.IntVar(value=220),
            "posterizar": tk.IntVar(value=220),
        }
        self.filtros_overflow_vars = {
            "blur": tk.BooleanVar(value=True),
            "rotacao": tk.BooleanVar(value=True),
            "onda": tk.BooleanVar(value=True),
            "brilho_contraste": tk.BooleanVar(value=True),
            "saturacao": tk.BooleanVar(value=True),
            "negativo": tk.BooleanVar(value=True),
            "polarizador": tk.BooleanVar(value=True),
            "sepia": tk.BooleanVar(value=True),
            "nitidez": tk.BooleanVar(value=True),
            "posterizar": tk.BooleanVar(value=True),
        }
        self.filtros_overflow_intensidades = {
            "blur": tk.IntVar(value=500),
            "rotacao": tk.IntVar(value=500),
            "onda": tk.IntVar(value=500),
            "brilho_contraste": tk.IntVar(value=500),
            "saturacao": tk.IntVar(value=500),
            "negativo": tk.IntVar(value=500),
            "polarizador": tk.IntVar(value=500),
            "sepia": tk.IntVar(value=500),
            "nitidez": tk.IntVar(value=500),
            "posterizar": tk.IntVar(value=500),
        }

        params = ttk.LabelFrame(self.cor_content, text="Parametros da troca de cor principal", padding=10)
        params.pack(fill="x", padx=10, pady=(0, 10))

        linha1 = ttk.Frame(params)
        linha1.pack(fill="x", pady=2)
        ttk.Label(linha1, text="Cor origem:").pack(side="left")
        ttk.Combobox(
            linha1,
            textvariable=self.var_cor_origem,
            values=["auto"] + list(COR_PARA_HUE.keys()),
            state="readonly",
            width=12,
        ).pack(side="left", padx=(5, 20))

        ttk.Label(linha1, text="Cor destino:").pack(side="left")
        ttk.Combobox(
            linha1,
            textvariable=self.var_cor_destino,
            values=list(COR_PARA_HUE.keys()),
            state="readonly",
            width=12,
        ).pack(side="left", padx=(5, 20))

        ttk.Label(linha1, text="Output:").pack(side="left")
        ttk.Entry(linha1, textvariable=self.var_cor_output_dir, width=35).pack(side="left", padx=5)

        linha_opt = ttk.Frame(params)
        linha_opt.pack(fill="x", pady=2)
        ttk.Checkbutton(
            linha_opt,
            text="Usar sliders de Hue (opcional)",
            variable=self.var_use_hue_sliders,
        ).pack(side="left")
        ttk.Checkbutton(
            linha_opt,
            text="Usar seletor RGB para cor destino",
            variable=self.var_use_rgb_picker,
        ).pack(side="left", padx=10)
        ttk.Button(linha_opt, text="Escolher cor final (RGB)", command=self.selecionar_cor_rgb_destino).pack(side="left", padx=10)
        self.lbl_rgb_destino = tk.Label(linha_opt, text="   ", bg="#00ff00", relief="groove")
        self.lbl_rgb_destino.pack(side="left")

        linha2 = ttk.Frame(params)
        linha2.pack(fill="x", pady=2)
        ttk.Label(linha2, text="Tolerancia de hue:").pack(side="left")
        ttk.Scale(linha2, from_=0, to=60, variable=self.var_cor_tolerancia, orient="horizontal", length=220).pack(side="left", padx=6)
        ttk.Label(linha2, textvariable=self.var_cor_tolerancia, width=4).pack(side="left")

        ttk.Label(linha2, text="Suavizacao:").pack(side="left", padx=(20, 0))
        ttk.Scale(linha2, from_=0, to=60, variable=self.var_cor_suavizacao, orient="horizontal", length=220).pack(side="left", padx=6)
        ttk.Label(linha2, textvariable=self.var_cor_suavizacao, width=4).pack(side="left")

        linha_hue = ttk.Frame(params)
        linha_hue.pack(fill="x", pady=2)
        ttk.Label(linha_hue, text="Hue origem (slider):").pack(side="left")
        ttk.Scale(linha_hue, from_=0, to=179, variable=self.var_hue_origem, orient="horizontal", length=220).pack(side="left", padx=6)
        ttk.Label(linha_hue, textvariable=self.var_hue_origem, width=4).pack(side="left")

        ttk.Label(linha_hue, text="Hue destino (slider):").pack(side="left", padx=(20, 0))
        ttk.Scale(linha_hue, from_=0, to=179, variable=self.var_hue_destino, orient="horizontal", length=220).pack(side="left", padx=6)
        ttk.Label(linha_hue, textvariable=self.var_hue_destino, width=4).pack(side="left")

        linha3 = ttk.Frame(params)
        linha3.pack(fill="x", pady=2)
        ttk.Label(linha3, text="Intensidade Memory Overflow:").pack(side="left")
        ttk.Scale(
            linha3,
            from_=1,
            to=20,
            variable=self.var_overflow_intensidade,
            orient="horizontal",
            length=220,
        ).pack(side="left", padx=6)
        ttk.Label(linha3, textvariable=self.var_overflow_intensidade, width=4).pack(side="left")

        self.btn_toggle_color_filters = ttk.Button(
            params,
            text="Mostrar filtros da troca de cor",
            command=self._toggle_color_filters,
        )
        self.btn_toggle_color_filters.pack(fill="x", pady=(8, 4))

        # Filtros para troca de cor
        self.frame_filtros_cor = ttk.LabelFrame(params, text="Filtros na troca de cor", padding=8)
        for nome_filtro, var in self.filtros_cor_vars.items():
            linha = ttk.Frame(self.frame_filtros_cor)
            linha.pack(fill="x", pady=2)
            ttk.Checkbutton(linha, text=nome_filtro.replace("_", " ").title(), variable=var).pack(side="left")
            ttk.Label(linha, text="0-1000%").pack(side="left", padx=(10, 4))
            ttk.Scale(
                linha,
                from_=0,
                to=1000,
                variable=self.filtros_cor_intensidades[nome_filtro],
                orient="horizontal",
                length=240,
            ).pack(side="left", padx=6)
            ttk.Label(linha, textvariable=self.filtros_cor_intensidades[nome_filtro], width=5).pack(side="left")

        self.btn_toggle_overflow_filters = ttk.Button(
            params,
            text="Mostrar filtros do Memory Overflow",
            command=self._toggle_overflow_filters,
        )
        self.btn_toggle_overflow_filters.pack(fill="x", pady=(8, 4))

        # Filtros para memory overflow
        self.frame_filtros_overflow = ttk.LabelFrame(params, text="Filtros no Memory Overflow", padding=8)
        for nome_filtro, var in self.filtros_overflow_vars.items():
            linha = ttk.Frame(self.frame_filtros_overflow)
            linha.pack(fill="x", pady=2)
            ttk.Checkbutton(linha, text=nome_filtro.replace("_", " ").title(), variable=var).pack(side="left")
            ttk.Label(linha, text="0-1000%").pack(side="left", padx=(10, 4))
            ttk.Scale(
                linha,
                from_=0,
                to=1000,
                variable=self.filtros_overflow_intensidades[nome_filtro],
                orient="horizontal",
                length=240,
            ).pack(side="left", padx=6)
            ttk.Label(linha, textvariable=self.filtros_overflow_intensidades[nome_filtro], width=5).pack(side="left")

        prev = ttk.LabelFrame(self.cor_content, text="Preview troca de cor", padding=10)
        prev.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.lbl_cor_original = ttk.Label(prev, text="Original", anchor="center")
        self.lbl_cor_original.grid(row=0, column=0, padx=10, pady=4)
        self.lbl_cor_resultado = ttk.Label(prev, text="Resultado", anchor="center")
        self.lbl_cor_resultado.grid(row=0, column=1, padx=10, pady=4)

        self.lbl_cor_img_original = ttk.Label(prev)
        self.lbl_cor_img_original.grid(row=1, column=0, padx=10, pady=4)
        self.lbl_cor_img_resultado = ttk.Label(prev)
        self.lbl_cor_img_resultado.grid(row=1, column=1, padx=10, pady=4)
        prev.columnconfigure(0, weight=1)
        prev.columnconfigure(1, weight=1)

        self.status_cor = tk.StringVar(value="Selecione uma imagem para troca de cor principal.")
        ttk.Label(self.cor_content, textvariable=self.status_cor, relief="sunken", anchor="w").pack(fill="x", side="bottom", ipady=4)

        self._color_filters_visible = False
        self._overflow_filters_visible = False

        self.var_cor_origem.trace_add("write", lambda *_: self._on_cor_param_change())
        self.var_cor_destino.trace_add("write", lambda *_: self._on_cor_param_change())
        self.var_cor_tolerancia.trace_add("write", lambda *_: self._on_cor_param_change())
        self.var_cor_suavizacao.trace_add("write", lambda *_: self._on_cor_param_change())
        self.var_overflow_intensidade.trace_add("write", lambda *_: self._on_cor_param_change())
        self.var_use_hue_sliders.trace_add("write", lambda *_: self._on_cor_param_change())
        self.var_use_rgb_picker.trace_add("write", lambda *_: self._on_cor_param_change())
        self.var_hue_origem.trace_add("write", lambda *_: self._on_cor_param_change())
        self.var_hue_destino.trace_add("write", lambda *_: self._on_cor_param_change())

        for var in (
            list(self.filtros_cor_vars.values())
            + list(self.filtros_cor_intensidades.values())
            + list(self.filtros_overflow_vars.values())
            + list(self.filtros_overflow_intensidades.values())
        ):
            var.trace_add("write", lambda *_: self._schedule_preview_cor_refresh())

    def _on_cor_param_change(self) -> None:
        if self.caminho_cor:
            self._schedule_preview_cor_refresh()

    def selecionar_imagem_cor(self) -> None:
        caminho = filedialog.askopenfilename(
            title="Selecione imagem para troca de cor",
            filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("Todos os arquivos", "*.*")],
        )
        if not caminho:
            return
        self.caminho_cor = caminho
        self.status_cor.set(f"Imagem selecionada: {os.path.basename(caminho)}")
        self._atualizar_preview_cor()

    def selecionar_pasta_saida_cor(self) -> None:
        pasta = filedialog.askdirectory(title="Selecione pasta de saida para troca de cor")
        if not pasta:
            return
        self.var_cor_output_dir.set(pasta)
        self.status_cor.set(f"Pasta de saida (troca de cor): {pasta}")

    def selecionar_cor_rgb_destino(self) -> None:
        rgb, _hex = colorchooser.askcolor(color="#%02x%02x%02x" % self.rgb_destino, title="Escolher cor final")
        if rgb is None:
            return
        r, g, b = [int(np.clip(v, 0, 255)) for v in rgb]
        self.rgb_destino = (r, g, b)
        hsv = cv2.cvtColor(np.array([[[b, g, r]]], dtype=np.uint8), cv2.COLOR_BGR2HSV)
        self.hue_destino_rgb = int(hsv[0, 0, 0])
        self.lbl_rgb_destino.configure(bg="#%02x%02x%02x" % (r, g, b))
        self.status_cor.set(f"Cor final RGB selecionada: ({r}, {g}, {b}) -> Hue {self.hue_destino_rgb}")
        self._on_cor_param_change()

    def _aplicar_filtro_cor_em_bgr(self, bgr: np.ndarray) -> tuple[np.ndarray, dict]:
        tol = int(self.var_cor_tolerancia.get())
        fea = int(self.var_cor_suavizacao.get())

        if self.var_use_hue_sliders.get():
            hue_origem = int(np.clip(self.var_hue_origem.get(), 0, 179))
        else:
            origem_nome = self.var_cor_origem.get()
            if origem_nome == "auto":
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
                hue_origem = int(detectar_hue_principal(hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]))
            else:
                hue_origem = int(COR_PARA_HUE[origem_nome])

        if self.var_use_rgb_picker.get():
            hue_destino = int(self.hue_destino_rgb)
        elif self.var_use_hue_sliders.get():
            hue_destino = int(np.clip(self.var_hue_destino.get(), 0, 179))
        else:
            hue_destino = int(COR_PARA_HUE[self.var_cor_destino.get()])

        saida_bgr, meta = shift_color(
            image_bgr=bgr,
            source_hue=hue_origem,
            target_hue=hue_destino,
            tolerance=tol,
            feather=fea,
        )
        filtros_selecionados = {nome for nome, var in self.filtros_cor_vars.items() if var.get()}
        if filtros_selecionados:
            intensidades = {nome: int(self.filtros_cor_intensidades[nome].get()) for nome in filtros_selecionados}
            saida_bgr = aplicar_filtros_selecionados(
                saida_bgr,
                filtros=filtros_selecionados,
                intensidades=intensidades,
                seed=0,
            )
        meta["modo_hue"] = "slider" if self.var_use_hue_sliders.get() else "combobox/auto"
        meta["modo_destino"] = "rgb_picker" if self.var_use_rgb_picker.get() else ("slider" if self.var_use_hue_sliders.get() else "combobox")
        return saida_bgr, meta

    def _salvar_preview_atual(self, saida_bgr: np.ndarray, sufixo: str) -> str:
        dir_saida = self.var_cor_output_dir.get().strip() or "../resultados/img_colorida"
        return salvar_troca_cor(
            self.caminho_cor,
            saida_bgr,
            dir_saida,
            sufixo=sufixo,
        )

    def salvar_preview_cor(self) -> None:
        if not self.caminho_cor:
            messagebox.showwarning("Aviso", "Selecione uma imagem na aba Troca de Cor.")
            return

        self._refresh_preview_cor_now()

        if self._ultima_preview_cor_bgr is None:
            messagebox.showerror("Erro", "Nao foi possivel salvar a imagem atual.")
            return

        if self.modo_preview_cor == "overflow":
            filtros_selecionados = {nome for nome, var in self.filtros_overflow_vars.items() if var.get()}
            filtros_str = "_".join(sorted(filtros_selecionados)) if filtros_selecionados else "default"
            caminho_saida = self._salvar_preview_atual(
                self._ultima_preview_cor_bgr,
                sufixo=f"preview_memory_overflow_i{int(self.var_overflow_intensidade.get())}_f{filtros_str}",
            )
        else:
            filtros_selecionados = {nome for nome, var in self.filtros_cor_vars.items() if var.get()}
            filtros_str = "_".join(sorted(filtros_selecionados)) if filtros_selecionados else "sem_filtros"
            caminho_saida = self._salvar_preview_atual(
                self._ultima_preview_cor_bgr,
                sufixo=f"preview_troca_cor_{filtros_str}",
            )

        self.status_cor.set(f"Imagem salva em: {caminho_saida}")

    def _atualizar_preview_cor(self) -> None:
        if not self.caminho_cor:
            return
        bgr = cv2.imread(self.caminho_cor, cv2.IMREAD_COLOR)
        if bgr is None:
            self.status_cor.set("Falha ao abrir imagem para preview.")
            return

        try:
            if self.modo_preview_cor == "overflow":
                filtros_selecionados = {nome for nome, var in self.filtros_overflow_vars.items() if var.get()}
                intensidades = {nome: int(self.filtros_overflow_intensidades[nome].get()) for nome in filtros_selecionados}
                saida_bgr = memory_overflow_glitch(
                    image_bgr=bgr,
                    intensity=int(self.var_overflow_intensidade.get()),
                    filtros=filtros_selecionados if filtros_selecionados else None,
                    intensidades=intensidades if intensidades else None,
                )
            else:
                saida_bgr, _ = self._aplicar_filtro_cor_em_bgr(bgr)
        except Exception as exc:
            self.status_cor.set(f"Erro no preview de troca de cor: {exc}")
            return

        orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res_rgb = cv2.cvtColor(saida_bgr, cv2.COLOR_BGR2RGB)
        self._ultima_preview_cor_bgr = saida_bgr.copy()
        t1 = self._imagem_para_thumbnail(orig_rgb, largura=360, altura=260)
        t2 = self._imagem_para_thumbnail(res_rgb, largura=360, altura=260)
        self.preview_color_images = [t1, t2]
        self.lbl_cor_img_original.configure(image=t1)
        self.lbl_cor_img_resultado.configure(image=t2)

    def aplicar_troca_cor_principal(self) -> None:
        if not self.caminho_cor:
            messagebox.showwarning("Aviso", "Selecione uma imagem na aba Troca de Cor.")
            return

        bgr = cv2.imread(self.caminho_cor, cv2.IMREAD_COLOR)
        if bgr is None:
            messagebox.showerror("Erro", "Falha ao abrir imagem para troca de cor.")
            return

        try:
            self.modo_preview_cor = "troca"
            saida_bgr, meta = self._aplicar_filtro_cor_em_bgr(bgr)
            dir_saida = self.var_cor_output_dir.get().strip() or "../resultados/img_colorida"
            
            # Coleta filtros selecionados
            filtros_selecionados = {nome for nome, var in self.filtros_cor_vars.items() if var.get()}
            filtros_str = "_".join(sorted(filtros_selecionados)) if filtros_selecionados else "sem_filtros"
            caminho_saida = salvar_troca_cor(
                self.caminho_cor,
                saida_bgr,
                dir_saida,
                sufixo=f"troca_cor_{filtros_str}",
            )
            
            self.status_cor.set(
                f"Troca aplicada. Origem hue={meta['source_hue']:.1f}, destino hue={meta['target_hue']:.1f}. Salvo em: {caminho_saida}"
            )
            self._refresh_preview_cor_now()
        except Exception as exc:
            messagebox.showerror("Erro", str(exc))
            self.status_cor.set("Erro ao aplicar troca de cor principal.")

    def aplicar_memory_overflow(self) -> None:
        if not self.caminho_cor:
            messagebox.showwarning("Aviso", "Selecione uma imagem na aba Troca de Cor.")
            return

        bgr = cv2.imread(self.caminho_cor, cv2.IMREAD_COLOR)
        if bgr is None:
            messagebox.showerror("Erro", "Falha ao abrir imagem para efeito Memory Overflow.")
            return

        try:
            self.modo_preview_cor = "overflow"
            
            # Coleta filtros selecionados para memory overflow
            filtros_selecionados = {nome for nome, var in self.filtros_overflow_vars.items() if var.get()}
            intensidades = {nome: int(self.filtros_overflow_intensidades[nome].get()) for nome in filtros_selecionados}
            
            saida_bgr = memory_overflow_glitch(
                image_bgr=bgr,
                intensity=int(self.var_overflow_intensidade.get()),
                filtros=filtros_selecionados if filtros_selecionados else None,
                intensidades=intensidades if intensidades else None,
            )
            dir_saida = self.var_cor_output_dir.get().strip() or "../resultados/img_colorida"
            
            # Nome dos filtros aplicados
            filtros_str = "_".join(sorted(filtros_selecionados)) if filtros_selecionados else "default"
            
            caminho_saida = salvar_troca_cor(
                self.caminho_cor,
                saida_bgr,
                dir_saida,
                sufixo=f"memory_overflow_i{int(self.var_overflow_intensidade.get())}_f{filtros_str}",
            )
            self.status_cor.set(
                f"Memory Overflow aplicado (intensidade={int(self.var_overflow_intensidade.get())}, filtros={filtros_str}). Salvo em: {caminho_saida}"
            )
            self._refresh_preview_cor_now()
        except Exception as exc:
            messagebox.showerror("Erro", str(exc))
            self.status_cor.set("Erro ao aplicar Memory Overflow.")

    def _add_entry(self, parent: ttk.Frame, key: str, label: str, default: str) -> None:
        ttk.Label(parent, text=label + ":").pack(side="left", padx=(0, 4))
        var = tk.StringVar(value=default)
        self.entries[key] = var
        ttk.Entry(parent, textvariable=var, width=8).pack(side="left", padx=(0, 12))

    def selecionar_imagens(self) -> None:
        caminhos = filedialog.askopenfilenames(
            title="Selecione uma ou mais imagens",
            filetypes=[
                ("Imagens", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if not caminhos:
            return

        novos = [c for c in caminhos if c not in self.imagens]
        self.imagens.extend(novos)
        for caminho in novos:
            self.lista.insert(tk.END, caminho)

        self.status.set(f"{len(novos)} imagem(ns) adicionada(s). Total: {len(self.imagens)}")
        if novos and not self.lista.curselection():
            self.lista.selection_set(0)
            self._atualizar_preview_selecionada()

    def remover_imagem(self) -> None:
        idx = self.lista.curselection()
        if not idx:
            return
        i = idx[0]
        self.lista.delete(i)
        self.imagens.pop(i)
        self.status.set(f"Imagem removida. Total: {len(self.imagens)}")
        if self.imagens:
            idx_novo = min(i, len(self.imagens) - 1)
            self.lista.selection_set(idx_novo)
            self._atualizar_preview_selecionada()
        else:
            self._limpar_preview()

    def limpar_lista(self) -> None:
        self.imagens.clear()
        self.lista.delete(0, tk.END)
        self._limpar_preview()
        self.status.set("Lista limpa.")

    def _on_selecao_lista(self, _event: tk.Event) -> None:
        self._atualizar_preview_selecionada()

    def _toggle_filtros_extras(self) -> None:
        visivel = self.var_filtros_visiveis.get()
        if visivel:
            self.frame_filtros_extras.pack_forget()
            self.btn_toggle_filtros.configure(text="Mostrar filtros extras")
            self.var_filtros_visiveis.set(False)
        else:
            self.frame_filtros_extras.pack(fill="x", pady=(6, 2))
            self.btn_toggle_filtros.configure(text="Ocultar filtros extras")
            self.var_filtros_visiveis.set(True)

    def _selecionar_todos_metodos(self) -> None:
        for var in self.method_vars.values():
            var.set(True)

    def _limpar_selecao_metodos(self) -> None:
        for var in self.method_vars.values():
            var.set(False)

    def _metodos_selecionados(self) -> set[str]:
        return {nome for nome, var in self.method_vars.items() if var.get()}

    def _ler_parametros(self) -> dict:
        try:
            params = {
                "preprocess": self.var_preprocess.get(),
                "equalize": self.var_equalize.get(),
                "clahe": self.var_clahe.get(),
                "stat_local": self.var_stat_local.get(),
                "save_figure": self.var_save_figure.get(),
                "save_individual": self.var_save_individual.get(),
                "show_comparison": self.var_show_comparison.get(),
                "show_analysis": self.var_show_analysis.get(),
                "save_analysis": self.var_save_analysis.get(),
                "save_extra_methods": self.var_save_extra_methods.get(),
                "save_histograms": self.var_save_histograms.get(),
                "selected_methods": self._metodos_selecionados(),
                "blur": self.var_blur.get(),
                "adaptive_method": self.var_adaptive_method.get(),
                "adaptive_polarity": self.var_adaptive_polarity.get(),
                "multi_mode": self.var_multi_mode.get(),
                "stat_polarity": self.var_stat_polarity.get(),
                "range_invert": self.var_range_invert.get(),
                "figure_dir": self.var_figure_dir.get().strip() or "resultados",
                "block_size": int(self.entries["block_size"].get()),
                "offset": float(self.entries["offset"].get()),
                "classes": int(self.entries["classes"].get()),
                "multi_target_class": int(self.entries["multi_target_class"].get()),
                "range_low": int(self.entries["range_low"].get()),
                "range_high": int(self.entries["range_high"].get()),
                "k": float(self.entries["k"].get()),
                "stat_window": int(self.entries["stat_window"].get()),
                "blur_kernel": int(self.entries["blur_kernel"].get()),
                "clahe_clip": float(self.entries["clahe_clip"].get()),
                "clahe_tile": int(self.entries["clahe_tile"].get()),
                "figure_width": int(self.entries["figure_width"].get()),
                "figure_height": int(self.entries["figure_height"].get()),
                "figure_dpi": int(self.entries["figure_dpi"].get()),
            }
        except ValueError as exc:
            raise ValueError("Verifique os parametros numericos.") from exc

        if params["block_size"] < 3:
            params["block_size"] = 3
        if params["block_size"] % 2 == 0:
            params["block_size"] += 1

        if params["stat_window"] < 3:
            params["stat_window"] = 3
        if params["stat_window"] % 2 == 0:
            params["stat_window"] += 1

        if params["blur"] == "none":
            params["blur"] = ""

        params["blur_kernel"] = max(3, params["blur_kernel"])
        if params["blur_kernel"] % 2 == 0:
            params["blur_kernel"] += 1

        params["clahe_tile"] = max(2, params["clahe_tile"])

        if not params["selected_methods"]:
            raise ValueError("Selecione ao menos um threshold para aplicar.")

        return params

    def selecionar_pasta(self) -> None:
        pasta = filedialog.askdirectory(title="Selecione pasta com imagens")
        if not pasta:
            return
        extensoes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        caminhos = []
        for nome in sorted(os.listdir(pasta)):
            caminho = os.path.join(pasta, nome)
            if os.path.isfile(caminho) and os.path.splitext(nome.lower())[1] in extensoes:
                caminhos.append(caminho)

        if not caminhos:
            messagebox.showinfo("Sem imagens", "Nenhuma imagem encontrada na pasta selecionada.")
            return

        novos = [c for c in caminhos if c not in self.imagens]
        self.imagens.extend(novos)
        for caminho in novos:
            self.lista.insert(tk.END, caminho)

        self.status.set(f"{len(novos)} imagem(ns) adicionada(s) da pasta. Total: {len(self.imagens)}")
        if novos and not self.lista.curselection():
            self.lista.selection_set(0)
            self._atualizar_preview_selecionada()

    def selecionar_pasta_saida(self) -> None:
        pasta = filedialog.askdirectory(title="Selecione pasta de saida")
        if not pasta:
            return
        self.var_figure_dir.set(pasta)
        self.status.set(f"Pasta de saida definida: {pasta}")

    def _processar_arquivo(self, caminho_imagem: str, params: dict) -> None:
        rgb, gray_proc, adaptativo, multi, faixa, estatistico, meta_adaptativo, meta_multi, meta_faixa, meta_estatistico = self._gerar_resultados(caminho_imagem, params)
        extras = threshold_metodos_globais(gray_proc, params["selected_methods"])
        dir_img = os.path.join(params["figure_dir"], "img")
        dir_hist = os.path.join(params["figure_dir"], "hist")
        dir_comp = os.path.join(params["figure_dir"], "comp")

        resultados_todos = {
            "adaptativo": (adaptativo, meta_adaptativo),
            "multi_otsu": (multi, meta_multi),
            "range": (faixa, meta_faixa),
            "estatistico": (estatistico, meta_estatistico),
        }
        resultados_todos.update(extras)
        resultados_selecionados = {
            nome: valor
            for nome, valor in resultados_todos.items()
            if nome in params["selected_methods"]
        }

        if params["save_individual"]:
            salvar_resultados_por_metodo(
                titulo_base=caminho_imagem,
                resultados=resultados_selecionados,
                pasta_saida=dir_img,
                largura_px=params["figure_width"],
                altura_px=params["figure_height"],
            )

        if params["show_comparison"]:
            mostrar_comparativo_selecionados(
                original_rgb=rgb,
                resultados=resultados_selecionados,
                titulo_base=caminho_imagem,
                salvar_figura=params["save_figure"],
                exibir=True,
                pasta_saida=dir_comp,
                dpi_saida=params["figure_dpi"],
                largura_px=params["figure_width"],
                altura_px=params["figure_height"],
            )

        if params["show_analysis"] or params["save_analysis"]:
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
                titulo_base=caminho_imagem,
                salvar_figura=params["save_analysis"],
                exibir=params["show_analysis"],
                pasta_saida=dir_hist,
                dpi_saida=params["figure_dpi"],
                largura_px=params["figure_width"],
                altura_px=params["figure_height"],
            )

        if params["save_histograms"]:
            mostrar_histogramas_filtros(
                titulo_base=caminho_imagem,
                gray_base=gray_proc,
                resultados=resultados_selecionados,
                pasta_saida=dir_hist,
                salvar_figura=True,
                exibir=True,
            )
            salvar_histogramas_individuais(
                titulo_base=caminho_imagem,
                gray_base=gray_proc,
                resultados=resultados_selecionados,
                pasta_saida=dir_hist,
            )

    def _gerar_resultados(
        self,
        caminho_imagem: str,
        params: dict,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        object,
        object,
        object,
        object,
    ]:
        bgr = cv2.imread(caminho_imagem, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Falha ao ler imagem: {caminho_imagem}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        gray_proc = preprocessar(
            gray,
            usar_preprocessamento=params["preprocess"],
            blur=params["blur"],
            blur_kernel=params["blur_kernel"],
            equalizar_histograma=params["equalize"],
            usar_clahe=params["clahe"],
            clahe_clip=params["clahe_clip"],
            clahe_tile=params["clahe_tile"],
        )

        adaptativo, meta_adaptativo = threshold_adaptativo_local(
            gray_proc,
            block_size=params["block_size"],
            offset=params["offset"],
            metodo=params["adaptive_method"],
            polaridade=params["adaptive_polarity"],
        )
        multi, meta_multi = threshold_multi_otsu(
            gray_proc,
            classes=params["classes"],
            modo_saida=params["multi_mode"],
            classe_alvo=params["multi_target_class"],
        )
        faixa, meta_faixa = threshold_range(
            gray_proc,
            baixo=params["range_low"],
            alto=params["range_high"],
            inverter=params["range_invert"],
        )
        estatistico, meta_estatistico = threshold_estatistico(
            gray_proc,
            k=params["k"],
            local=params["stat_local"],
            janela_local=params["stat_window"],
            polaridade=params["stat_polarity"],
        )

        return (
            rgb,
            gray_proc,
            adaptativo,
            multi,
            faixa,
            estatistico,
            meta_adaptativo,
            meta_multi,
            meta_faixa,
            meta_estatistico,
        )

    def _imagem_para_thumbnail(self, img: np.ndarray, largura: int = 150, altura: int = 95) -> ImageTk.PhotoImage:
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = img
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((largura, altura), RESAMPLING)
        return ImageTk.PhotoImage(pil_img)

    def _limpar_preview(self) -> None:
        self.preview_images = []
        for lbl in self.preview_labels:
            lbl.configure(image="")

    def _atualizar_preview_selecionada(self) -> None:
        idx = self.lista.curselection()
        if not idx:
            return

        try:
            params = self._ler_parametros()
            caminho = self.imagens[idx[0]]
            rgb, _, adaptativo, multi, faixa, estatistico, _, _, _, _ = self._gerar_resultados(caminho, params)
            thumbs = [
                self._imagem_para_thumbnail(rgb),
                self._imagem_para_thumbnail(adaptativo),
                self._imagem_para_thumbnail(multi),
                self._imagem_para_thumbnail(faixa),
                self._imagem_para_thumbnail(estatistico),
            ]
            self.preview_images = thumbs
            for lbl, thumb in zip(self.preview_labels, thumbs):
                lbl.configure(image=thumb)
            self.status.set(f"Preview atualizado: {os.path.basename(caminho)}")
        except Exception as exc:
            self._limpar_preview()
            self.status.set("Falha ao atualizar preview.")
            messagebox.showerror("Erro no preview", str(exc))

    def processar_selecionada(self) -> None:
        if not self.imagens:
            messagebox.showwarning("Aviso", "Selecione ao menos uma imagem.")
            return

        idx = self.lista.curselection()
        if not idx:
            messagebox.showwarning("Aviso", "Selecione uma imagem na lista.")
            return

        try:
            params = self._ler_parametros()
            caminho = self.imagens[idx[0]]
            self.status.set(f"Processando: {os.path.basename(caminho)}")
            self.progress.configure(maximum=1, value=0)
            self.root.update_idletasks()
            self._processar_arquivo(caminho, params)
            self.progress.configure(value=1)
            self.status.set("Processamento finalizado.")
        except Exception as exc:
            messagebox.showerror("Erro", str(exc))
            self.status.set("Erro durante processamento.")
        finally:
            self.root.update_idletasks()

    def processar_todas(self) -> None:
        if not self.imagens:
            messagebox.showwarning("Aviso", "Selecione ao menos uma imagem.")
            return

        try:
            params = self._ler_parametros()
            total = len(self.imagens)
            self.progress.configure(maximum=total, value=0)
            for i, caminho in enumerate(self.imagens, start=1):
                self.status.set(f"Processando {i}/{total}: {os.path.basename(caminho)}")
                self.root.update_idletasks()
                self._processar_arquivo(caminho, params)
                self.progress.configure(value=i)
                self.root.update_idletasks()
            self.status.set("Processamento de todas as imagens finalizado.")
        except Exception as exc:
            messagebox.showerror("Erro", str(exc))
            self.status.set("Erro durante processamento em lote.")


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    app = AppThresholdGUI(root)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
