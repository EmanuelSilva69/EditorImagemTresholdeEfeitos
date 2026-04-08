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


class AppThresholdGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("PDI - Thresholding Local (Tkinter)")
        self.root.geometry("1200x760")

        self.imagens: list[str] = []
        self.preview_images: list[ImageTk.PhotoImage] = []

        self._build_ui()

    def _build_ui(self) -> None:
        frame_top = ttk.Frame(self.root, padding=10)
        frame_top.pack(fill="x")

        ttk.Button(frame_top, text="Selecionar imagens", command=self.selecionar_imagens).pack(side="left")
        ttk.Button(frame_top, text="Selecionar pasta", command=self.selecionar_pasta).pack(side="left", padx=8)
        ttk.Button(frame_top, text="Remover selecionada", command=self.remover_imagem).pack(
            side="left", padx=8
        )
        ttk.Button(frame_top, text="Limpar lista", command=self.limpar_lista).pack(side="left")

        frame_lista = ttk.LabelFrame(self.root, text="Imagens selecionadas", padding=10)
        frame_lista.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.lista = tk.Listbox(frame_lista, height=8)
        self.lista.pack(fill="both", expand=True)
        self.lista.bind("<<ListboxSelect>>", self._on_selecao_lista)

        frame_preview = ttk.LabelFrame(self.root, text="Preview (imagem selecionada)", padding=10)
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

        frame_params = ttk.LabelFrame(self.root, text="Parametros", padding=10)
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

        frame_acoes = ttk.Frame(self.root, padding=10)
        frame_acoes.pack(fill="x")

        ttk.Button(frame_acoes, text="Processar selecionada", command=self.processar_selecionada).pack(side="left")
        ttk.Button(frame_acoes, text="Processar todas", command=self.processar_todas).pack(side="left", padx=8)

        self.progress = ttk.Progressbar(frame_acoes, orient="horizontal", mode="determinate", length=280)
        self.progress.pack(side="left", padx=10)

        self.status = tk.StringVar(value="Pronto.")
        ttk.Label(self.root, textvariable=self.status, relief="sunken", anchor="w").pack(
            fill="x", side="bottom", ipady=4
        )

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
