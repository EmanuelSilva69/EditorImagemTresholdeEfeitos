from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk


class LauncherApp:
    def __init__(self, root: tk.Tk, threshold_runner, color_runner, module_three_runner=None) -> None:
        self.root = root
        self.threshold_runner = threshold_runner
        self.color_runner = color_runner
        self.module_three_runner = module_three_runner or self._module_three
        self.root.title("Launcher - PDI")
        self.root.geometry("820x520")
        self.root.minsize(720, 460)
        self.root.configure(bg="#101418")
        self.root.protocol("WM_DELETE_WINDOW", self._close)

        self._build_ui()

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure("Launcher.TFrame", background="#101418")
        style.configure("LauncherTitle.TLabel", background="#101418", foreground="#f4f7fb", font=("Segoe UI", 22, "bold"))
        style.configure("LauncherSub.TLabel", background="#101418", foreground="#c0cad6", font=("Segoe UI", 11))

        container = ttk.Frame(self.root, style="Launcher.TFrame", padding=24)
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container, style="Launcher.TFrame")
        header.pack(fill="x", pady=(0, 22))
        ttk.Label(header, text="Selecione o módulo", style="LauncherTitle.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Thresholding ATV1 e Processamento de Cores T2 foram separados em aplicativos independentes.",
            style="LauncherSub.TLabel",
            wraplength=680,
        ).pack(anchor="w", pady=(8, 0))

        cards = ttk.Frame(container, style="Launcher.TFrame")
        cards.pack(fill="both", expand=True)
        cards.columnconfigure(0, weight=1)
        cards.columnconfigure(1, weight=1)
        cards.columnconfigure(2, weight=1)

        self._create_card(
            cards,
            0,
            "Thresholding\nATV1",
            "Abre a interface original de limiarizacao, segmentacao e relatorios.",
            "#1f4b99",
            self._open_threshold,
        )
        self._create_card(
            cards,
            1,
            "Processamento de Cores\nT2",
            "Abre HSV, Histogram Matching, SSIM e processamento de imagem/video.",
            "#0f7b53",
            self._open_color,
        )
        self._create_card(
            cards,
            2,
            "Editor de Filtros\nCriativos",
            "Abre o editor com filtros, salvamento e seletor HSV de cor.",
            "#6e5a14",
            self.module_three_runner,
        )

    def _create_card(self, parent: ttk.Frame, column: int, title: str, description: str, color: str, command) -> None:
        frame = tk.Frame(parent, bg=color, bd=0, highlightthickness=0)
        frame.grid(row=0, column=column, sticky="nsew", padx=10)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        inner = tk.Frame(frame, bg=color, padx=18, pady=18)
        inner.pack(fill="both", expand=True)

        tk.Label(
            inner,
            text=title,
            bg=color,
            fg="#ffffff",
            font=("Segoe UI", 20, "bold"),
            justify="center",
            wraplength=220,
        ).pack(pady=(8, 14), fill="x")

        tk.Label(
            inner,
            text=description,
            bg=color,
            fg="#edf2f7",
            font=("Segoe UI", 10),
            justify="center",
            wraplength=220,
        ).pack(pady=(0, 18), fill="x")

        btn = tk.Button(
            inner,
            text="Abrir",
            command=command,
            bg="#f4f7fb",
            fg="#101418",
            activebackground="#dfe7f1",
            activeforeground="#101418",
            relief="flat",
            font=("Segoe UI", 12, "bold"),
            padx=18,
            pady=12,
        )
        btn.pack(fill="x", pady=(22, 0))

    def _open_threshold(self) -> None:
        self.root.destroy()
        self.threshold_runner()

    def _open_color(self) -> None:
        self.root.destroy()
        self.color_runner()

    def _module_three(self) -> None:
        messagebox.showinfo("Modulo 3", "Espaco reservado para um modulo futuro.")

    def _close(self) -> None:
        self.root.destroy()


def run_launcher_app(threshold_runner, color_runner, module_three_runner=None) -> None:
    root = tk.Tk()
    app = LauncherApp(root, threshold_runner, color_runner, module_three_runner)
    _ = app
    root.mainloop()
