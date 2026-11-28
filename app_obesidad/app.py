"""CustomTkinter desktop GUI for the obesity-level classifier.

The GUI only collects inputs and renders results; encoding and prediction live in `core`.
UI copy is in Spanish because the end users are Spanish speakers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import customtkinter as ctk

from app_obesidad import core

WINDOW_TITLE = "Clasificador de Obesidad"

CLASS_LABELS_ES = {
    "Insufficient_Weight": "Peso insuficiente",
    "Normal_Weight": "Peso normal",
    "Overweight_Level_I": "Sobrepeso nivel I",
    "Overweight_Level_II": "Sobrepeso nivel II",
    "Obesity_Type_I": "Obesidad tipo I",
    "Obesity_Type_II": "Obesidad tipo II",
    "Obesity_Type_III": "Obesidad tipo III",
}

YES_NO = {"Sí": "yes", "No": "no"}
FREQUENCY_ES = {"No": "no", "A veces": "Sometimes", "Frecuentemente": "Frequently", "Siempre": "Always"}
GENDER_ES = {"Mujer": "Female", "Hombre": "Male"}
TRANSPORT_ES = {
    "Automóvil": "Automobile",
    "Bicicleta": "Bike",
    "Motocicleta": "Motorbike",
    "Transporte público": "Public_Transportation",
    "Caminando": "Walking",
}

DEFAULT_PROFILE: Dict[str, Any] = {
    "Gender": "Female",
    "Age": 23,
    "family_history_with_overweight": "yes",
    "FAVC": "yes",
    "FCVC": 2,
    "NCP": 3,
    "CAEC": "Sometimes",
    "SMOKE": "no",
    "CH2O": 2,
    "SCC": "no",
    "FAF": 1,
    "TUE": 1,
    "CALC": "Sometimes",
    "MTRANS": "Public_Transportation",
}

DISCLAIMER = (
    "Modelo académico con fines educativos. "
    "No constituye un diagnóstico médico ni reemplaza la consulta con un profesional de la salud."
)

HIGHLIGHT_COLOR = "#d9534f"
BAR_COLOR = "#3b8ed0"


@dataclass(frozen=True)
class SliderSpec:
    field: str
    label: str
    describe: Callable[[int], str]


SLIDERS: List[SliderSpec] = [
    SliderSpec("Age", "Edad (años)", lambda v: f"{v} años"),
    SliderSpec("FCVC", "Consumo de verduras",
               lambda v: {1: "Nunca", 2: "A veces", 3: "Siempre"}.get(v, str(v))),
    SliderSpec("NCP", "Comidas principales al día", lambda v: f"{v}"),
    SliderSpec("CH2O", "Agua al día",
               lambda v: {1: "Menos de 1 L", 2: "Entre 1 y 2 L", 3: "Más de 2 L"}.get(v, str(v))),
    SliderSpec("FAF", "Actividad física semanal",
               lambda v: {0: "Ninguna", 1: "1 a 2 días", 2: "2 a 4 días", 3: "4 a 5 días"}.get(v, str(v))),
    SliderSpec("TUE", "Uso diario de dispositivos",
               lambda v: {0: "0 a 2 h", 1: "3 a 5 h", 2: "Más de 5 h"}.get(v, str(v))),
]

SWITCHES = [
    ("family_history_with_overweight", "Antecedentes familiares de sobrepeso"),
    ("FAVC", "Consumo frecuente de alimentos calóricos"),
    ("SMOKE", "Fuma"),
    ("SCC", "Controla las calorías que consume"),
]

OPTION_MENUS = [
    ("Gender", "Género", GENDER_ES),
    ("CAEC", "Come entre comidas", FREQUENCY_ES),
    ("CALC", "Consumo de alcohol", FREQUENCY_ES),
    ("MTRANS", "Medio de transporte habitual", TRANSPORT_ES),
]


def _label_for(mapping: Dict[str, str], raw_value: str) -> str:
    return next(label for label, raw in mapping.items() if raw == raw_value)


class ObesityApp(ctk.CTk):
    """Main window: model sidebar, input panel and probability bars."""

    def __init__(self) -> None:
        super().__init__()
        self.title(WINDOW_TITLE)
        self.geometry("1150x720")
        self.minsize(980, 620)
        ctk.set_appearance_mode("system")

        self.metadata = core.load_metadata()
        self.ranges = self.metadata["rangos"]
        self.slider_vars: Dict[str, ctk.IntVar] = {}
        self.slider_value_labels: Dict[str, ctk.CTkLabel] = {}
        self.switch_vars: Dict[str, ctk.StringVar] = {}
        self.option_vars: Dict[str, ctk.StringVar] = {}
        self.auto_classify = ctk.BooleanVar(value=False)
        self.bars: Dict[str, Dict[str, Any]] = {}

        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_inputs()
        self._build_results()
        self.reset()

    # ----- layout -----
    def _build_sidebar(self) -> None:
        sidebar = ctk.CTkFrame(self, width=240, corner_radius=0)
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_rowconfigure(20, weight=1)

        ctk.CTkLabel(sidebar, text="Modelo", font=ctk.CTkFont(size=20, weight="bold")).grid(
            row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        info = [
            ("Algoritmo", self.metadata.get("algoritmo", "-")),
            ("Preprocesamiento", self.metadata.get("preprocesamiento", "-")),
            ("Benchmark", self.metadata.get("benchmark", "-")),
            ("Accuracy (prueba)", f"{self.metadata.get('accuracy', 0):.2%}"),
            ("F1 macro (prueba)", f"{self.metadata.get('f1_macro', 0):.4f}"),
            ("F1 macro (validación cruzada)", self.metadata.get("cv_f1_macro", "-")),
            ("Fecha de entrenamiento", self.metadata.get("fecha", "-")),
        ]
        for i, (title, value) in enumerate(info, start=1):
            ctk.CTkLabel(sidebar, text=title, font=ctk.CTkFont(size=12, weight="bold"),
                         anchor="w").grid(row=2 * i - 1, column=0, padx=20, pady=(8, 0), sticky="w")
            ctk.CTkLabel(sidebar, text=str(value), anchor="w", justify="left",
                         wraplength=200).grid(row=2 * i, column=0, padx=20, sticky="w")

        ctk.CTkLabel(sidebar, text=DISCLAIMER, wraplength=200, justify="left",
                     font=ctk.CTkFont(size=11, slant="italic"), text_color="gray").grid(
            row=21, column=0, padx=20, pady=20, sticky="sw")

    def _build_inputs(self) -> None:
        panel = ctk.CTkScrollableFrame(self, label_text="Medidas del perfil")
        panel.grid(row=0, column=1, padx=(20, 10), pady=20, sticky="nsew")
        panel.grid_columnconfigure(1, weight=1)
        row = 0

        for field, label, mapping in OPTION_MENUS:
            var = ctk.StringVar()
            self.option_vars[field] = var
            ctk.CTkLabel(panel, text=label, anchor="w").grid(row=row, column=0, padx=10, pady=6, sticky="w")
            ctk.CTkOptionMenu(panel, values=list(mapping), variable=var,
                              command=lambda _value: self._on_change()).grid(
                row=row, column=1, columnspan=2, padx=10, pady=6, sticky="ew")
            row += 1

        for spec in SLIDERS:
            low, high = (int(v) for v in self.ranges[spec.field])
            var = ctk.IntVar()
            self.slider_vars[spec.field] = var
            ctk.CTkLabel(panel, text=spec.label, anchor="w").grid(row=row, column=0, padx=10, pady=6, sticky="w")
            ctk.CTkSlider(panel, from_=low, to=high, number_of_steps=high - low, variable=var,
                          command=lambda _value, f=spec.field: self._on_slider(f)).grid(
                row=row, column=1, padx=10, pady=6, sticky="ew")
            value_label = ctk.CTkLabel(panel, text="", width=110, anchor="w")
            value_label.grid(row=row, column=2, padx=10, pady=6, sticky="w")
            self.slider_value_labels[spec.field] = value_label
            row += 1

        for field, label in SWITCHES:
            var = ctk.StringVar(value="no")
            self.switch_vars[field] = var
            ctk.CTkSwitch(panel, text=label, variable=var, onvalue="yes", offvalue="no",
                          command=self._on_change).grid(
                row=row, column=0, columnspan=3, padx=10, pady=6, sticky="w")
            row += 1

        buttons = ctk.CTkFrame(panel, fg_color="transparent")
        buttons.grid(row=row, column=0, columnspan=3, padx=10, pady=(16, 6), sticky="ew")
        ctk.CTkButton(buttons, text="Clasificar", command=self.classify).pack(side="left", padx=(0, 10))
        ctk.CTkButton(buttons, text="Restablecer", fg_color="gray", command=self.reset).pack(side="left")
        ctk.CTkCheckBox(panel, text="Clasificar automáticamente al cambiar un valor",
                        variable=self.auto_classify).grid(
            row=row + 1, column=0, columnspan=3, padx=10, pady=6, sticky="w")

    def _build_results(self) -> None:
        panel = ctk.CTkFrame(self)
        panel.grid(row=0, column=2, padx=(10, 20), pady=20, sticky="nsew")
        panel.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(panel, text="Resultado", font=ctk.CTkFont(size=18, weight="bold")).grid(
            row=0, column=0, columnspan=3, padx=15, pady=(15, 5), sticky="w")
        self.result_label = ctk.CTkLabel(panel, text="", font=ctk.CTkFont(size=22, weight="bold"),
                                         wraplength=380, justify="left")
        self.result_label.grid(row=1, column=0, columnspan=3, padx=15, pady=(0, 15), sticky="w")

        ctk.CTkLabel(panel, text="Probabilidad por clase", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=2, column=0, columnspan=3, padx=15, pady=(0, 8), sticky="w")
        for i, name in enumerate(core.class_names(), start=3):
            name_label = ctk.CTkLabel(panel, text=CLASS_LABELS_ES[name], anchor="w", width=140)
            name_label.grid(row=i, column=0, padx=(15, 5), pady=6, sticky="w")
            bar = ctk.CTkProgressBar(panel, progress_color=BAR_COLOR)
            bar.grid(row=i, column=1, padx=5, pady=6, sticky="ew")
            percent = ctk.CTkLabel(panel, text="", width=60, anchor="e")
            percent.grid(row=i, column=2, padx=(5, 15), pady=6, sticky="e")
            self.bars[name] = {"name": name_label, "bar": bar, "percent": percent}

    # ----- behaviour -----
    def collect_profile(self) -> Dict[str, Any]:
        """Read widget values as raw survey answers understood by `core.encode`."""
        profile: Dict[str, Any] = {}
        for field, _label, mapping in OPTION_MENUS:
            profile[field] = mapping[self.option_vars[field].get()]
        for field, var in self.slider_vars.items():
            profile[field] = var.get()
        for field, var in self.switch_vars.items():
            profile[field] = var.get()
        return profile

    def classify(self) -> Optional[core.Prediction]:
        try:
            result = core.predict(self.collect_profile())
        except ValueError as error:
            self.result_label.configure(text=f"Entrada no válida: {error}", text_color=HIGHLIGHT_COLOR)
            return None
        self._show(result)
        return result

    def reset(self) -> None:
        for field, _label, mapping in OPTION_MENUS:
            self.option_vars[field].set(_label_for(mapping, DEFAULT_PROFILE[field]))
        for field, var in self.slider_vars.items():
            var.set(int(DEFAULT_PROFILE[field]))
            self._update_slider_label(field)
        for field, var in self.switch_vars.items():
            var.set(DEFAULT_PROFILE[field])
        self._clear_results()

    def _show(self, result: core.Prediction) -> None:
        self.result_label.configure(text=CLASS_LABELS_ES[result.label], text_color=HIGHLIGHT_COLOR)
        for name, widgets in self.bars.items():
            probability = result.probabilities[name]
            is_predicted = name == result.label
            widgets["bar"].set(probability)
            widgets["bar"].configure(progress_color=HIGHLIGHT_COLOR if is_predicted else BAR_COLOR)
            widgets["percent"].configure(text=f"{probability:.1%}")
            weight = "bold" if is_predicted else "normal"
            widgets["name"].configure(font=ctk.CTkFont(weight=weight))

    def _clear_results(self) -> None:
        self.result_label.configure(text="Ajuste las medidas y presione Clasificar.",
                                    text_color=("gray20", "gray80"))
        for widgets in self.bars.values():
            widgets["bar"].set(0)
            widgets["bar"].configure(progress_color=BAR_COLOR)
            widgets["percent"].configure(text="")
            widgets["name"].configure(font=ctk.CTkFont(weight="normal"))

    def _update_slider_label(self, field: str) -> None:
        spec = next(s for s in SLIDERS if s.field == field)
        self.slider_value_labels[field].configure(text=spec.describe(self.slider_vars[field].get()))

    def _on_slider(self, field: str) -> None:
        self._update_slider_label(field)
        self._on_change()

    def _on_change(self) -> None:
        if self.auto_classify.get():
            self.classify()


def main() -> None:
    """Entry point for `uv run app`."""
    app = ObesityApp()
    app.mainloop()


if __name__ == "__main__":
    main()
