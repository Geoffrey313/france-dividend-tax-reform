"""Render bilingual PFU-US-11 firm tables and event-study figures from result objects.

Inputs are the harmonized common-sample result dictionary and the country-composition robustness
payload. Outputs are LaTeX fragments written to both manuscript language trees. No reported number
is typed into a manuscript source by hand.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common import paths


SPECIFICATION = "C_controls_no_ptbi"
FIGURE_DIRS = {
    language: paths.ROOT / "manuscript" / language / "ssrn" / "figures"
    for language in ("fr", "en")
}

PANELS = (
    ("private_large_vl", ("dlog_capital", "cap_increase", "equity_growth_real", "equity_ratio"),
     "Financement des sociétés privées", "Private-firm financing"),
    ("private_large_vl", ("asset_growth_real", "tangible_growth_real", "intangible_growth_real"),
     "Investissement des sociétés privées", "Private-firm investment"),
    ("private_large_vl", ("reserves_ratio", "cash_at", "fin_debt_ratio", "dlog_debt"),
     "Ressources des sociétés privées", "Private-firm resources"),
    ("public_listed", ("net_ext_equity_at", "equity_growth_real", "equity_ratio"),
     "Financement des sociétés cotées", "Listed-firm financing"),
    ("public_listed", ("asset_growth_real", "tangible_growth_real", "intangible_growth_real", "capex_at",
                       "sales_growth_real", "operating_roa"),
     "Investissement et activité des sociétés cotées", "Listed-firm investment and activity"),
    ("public_listed", ("retained_earnings_at", "cash_at", "fin_debt_ratio", "dlog_debt"),
     "Ressources des sociétés cotées", "Listed-firm resources"),
)

LABELS = {
    "fr": {
        "dlog_capital": "Capital souscrit",
        "cap_increase": "Hausse du capital",
        "equity_growth_real": "Fonds propres",
        "equity_ratio": "Ratio de fonds propres",
        "asset_growth_real": "Actif total",
        "tangible_growth_real": "Immobilisations corporelles",
        "intangible_growth_real": "Immobilisations incorporelles",
        "reserves_ratio": "Fonds propres non contribués",
        "net_ext_equity_at": "Financement externe",
        "capex_at": "Dépenses d'investissement",
        "sales_growth_real": "Chiffre d'affaires",
        "operating_roa": "Rentabilité opérationnelle",
        "retained_earnings_at": "Bénéfices non distribués",
        "cash_at": "Trésorerie",
        "fin_debt_ratio": "Levier financier",
        "dlog_debt": "Dette financière",
        "lag_book_leverage": "Levier $(t-1)$",
        "lag_log_assets_real": "Taille $(t-1)$",
        "lag_log_firm_age": "Âge $(t-1)$",
    },
    "en": {
        "dlog_capital": "Subscribed capital",
        "cap_increase": "Capital increase",
        "equity_growth_real": "Equity",
        "equity_ratio": "Equity ratio",
        "asset_growth_real": "Total assets",
        "tangible_growth_real": "Tangible assets",
        "intangible_growth_real": "Intangible assets",
        "reserves_ratio": "Non-contributed equity",
        "net_ext_equity_at": "External financing",
        "capex_at": "Capital expenditure",
        "sales_growth_real": "Sales",
        "operating_roa": "Operating return on assets",
        "retained_earnings_at": "Retained earnings",
        "cash_at": "Cash",
        "fin_debt_ratio": "Financial leverage",
        "dlog_debt": "Financial debt",
        "lag_book_leverage": "Leverage $(t-1)$",
        "lag_log_assets_real": "Size $(t-1)$",
        "lag_log_firm_age": "Age $(t-1)$",
    },
}

EVENT_GROUPS = {
    "private_large_vl": (
        ("dlog_capital", "asset_growth_real", "tangible_growth_real"),
        ("equity_growth_real", "intangible_growth_real", "reserves_ratio"),
    ),
    "public_listed": (
        ("net_ext_equity_at", "asset_growth_real", "tangible_growth_real"),
        ("equity_growth_real", "intangible_growth_real", "retained_earnings_at"),
        ("capex_at", "sales_growth_real", "operating_roa"),
    ),
}


def _decimal(value: float, digits: int, language: str, *, signed: bool = False) -> str:
    body = f"{value:+,.{digits}f}" if signed else f"{value:,.{digits}f}"
    body = body.replace(",", r"\,")
    if language == "fr":
        body = body.replace(".", "{,}")
    return body


def _integer(value: int) -> str:
    return f"{int(value):,}".replace(",", r"\,")


def _cell(output: dict, population: str, shock: str, outcome: str) -> tuple[dict, dict]:
    result = output["populations"][population][shock][outcome]
    return result, result["specifications"][SPECIFICATION]


def _contrast(output: dict, population: str, outcome: str) -> dict:
    key = "private_cross_shock_contrasts" if population == "private_large_vl" else "public_cross_shock_contrasts"
    return output[key]["2018_precovid"][outcome]


def _row(label: str, values: list[str]) -> str:
    return f"{label} & " + " & ".join(values) + r" \\" + "\n"


def _results_panel(output: dict, population: str, outcomes: tuple[str, ...], title: str,
                   language: str, shock: str, reform_year: int) -> str:
    n = len(outcomes)
    yes = "Oui" if language == "fr" else "Yes"
    country = "Pays" if language == "fr" else "Country"
    labels = LABELS[language]
    after = "France $\\times$ après" if language == "fr" else "France $\\times$ after"
    placebo = "Placebo $p$ bilatéral" if language == "fr" else "Two-sided placebo $p$"
    lines = [
        rf"\begin{{tblr}}{{rowsep=0pt,colspec={{X[1.75,l]*{{{n}}}{{X[1,c]}}}}}}",
        r"\toprule",
        rf"\SetCell[c={n + 1}]{{c}}{{\textbf{{{title}}}}}" + " &" * n + r" \\",
        r"\midrule",
        _row("", [f"({index})" for index in range(1, n + 1)]).rstrip(),
        _row("Variables expliquées :" if language == "fr" else "Dependent variables:",
             [rf"\textit{{{labels[outcome]}}}" for outcome in outcomes]).rstrip(),
        r"\midrule",
    ]
    results = [_cell(output, population, shock, outcome) for outcome in outcomes]
    lines.append(_row(f"\\textit{{{after} {reform_year}}}", [
        f"${_decimal(100 * spec['coef'], 3, language, signed=True)}$"
        for _, spec in results
    ]).rstrip())
    lines.append(_row("", [
        f"$({_decimal(spec['coef'] / spec['std_error'], 2, language, signed=True)})$"
        for _, spec in results
    ]).rstrip())
    lines.append(_row("", ["" for _ in outcomes]).rstrip())
    lines.append(_row(r"\SetRow{font=\scriptsize\itshape} " + placebo, [
        f"${_decimal(spec['placebo_p'], 3, language)}$" for _, spec in results
    ]).rstrip())
    lines.append(_row("", ["" for _ in outcomes]).rstrip())
    for control in ("lag_book_leverage", "lag_log_assets_real", "lag_log_firm_age"):
        estimates = [spec.get("control_estimates", {}).get(control) for _, spec in results]
        lines.append(_row(labels[control], [
            (f"${_decimal(100 * estimate['coef'], 3, language, signed=True)}$"
             if estimate is not None else "--")
            for estimate in estimates
        ]).rstrip())
        lines.append(_row("", [
            (f"$({_decimal(estimate['coef'] / estimate['std_error'], 2, language, signed=True)})$"
             if estimate is not None else "")
            for estimate in estimates
        ]).rstrip())
    lines.extend([
        _row("", ["" for _ in outcomes]).rstrip(),
        _row("Effets fixes entreprise" if language == "fr" else "Firm fixed effects",
             [yes] * n).rstrip(),
        _row("Effets fixes secteur $\\times$ année" if language == "fr" else
             "Industry $\\times$ year fixed effects", [yes] * n).rstrip(),
        _row("Regroupement" if language == "fr" else "Clustering", [country] * n).rstrip(),
        _row("", ["" for _ in outcomes]).rstrip(),
        _row("Observations", [
            _integer(_cell(output, population, shock, outcome)[0]["n_obs"])
            for outcome in outcomes
        ]).rstrip(),
        r"\bottomrule",
        r"\end{tblr}",
    ])
    return "\n".join(lines)


def _results_table(output: dict, language: str, shock: str, reform_year: int) -> str:
    notes = (
        f"Le tableau porte uniquement sur la réforme de {reform_year}. Les coefficients sont exprimés "
        "en points de pourcentage. Les statistiques conventionnelles, "
        "regroupées au niveau pays, figurent entre parenthèses et restent descriptives avec six groupes. "
        "Les coefficients des contrôles sont également exprimés en points de pourcentage ; un tiret indique "
        "qu'un contrôle est exclu parce qu'il est mécaniquement lié à la variable expliquée. Les valeurs $p$ "
        "principales sont les rangs placebo bilatéraux obtenus en traitant successivement "
        "chacun des six pays. Les échantillons sont communs à toutes les variables de chaque panneau. "
        "Toutes les spécifications absorbent les effets fixes entreprise et secteur $\\times$ année et "
        "incluent Taille $(t-1)$, Âge $(t-1)$ et Levier $(t-1)$ lorsque ce dernier n'est pas mécaniquement "
        "lié à la variable expliquée. Les variables sont winsorisées aux premier et quatre-vingt-dix-neuvième "
        "percentiles pays-année."
        if language == "fr" else
        f"The table reports only the {reform_year} reform. Coefficients are expressed in percentage points. "
        "Conventional country-clustered statistics are "
        "reported in parentheses and remain descriptive with six clusters. Control coefficients are also "
        "expressed in percentage points; a dash denotes a control excluded because it is mechanically related "
        "to the dependent variable. The primary $p$-values are "
        "two-sided placebo ranks obtained by treating each of the six countries in turn. Samples are common "
        "to all variables within each panel. Every specification absorbs firm and industry $\\times$ year "
        "fixed effects and includes Size $(t-1)$, Age $(t-1)$, and Leverage $(t-1)$ when leverage is not "
        "mechanically related to the dependent variable. Variables are winsorized at the first and "
        "ninety-ninth percentiles within country-year."
    )
    population_labels = {
        "private_large_vl": ("sociétés privées", "Private Firms", "private"),
        "public_listed": ("sociétés cotées", "Listed Firms", "listed"),
    }
    blocks = [r"\begin{landscape}", r"{"]
    for population in ("private_large_vl", "public_listed"):
        fr_population, en_population, label_population = population_labels[population]
        caption = (
            f"Résultats comptables des {fr_population} : réforme de {reform_year}"
            if language == "fr" else
            f"{en_population} Accounting Results: {reform_year} Reform"
        )
        panels = [panel for panel in PANELS if panel[0] == population]
        for panel_index, (_, outcomes, fr_title, en_title) in enumerate(panels):
            blocks.extend([r"\begin{table}[p]", r"\centering"])
            if panel_index:
                blocks.append(r"\ContinuedFloat")
                continued = " (suite)" if language == "fr" else " (continued)"
                blocks.append(rf"\caption[]{{{caption}{continued}}}")
            else:
                blocks.extend([
                    rf"\caption{{{caption}}}\label{{table:firm-square-results-{reform_year}-{label_population}}}",
                    rf"\tablenotes{{{notes}\\}}",
                ])
                if population == "private_large_vl":
                    blocks.append(rf"\label{{table:firm-square-results-{reform_year}}}")
            panel_letter = chr(ord("A") + panel_index)
            panel_name = fr_title if language == "fr" else en_title
            panel_title = f"Panel {panel_letter} : {panel_name}"
            blocks.append(r"\scriptsize")
            blocks.append(_results_panel(
                output, population, outcomes, panel_title, language, shock, reform_year,
            ))
            blocks.append(r"\end{table}")
    blocks.extend([r"}", r"\end{landscape}"])
    return "\n".join(blocks) + "\n"


def _robustness_table(payload: dict, language: str) -> str:
    rows = payload["summary"]
    lookup = {(row["population"], row["shock"], row["outcome"]): row for row in rows}
    caption = (
        "Robustesse à la composition et au poids des pays"
        if language == "fr" else "Robustness to Country Composition and Weighting"
    )
    note = (
        "La colonne de référence reproduit la spécification principale sur le même échantillon d'estimation. "
        "La colonne de pondération donne le même poids total à chacun des six pays. L'intervalle rapporte "
        "le minimum et le maximum des cinq estimations obtenues en retirant successivement un pays témoin. "
        "Ces comparaisons sont descriptives."
        if language == "fr" else
        "The baseline column reproduces the preferred specification on the same estimation sample. Equal-country "
        "weighting assigns the same total weight to each of the six countries. The interval gives the minimum "
        "and maximum across the five estimates that omit one control country in turn. These comparisons are "
        "descriptive."
    )
    header = (
        ["Variable", "Référence", "Poids pays égaux", "Intervalle sans un témoin", "Signe stable"]
        if language == "fr" else
        ["Variable", "Baseline", "Equal-country weights", "Leave-one-control range", "Stable sign"]
    )
    lines = [r"{", r"\hfuzz=9pt", r"\begin{table}[p]", r"\centering",
             rf"\caption{{{caption}}}\label{{table:firm-country-robustness}}",
             rf"\tablenotes{{{note}\\}}", r"\tiny",
             r"\begin{tblr}{colspec={X[2.2,l]*{4}{X[1,c]}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for population, outcomes, fr_title, en_title in PANELS:
        title = fr_title if language == "fr" else en_title
        lines.append(rf"\SetCell[c=5]{{c}}{{\textbf{{{title}}}}} & & & & \\")
        lines.append(r"\midrule")
        for outcome in outcomes:
            if (population, "2018_precovid", outcome) not in lookup:
                continue
            row = lookup[(population, "2018_precovid", outcome)]
            sign = ("Oui" if row["all_variants_same_sign"] else "Non") if language == "fr" else (
                "Yes" if row["all_variants_same_sign"] else "No"
            )
            interval = (
                f"[{_decimal(100 * row['leave_one_out_min'], 3, language, signed=True)}; "
                f"{_decimal(100 * row['leave_one_out_max'], 3, language, signed=True)}]"
            )
            lines.append(_row(
                rf"\textit{{{LABELS[language][outcome]}}}",
                [
                    _decimal(100 * row["baseline_coef"], 3, language, signed=True),
                    _decimal(100 * row["equal_country_coef"], 3, language, signed=True),
                    interval, sign,
                ],
            ).rstrip())
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tblr}", r"\end{table}", r"}"])
    return "\n".join(lines) + "\n"


def _pretrend(value: float, language: str) -> str:
    if value < 0.001:
        return "$<0{,}001$" if language == "fr" else "$<0.001$"
    return f"${_decimal(value, 3, language)}$"


def _pretrend_p(value: float, language: str) -> str:
    """Format joint-test p-values and make conventional rejections easy to scan."""
    if value < 0.001:
        body = "<0{,}001" if language == "fr" else "<0.001"
    else:
        body = _decimal(value, 3, language)
    return f"$\\mathbf{{{body}}}$" if value < 0.05 else f"${body}$"


def _event_study_panel(output: dict, population: str, shock: str, outcomes: tuple[str, ...],
                       language: str, reform_year: int) -> str:
    """Render pre- and post-reform event coefficients for one outcome family."""
    n = len(outcomes)
    yes = "Oui" if language == "fr" else "Yes"
    country = "Pays" if language == "fr" else "Country"
    reference_year = reform_year - 1
    years = sorted({
        int(value["year"])
        for outcome in outcomes
        for value in _cell(output, population, shock, outcome)[1]["event_study"]
        if int(value["year"]) != reference_year
    })
    lines = [rf"\begin{{tblr}}{{width=0.97\linewidth,rowsep=0pt,colspec={{X[1.7,l]*{{{n}}}{{X[1,c]}}}}}}", r"\toprule",
             _row("", [f"({index})" for index in range(1, n + 1)]).rstrip(),
             _row("Variables expliquées :" if language == "fr" else "Dependent variables:",
                  [rf"\textit{{{LABELS[language][outcome]}}}" for outcome in outcomes]).rstrip(),
             r"\midrule"]
    for year in years:
        relative = year - reform_year
        if relative == 0:
            relative_label = "t"
        elif relative > 0:
            relative_label = f"t+{relative}"
        else:
            relative_label = f"t{relative}"
        label = (f"Année $({relative_label})$" if language == "fr" else
                 f"Year $({relative_label})$")
        coefficients = []
        statistics = []
        for outcome in outcomes:
            _, specification = _cell(output, population, shock, outcome)
            event_lookup = {int(value["year"]): value for value in specification["event_study"]}
            value = event_lookup[year]
            coefficients.append(f"${_decimal(100 * float(value['coef']), 2, language, signed=True)}$")
            statistics.append(
                f"$({_decimal(float(value['coef']) / float(value['std_error']), 2, language, signed=True)})$"
            )
        lines.extend([_row(label, coefficients).rstrip(), _row("", statistics).rstrip()])
    diagnostics = [
        _cell(output, population, shock, outcome)[1]["pretrend_joint_test"]
        for outcome in outcomes
    ]
    lines.extend([
        _row("", ["" for _ in outcomes]).rstrip(),
        _row("Wald conjoint des pré-tendances" if language == "fr" else "Joint pre-trend Wald", [
            _decimal(float(value.get("statistic", float("nan"))), 2, language)
            for value in diagnostics
        ]).rstrip(),
        _row("Valeur $p$ conjointe" if language == "fr" else "Joint $p$-value", [
            _pretrend_p(float(value.get("conventional_p", float("nan"))), language)
            for value in diagnostics
        ]).rstrip(),
        _row("", ["" for _ in outcomes]).rstrip(),
        _row("Effets fixes entreprise" if language == "fr" else "Firm fixed effects", [yes] * n).rstrip(),
        _row("Effets fixes secteur $\\times$ année" if language == "fr" else
             "Industry $\\times$ year fixed effects", [yes] * n).rstrip(),
        _row("Levier, taille et âge $(t-1)$" if language == "fr" else
             "Leverage, size, and age $(t-1)$", [yes] * n).rstrip(),
        _row("Regroupement" if language == "fr" else "Clustering", [country] * n).rstrip(),
        _row("", ["" for _ in outcomes]).rstrip(),
        _row("Observations entreprise-année" if language == "fr" else "Firm-year observations", [
            _integer(_cell(output, population, shock, outcome)[0]["n_obs"]) for outcome in outcomes
        ]).rstrip(),
        r"\bottomrule", r"\end{tblr}",
    ])
    return "\n".join(lines)


def _event_study_table(output: dict, language: str, shock: str, reform_year: int) -> str:
    caption = ("Étude d'événement autour de la réforme de {}" if language == "fr" else
               "Event-study Estimates around the {} Reform")
    tables = []
    for shock, reform_year in ((shock, reform_year),):
        label = f"table:firm-event-study-{reform_year}"
        reference_year = reform_year - 1
        note = (
            f"Ce tableau présente les coefficients antérieurs et postérieurs à la réforme dans les études d'événement. Dans chaque "
            f"panneau, les variables expliquées figurent en colonnes et les années relatives en lignes ; $t-1$, soit "
            f"{reference_year}, est la période de référence omise. Les coefficients sont exprimés en points de "
            "pourcentage et les statistiques de Student regroupées au niveau pays figurent entre parenthèses. Le test "
            "de Wald évalue conjointement l'hypothèse que tous les coefficients antérieurs sont nuls ; les valeurs $p$ "
            "inférieures à 0{,}05 sont en gras. Chaque panneau utilise un échantillon commun à ses variables, la "
            "spécification C, six groupes pays et les contrôles mesurés en $t-1$."
            if language == "fr" else
            f"This table presents pre- and post-reform coefficients from the event studies. Within each panel, dependent "
            f"variables appear in columns and relative years in rows; $t-1$, or {reference_year}, is the omitted "
            "reference period. Coefficients are in percentage points and country-clustered t-values are in "
            "parentheses. The Wald test jointly evaluates whether all pre-reform coefficients equal zero; $p$-values "
            "below 0.05 are in bold. Each panel uses a sample common to its outcomes, specification C, six "
            "country clusters, and controls measured at $t-1$."
        )
        lines = [r"{", r"\captionsetup[subfloat]{position=top}"]
        for index, (population, outcomes, fr_title, en_title) in enumerate(PANELS):
            wide = len(outcomes) >= 5
            if index:
                lines.append(r"\clearpage")
            if wide:
                lines.append(r"\begin{landscape}")
            lines.append(r"\begin{table}[p]" if index == 0 else r"\begin{table}[H]")
            if index:
                lines.append(r"\ContinuedFloat")
            lines.append(r"\centering")
            if index == 0:
                lines.extend([rf"\caption{{{caption.format(reform_year)}}}\label{{{label}}}",
                              r"\xdef\firmeventtablecounter{\number\value{table}}",
                              rf"\tablenotes{{{note}\\}}"])
            panel_title = fr_title if language == "fr" else en_title
            safe_population = population.replace("_", "-")
            panel_label = f"table-panel:firm-event-study-{reform_year}-{safe_population}-{index + 1}"
            lines.extend([r"\footnotesize" if len(outcomes) <= 4 else r"\scriptsize",
                          rf"\subfloat[{panel_title}\label{{{panel_label}}}]{{",
                          _event_study_panel(output, population, shock, outcomes, language, reform_year),
                          r"}", r"\end{table}"])
            if wide:
                lines.append(r"\end{landscape}")
        # caption/subfig decrements the table counter for continuation floats that
        # have no repeated main caption. Restore it before starting the next table.
        lines.extend([r"\setcounter{table}{\firmeventtablecounter}", r"}"])
        tables.append("\n".join(lines))
    return "\n\n".join(tables) + "\n"


def _specification_table(output: dict, language: str) -> str:
    """Render the prespecified A/B/C sequence and one-control omissions."""
    specifications = (
        ("A_firm_year_fe_common_sample", "A"),
        ("B_firm_industry_year_fe_common_sample", "B"),
        ("C_controls_no_ptbi", "C"),
        ("C_controls_no_ptbi_no_age", "C sans âge" if language == "fr" else "C without age"),
        ("C_controls_no_ptbi_no_size", "C sans taille" if language == "fr" else "C without size"),
        ("C_controls_no_ptbi_no_leverage", "C sans levier" if language == "fr" else "C without leverage"),
    )
    selected = (
        ("private_large_vl", "dlog_capital"),
        ("private_large_vl", "asset_growth_real"),
        ("private_large_vl", "tangible_growth_real"),
        ("private_large_vl", "reserves_ratio"),
        ("public_listed", "net_ext_equity_at"),
        ("public_listed", "capex_at"),
    )
    caption = ("Séquence des spécifications et omissions de contrôles" if language == "fr"
               else "Specification Sequence and Control Omissions")
    note = (
        "Chaque cellule rapporte le coefficient France $\\times$ après 2018, en points de pourcentage, "
        "sur la fenêtre 2013--2019 et l'échantillon commun de la famille. A absorbe les effets fixes "
        "entreprise et année ; B remplace ces derniers par secteur $\\times$ année ; C ajoute les "
        "contrôles mesurés en $t-1$. Les trois dernières lignes retirent un contrôle à la fois."
        if language == "fr" else
        "Each cell reports the France $\\times$ post-2018 coefficient, in percentage points, over the "
        "2013--2019 window and the common sample for the relevant family. A absorbs firm and year fixed effects; B "
        "replaces year effects with industry $\\times$ year effects; C adds controls measured at $t-1$. The final "
        "three rows omit one control at a time."
    )
    lines = [r"{", r"\hfuzz=9pt", r"\begin{table}[p]", r"\centering", rf"\caption{{{caption}}}\label{{table:firm-specifications}}",
             rf"\tablenotes{{{note}\\}}", r"\tiny",
             r"\begin{tblr}{colspec={X[1.35,l]*{6}{X[1,c]}}}", r"\toprule",
             (r"Spécification & Capital souscrit & Actif & Immobilisations corporelles & Fonds propres non contribués & Financement externe & Investissement \\"
              if language == "fr" else
              r"Specification & Subscribed capital & Assets & Tangible assets & Non-contributed equity & External financing & Capital expenditure \\"),
             r"\midrule"]
    for key, label in specifications:
        values = []
        for population, outcome in selected:
            spec = output["populations"][population]["2018_precovid"][outcome]["specifications"][key]
            values.append(_decimal(100 * spec["coef"], 3, language, signed=True))
        lines.append(_row(label, values).rstrip())
    lines.extend([r"\bottomrule", r"\end{tblr}", r"\end{table}", r"}"])
    return "\n".join(lines) + "\n"


def _descriptive_table(language: str) -> str:
    frame = pd.read_csv(paths.RESULTS / "firm_growth_harmonized_descriptive.csv")
    caption = ("Statistiques descriptives des échantillons d'estimation finaux"
               if language == "fr" else
               "Descriptive Statistics for the Final Estimation Samples")
    panels = [
        ("2018_precovid", "private_large_vl", "capital",
         "2018 — sociétés privées, financement" if language == "fr" else
         "2018 — private firms, financing"),
        ("2018_precovid", "private_large_vl", "investment",
         "2018 — sociétés privées, investissement" if language == "fr" else
         "2018 — private firms, investment"),
        ("2018_precovid", "private_large_vl", "resources",
         "2018 — sociétés privées, ressources conservées" if language == "fr" else
         "2018 — private firms, retained resources"),
        ("2018_precovid", "public_listed", "complete",
         "2018 — sociétés cotées, échantillon commun" if language == "fr" else
         "2018 — listed firms, common sample"),
        ("2013", "private_large_vl", "capital",
         "2013 — sociétés privées, financement" if language == "fr" else
         "2013 — private firms, financing"),
        ("2013", "private_large_vl", "investment",
         "2013 — sociétés privées, investissement" if language == "fr" else
         "2013 — private firms, investment"),
        ("2013", "private_large_vl", "resources",
         "2013 — sociétés privées, ressources conservées" if language == "fr" else
         "2013 — private firms, retained resources"),
        ("2013", "public_listed", "complete",
         "2013 — sociétés cotées, échantillon commun" if language == "fr" else
         "2013 — listed firms, common sample"),
    ]
    lines = [r"{", r"\hfuzz=4pt", r"\tiny", r"\begin{longtblr}[", rf"caption={{{caption}}},label={{table:stats-descriptives}},",
             r"note{}={Chaque panneau porte sur l'échantillon commun final de la réforme et de la famille indiquées. Le nombre d'observations correspond donc exactement à celui du panneau de résultats associé. Toutes les statistiques et magnitudes économiques comptables sont calculées sur ces mêmes observations ; les calculs utilisent les valeurs non arrondies sous-jacentes. Les variables continues sont winsorisées par pays-année.}" if language == "fr" else
             r"note{}={Each panel uses the final common sample for the stated reform and outcome family. The observation count therefore exactly matches the associated results panel. All accounting statistics and economic magnitudes use these same observations; calculations use the underlying unrounded values. Continuous variables are winsorized within country-year.}",
             r"]{colspec={X[2.5,l]*{6}{X[1,r]}},rowhead=1}", r"\toprule",
             (r"Variable & N & Moyenne & Écart-type & P25 & Médiane & P75 \\"
              if language == "fr" else r"Variable & N & Mean & SD & P25 & Median & P75 \\"), r"\midrule"]
    for panel_index, (shock, population, family, panel_label) in enumerate(panels):
        subset = frame[
            frame["shock"].eq(shock)
            & frame["population"].eq(population)
            & frame["family"].eq(family)
        ]
        if subset.empty:
            raise ValueError(f"missing descriptive panel: {shock}/{population}/{family}")
        letter = chr(ord("A") + panel_index)
        lines.append(
            rf"\SetCell[c=7]{{l}} \textbf{{Panel {letter} : {panel_label}}} & & & & & & \\"
        )
        for row in subset.itertuples(index=False):
            if row.variable not in LABELS[language]:
                continue
            lines.append(_row(rf"\textit{{{LABELS[language][row.variable]}}}", [
                _integer(row.n),
                *[_decimal(float(getattr(row, statistic)), 3, language)
                  for statistic in ("mean", "sd", "p25", "median", "p75")],
            ]).rstrip())
        if panel_index != len(panels) - 1:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{longtblr}", r"}"])
    return "\n".join(lines) + "\n"


def _window_table(output: dict, language: str) -> str:
    caption = ("Sensibilité à la fenêtre incluant la crise sanitaire" if language == "fr"
               else "Sensitivity to the Pandemic-inclusive Window")
    note = (
        "La fenêtre principale s'arrête en 2019. La sensibilité prolonge la période jusqu'en 2025 et "
        "inclut donc la crise sanitaire et les dispositifs de soutien. Les coefficients sont en points "
        "de pourcentage ; les valeurs $p$ sont les rangs placebo bilatéraux sur six pays."
        if language == "fr" else
        "The main window ends in 2019. The sensitivity extends through 2025 and therefore includes the "
        "pandemic and support programs. Coefficients are in percentage points; $p$-values are two-sided "
        "placebo ranks across six countries."
    )
    lines = [r"{", r"\begin{table}[p]", r"\centering",
             rf"\caption{{{caption}}}\label{{table:firm-window-sensitivity}}",
             rf"\tablenotes{{{note}\\}}", r"\scriptsize",
             r"\begin{tblr}{colspec={X[1.35,l]X[2,l]*{4}{X[1,c]}}}", r"\toprule",
             (r"Population & Variable & 2013--2019 & Rang $p$ & 2013--2025 & Rang $p$ \\"
              if language == "fr" else
              r"Population & Variable & 2013--2019 & Rank $p$ & 2013--2025 & Rank $p$ \\"),
             r"\midrule"]
    for population, outcomes, _, _ in PANELS:
        pop_label = (("Privées" if language == "fr" else "Private")
                     if population == "private_large_vl" else
                     ("Cotées" if language == "fr" else "Listed"))
        for outcome in outcomes:
            preferred = output["populations"][population]["2018_precovid"][outcome]["specifications"][SPECIFICATION]
            full = output["populations"][population]["2018_full"][outcome]["specifications"][SPECIFICATION]
            lines.append(_row(pop_label, [
                rf"\textit{{{LABELS[language][outcome]}}}",
                _decimal(100 * preferred["coef"], 3, language, signed=True),
                _decimal(preferred["placebo_p"], 3, language),
                _decimal(100 * full["coef"], 3, language, signed=True),
                _decimal(full["placebo_p"], 3, language),
            ]).rstrip())
    lines.extend([r"\bottomrule", r"\end{tblr}", r"\end{table}", r"}"])
    return "\n".join(lines) + "\n"


def _recent_coverage_table(language: str) -> str:
    """Report recent harmonized-panel counts so provisional 2025 coverage is visible."""
    target = paths.RESULTS / "harmonized_firm_panel_coverage.csv"
    if not target.exists():
        target = (
            paths.DATA / "derived" / "square_samples_canonical" / "recent_coverage.csv"
        )
    frame = pd.read_csv(target)
    frame = frame[
        frame["variable"].eq("lag_log_assets_real")
        & frame["year"].between(2022, 2025)
    ].copy()
    counts = frame.pivot(index=["population", "country"], columns="year", values="rows")
    populations = {
        "private_large_vl": "Privées" if language == "fr" else "Private",
        "public_listed": "Cotées" if language == "fr" else "Listed",
    }
    caption = (
        "Couverture récente des panels d'entreprises"
        if language == "fr" else "Recent Firm-panel Coverage"
    )
    note = (
        "Le tableau indique le nombre de lignes société-année disponibles dans chaque panel "
        "harmonisé avant la sélection propre à chaque régression. Le millésime 2025 est provisoire "
        "et particulièrement incomplet dans Orbis ; ses coefficients dynamiques sont donc des "
        "résultats de sensibilité."
        if language == "fr" else
        "The table reports available firm-year rows in each harmonized panel before "
        "regression-specific selection. The 2025 vintage is provisional and particularly "
        "incomplete in Orbis; its dynamic coefficients are therefore sensitivity results."
    )
    lines = [
        r"{", r"\begin{table}[p]", r"\centering",
        rf"\caption{{{caption}}}\label{{table:firm-recent-coverage}}",
        rf"\tablenotes{{{note}\\}}", r"\footnotesize",
        r"\begin{tblr}{colspec={X[1.8,l]X[0.8,c]*{4}{X[1,c]}}}", r"\toprule",
        ((r"Population & Pays & 2022 & 2023 & 2024 & 2025 \\") if language == "fr" else
         (r"Population & Country & 2022 & 2023 & 2024 & 2025 \\")),
        r"\midrule",
    ]
    for (population, country), row in counts.sort_index().iterrows():
        lines.append(_row(populations[population], [country, *(
            _integer(row.get(year, 0)) for year in range(2022, 2026)
        )]).rstrip())
    lines.extend([r"\bottomrule", r"\end{tblr}", r"\end{table}", r"}"])
    return "\n".join(lines) + "\n"


def _event_figure(output: dict, population: str, outcomes: tuple[str, ...], group_index: int,
                  language: str) -> str:
    private = population == "private_large_vl"
    prefix = "" if private else "public_"
    label = f"fig:firm-events-{'private' if private else 'listed'}-{group_index + 1}"
    caption = (
        ("Études d'événement des sociétés privées" if private else "Études d'événement des sociétés cotées")
        if language == "fr" else
        ("Private-firm event studies" if private else "Listed-firm event studies")
    )
    p_values = []
    for outcome in outcomes:
        values = []
        for shock in ("2013", "2018_precovid"):
            _, spec = _cell(output, population, shock, outcome)
            values.append(_pretrend(float(spec["pretrend_joint_test"].get("conventional_p", float("nan"))), language))
        p_values.append(f"{LABELS[language][outcome]}: 2013 {values[0]}, 2018 {values[1]}")
    note_start = (
        "Les panneaux de gauche portent sur 2013 avec 2012 comme référence ; ceux de droite portent sur "
        "2018 avec 2017 comme référence. L'ordonnée est en points de pourcentage et les barres "
        r"représentent les intervalles conventionnels à 95\,\%. "
        "Les valeurs $p$ des tests conjoints de pré-tendance sont : "
        if language == "fr" else
        "Left-hand panels cover 2013 with 2012 as the reference year; right-hand panels cover 2018 with "
        "2017 as the reference year. The vertical axis is in percentage points and bars are conventional "
        r"95\% intervals. Joint pre-trend $p$-values are: "
    )
    lines = [r"\begin{figure}[p]", r"\centering", r"\begin{tikzpicture}",
             r"\begin{groupplot}[",
             r"group style={group size=2 by 3,horizontal sep=1.4cm,vertical sep=0.85cm},",
             r"width=6.7cm,height=3.85cm,axis lines=left,tick align=outside,",
             r"tick label style={font=\scriptsize,/pgf/number format/1000 sep={}},",
             r"label style={font=\scriptsize},ylabel style={font=\scriptsize},xlabel={" + ("Année" if language == "fr" else "Year") + r"},",
             r"extra x ticks={2012.5,2017.5},extra x tick labels={},",
             r"extra x tick style={grid=major,grid style={densely dashed,black!55}},",
             r"extra y ticks={0},extra y tick labels={},",
             r"extra y tick style={grid=major,grid style={solid,black!35}},",
             r"error bars/y dir=both,error bars/y explicit,error bars/error mark=none,]",]
    for outcome_index, outcome in enumerate(outcomes):
        ylabel = LABELS[language][outcome]
        for shock in ("2013", "2018_precovid"):
            options = [rf"ylabel={{\textit{{{ylabel}}}}}" if shock == "2013" else "ylabel={}"]
            if outcome_index == 0:
                options.append("title={2013}" if shock == "2013" else "title={2018}")
            lines.append(r"\nextgroupplot[" + ",".join(options) + "]")
            name = f"{prefix}firm_accounting_{shock}_{outcome}.csv"
            lines.append(r"\addplot+[color=franceblue,mark=*] "
                         rf"table[x=year,y=coef,y error=ci95,col sep=comma]{{figures/{name}}};")
    lines.extend([r"\end{groupplot}", r"\end{tikzpicture}",
                  rf"\caption{{{caption}}}", rf"\label{{{label}}}",
                  rf"\fignote{{{note_start}{'; '.join(p_values)}.}}", r"\end{figure}"])
    return "\n".join(lines) + "\n"


def write_firm_reporting(output: dict, robustness: dict) -> list[Path]:
    """Write all bilingual firm-accounting table and figure fragments."""
    written: list[Path] = []
    for language, directory in FIGURE_DIRS.items():
        directory.mkdir(parents=True, exist_ok=True)
        fragments = {
            "firm_square_results_2018_table.tex": _results_table(
                output, language, "2018_precovid", 2018,
            ),
            "firm_square_results_2013_table.tex": _results_table(
                output, language, "2013", 2013,
            ),
            "firm_specification_table.tex": _specification_table(output, language),
            "firm_country_robustness_table.tex": _robustness_table(robustness, language),
            "firm_event_study_2018_table.tex": _event_study_table(
                output, language, "2018_full", 2018,
            ),
            "firm_event_study_2013_table.tex": _event_study_table(
                output, language, "2013", 2013,
            ),
            # Keep legacy generated paths as explicit tombstones so stale mixed-reform
            # tables cannot silently survive a regeneration in an existing checkout.
            "firm_square_results_table.tex": "% Superseded by the separate 2018 and 2013 tables.\n",
            "firm_pretrend_table.tex": "% Superseded by the separate 2018 and 2013 tables.\n",
            "firm_pretrend_2018_table.tex": "% Superseded by the 2018 event-study table.\n",
            "firm_pretrend_2013_table.tex": "% Superseded by the 2013 event-study table.\n",
            "firm_descriptive_table.tex": _descriptive_table(language),
            "firm_recent_coverage_table.tex": _recent_coverage_table(language),
            "firm_window_sensitivity_table.tex": _window_table(output, language),
        }
        for population, groups in EVENT_GROUPS.items():
            population_name = "private" if population == "private_large_vl" else "listed"
            for index, outcomes in enumerate(groups):
                fragments[f"firm_events_{population_name}_{index + 1}.tex"] = _event_figure(
                    output, population, outcomes, index, language,
                )
        for name, content in fragments.items():
            target = directory / name
            target.write_text(content, encoding="utf-8")
            written.append(target)
    return written
