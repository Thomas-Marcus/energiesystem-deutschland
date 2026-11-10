#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMARD-Auswertungen · THE PATTERN MIND · Energiesystem Deutschland
=================================================================

Analysen
  1. Residuallast-Dauerlinie (mehrere Jahre überlagert)
  2. Heatmap Day-Ahead-Preis (Stunde × Monat) plus Heatmap negativer Preisstunden
  3. Marktwert Solar und Wind gegenüber Base-Preis je Monat (Capture Rate)
  4. Kennzahlen-Tabelle mit Datenvollständigkeit und Validierung gegen BNetzA-Referenzwerte
  5. Hypothesentests H1 (Kannibalisierung), H2 (gesicherte Leistung), H3 (Flexibilitätswert)

Datenquellen
  --quelle api    SMARD-Chart-API der Bundesnetzagentur (Standard, mit lokalem Cache)
  --quelle csv    CSV-Exporte aus dem SMARD-Downloadcenter (smard.de → Downloadcenter)
  --quelle demo   synthetische Testdaten, ausschließlich für den Funktionstest

Beispiele
  python smard_analyse.py --jahre 2025
  python smard_analyse.py --jahre 2024 2025 --ausgabe ergebnisse
  python smard_analyse.py --jahre 2015 2020 2025          (Test H2 über zehn Jahre)
  python smard_analyse.py --jahre 2025 --quelle csv --csv erzeugung.csv verbrauch.csv preise.csv
  python smard_analyse.py --jahre 2025 --quelle demo

Voraussetzungen
  pip install pandas numpy matplotlib

Ausgabe (Ordner --ausgabe)
  01_residuallast_dauerlinie.png
  02_preis_heatmap_<jahr>.png
  03_marktwert_solar_wind_<jahr>.png
  04_hypothesen_h1_h3.png
  hypothesen.txt, hypothesen_monatsbasis.csv
  kennzahlen.csv            (Semikolon, Dezimalkomma – öffnet direkt in deutschem Excel)
  marktwerte_monat_<jahr>.csv
  stundenwerte.csv          (bereinigter Stundendatensatz für Tableau / Power BI)
  validierung.txt
"""
from __future__ import annotations

import argparse
import calendar
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
TZ = "Europe/Berlin"
SMARD_BASIS = "https://www.smard.de/app/chart_data"

# SMARD-Filter-IDs der Chart-API und zugehörige Region
FILTER = {
    "preis":    (4169, "DE-LU"),  # Day-Ahead-Großhandelspreis DE/LU [€/MWh]
    "netzlast": (410,  "DE"),     # Stromverbrauch: Netzlast [MWh]
    "pv":       (4068, "DE"),     # Erzeugung Photovoltaik [MWh]
    "wind_on":  (4067, "DE"),     # Erzeugung Wind an Land [MWh]
    "wind_off": (1225, "DE"),     # Erzeugung Wind auf See [MWh]
}

# Spaltenpräfixe in Downloadcenter-CSVs (neues und älteres Format)
CSV_PRAEFIXE = {
    "preis":    ["Deutschland/Luxemburg ["],
    "netzlast": ["Netzlast [", "Gesamt (Netzlast)"],
    "pv":       ["Photovoltaik ["],
    "wind_on":  ["Wind Onshore [", "Wind an Land"],
    "wind_off": ["Wind Offshore [", "Wind auf See"],
}

# Referenzwerte: Bundesnetzagentur, Pressemitteilungen 03.01.2025 und 05.01.2026
REFERENZ = {
    2024: {"Day-Ahead-Mittel [€/MWh]": 78.51, "Stunden < 0 €/MWh": 457,
           "Stunden > 300 €/MWh": 41},
    2025: {"Day-Ahead-Mittel [€/MWh]": 89.32, "Stunden < 0 €/MWh": 573,
           "Stunden > 300 €/MWh": 40},
}

# Farben aus dem Deck
CI = {"navy": "#13294B", "blau": "#3A78B5", "teal": "#1A8C9E", "gruen": "#268C78",
      "gold": "#D49A2F", "rot": "#C8505A", "grau": "#6B7A90", "hell": "#F4F7FB",
      "linie": "#E3E8EF"}
MONATE = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun", "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]
QUELLE = "Quelle: Bundesnetzagentur | SMARD.de · eigene Berechnung THE PATTERN MIND"


def de(x: float, nd: int = 1) -> str:
    """Zahl im deutschen Format (1.234,5)."""
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ---------------------------------------------------------------------------
# Daten laden: SMARD-API
# ---------------------------------------------------------------------------
def _hole_json(url: str, cache: Path | None) -> dict:
    if cache is not None and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    anfrage = urllib.request.Request(
        url, headers={"User-Agent": "THE-PATTERN-MIND-SMARD-Analyse/1.0"})
    with urllib.request.urlopen(anfrage, timeout=30) as antwort:
        daten = json.loads(antwort.read().decode("utf-8"))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(daten), encoding="utf-8")
    time.sleep(0.2)  # Server schonen
    return daten


def lade_reihe_api(name: str, jahre: list[int], aufloesung: str, cache_dir: Path) -> pd.Series:
    """Lädt eine Zeitreihe wochenweise aus der SMARD-Chart-API."""
    fid, region = FILTER[name]
    index = _hole_json(f"{SMARD_BASIS}/{fid}/{region}/index_{aufloesung}.json", None)
    start_ms = pd.Timestamp(f"{min(jahre)}-01-01", tz=TZ).value // 10**6
    ende_ms = pd.Timestamp(f"{max(jahre) + 1}-01-01", tz=TZ).value // 10**6
    puffer = 8 * 86_400_000          # eine Datei umfasst eine Woche
    jetzt_ms = time.time() * 1000
    punkte: list = []
    for ts in index["timestamps"]:
        if not (start_ms - puffer <= ts < ende_ms):
            continue
        url = f"{SMARD_BASIS}/{fid}/{region}/{fid}_{region}_{aufloesung}_{ts}.json"
        # nur abgeschlossene Wochen cachen
        cache = cache_dir / f"{fid}_{region}_{aufloesung}_{ts}.json" if ts + puffer < jetzt_ms else None
        punkte.extend(_hole_json(url, cache).get("series", []))
    if not punkte:
        raise RuntimeError(f"Keine API-Daten für '{name}' im Zeitraum {jahre}.")
    zeit = pd.to_datetime([p[0] for p in punkte], unit="ms", utc=True).tz_convert(TZ)
    werte = pd.to_numeric(pd.Series([p[1] for p in punkte], index=zeit), errors="coerce")
    werte = werte[~werte.index.duplicated(keep="last")].sort_index()
    return werte.rename(name)


# ---------------------------------------------------------------------------
# Daten laden: Downloadcenter-CSV
# ---------------------------------------------------------------------------
def _lokalisiere(naiv: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Ortszeit → zeitzonenbewusst. Bei der Rückstellung im Oktober tritt 02:xx doppelt auf:
    erstes Auftreten = Sommerzeit, zweites = Winterzeit."""
    sommerzeit = ~naiv.duplicated(keep="first")
    return naiv.tz_localize(TZ, ambiguous=sommerzeit, nonexistent="shift_forward")


def lade_csv(pfade: list[str]) -> dict[str, pd.Series]:
    reihen: dict[str, pd.Series] = {}
    for pfad in pfade:
        df = pd.read_csv(pfad, sep=";", decimal=",", thousands=".", na_values=["-", ""],
                         encoding="utf-8-sig", low_memory=False)
        if "Datum von" in df.columns:
            naiv = pd.to_datetime(df["Datum von"], format="%d.%m.%Y %H:%M")
        elif {"Datum", "Anfang"} <= set(df.columns):
            naiv = pd.to_datetime(df["Datum"] + " " + df["Anfang"], format="%d.%m.%Y %H:%M")
        else:
            raise ValueError(f"{pfad}: keine Datumsspalte ('Datum von' oder 'Datum'/'Anfang').")
        index = _lokalisiere(pd.DatetimeIndex(naiv))
        for name, praefixe in CSV_PRAEFIXE.items():
            if name in reihen:
                continue
            spalte = next((c for c in df.columns if any(c.startswith(p) for p in praefixe)), None)
            if spalte is not None:
                s = pd.to_numeric(df[spalte], errors="coerce")
                s.index = index
                reihen[name] = s.rename(name)
                print(f"  {name:<9} ← {Path(pfad).name}: '{spalte}'")
    fehlend = set(FILTER) - set(reihen)
    if fehlend:
        raise ValueError(f"In den CSV-Dateien fehlen Spalten für: {', '.join(sorted(fehlend))}")
    return reihen


# ---------------------------------------------------------------------------
# Daten laden: synthetische Demodaten
# ---------------------------------------------------------------------------
def demo_daten(jahre: list[int]) -> dict[str, pd.Series]:
    """Plausibel geformte, aber frei erfundene Daten. Nur für den Funktionstest."""
    rng = np.random.default_rng(2025)
    idx = pd.date_range(f"{min(jahre)}-01-01", f"{max(jahre) + 1}-01-01",
                        freq="h", tz=TZ, inclusive="left")
    n = len(idx)
    stunde = idx.hour.to_numpy() + 0.5
    sommer = np.cos(2 * np.pi * (idx.dayofyear.to_numpy() - 172) / 365.25)
    taglaenge = 12 + 4 * sommer
    pv_form = np.clip(np.sin(np.pi * (stunde - (13 - taglaenge / 2)) / taglaenge), 0, None)
    wolken = np.repeat(rng.beta(2.5, 1.8, size=n // 24 + 1), 24)[:n]
    pv = 50_000 * (0.5 + 0.5 * sommer) * pv_form * wolken
    z = np.empty(n)
    x = 0.0
    for i in range(n):
        x = 0.985 * x + rng.normal(0, 0.17)
        z[i] = x
    wind = np.clip(14_000 * (1 - 0.35 * sommer) * np.exp(0.6 * z / z.std()), 300, 58_000)
    last = (46_000 + 14_000 * 0.5 * (1 - np.cos(2 * np.pi * (stunde - 4) / 24))
            - 5_000 * sommer - 7_000 * (idx.dayofweek.to_numpy() >= 5))
    rest = last - pv - wind
    preis = (25 + 0.0022 * rest + rng.normal(0, 10, n)
             + np.where(rest > 48_000, rng.exponential(80, n), 0))
    daten = {"preis": np.clip(preis, -150, None), "netzlast": last, "pv": pv,
             "wind_on": 0.8 * wind, "wind_off": 0.2 * wind}
    return {k: pd.Series(v, index=idx, name=k) for k, v in daten.items()}


# ---------------------------------------------------------------------------
# Aufbereitung und Kennzahlen
# ---------------------------------------------------------------------------
def auf_stundenbasis(reihen: dict[str, pd.Series]) -> pd.DataFrame:
    """Viertelstunden → Stunden: Energie [MWh] summieren, Preise mitteln."""
    spalten = {}
    for name, s in reihen.items():
        r = s.resample("h")
        spalten[name] = r.mean() if name == "preis" else r.sum(min_count=1)
    return pd.DataFrame(spalten)


def residuallast(d: pd.DataFrame) -> pd.Series:
    """SMARD-Definition: Netzlast minus Wind an Land, Wind auf See und Photovoltaik."""
    return (d["netzlast"] - d["pv"] - d["wind_on"] - d["wind_off"]).rename("residuallast")


def kennzahlen(df: pd.DataFrame, jahr: int) -> dict:
    d = df.loc[str(jahr)]
    p = d["preis"].dropna()
    res = residuallast(d).dropna()
    voll = d.dropna(subset=["preis", "pv", "wind_on", "wind_off"])
    wind = voll["wind_on"] + voll["wind_off"]
    base = voll["preis"].mean()
    mw_pv = (voll["pv"] * voll["preis"]).sum() / voll["pv"].sum()
    mw_wind = (wind * voll["preis"]).sum() / wind.sum()
    k = {"Jahr": jahr, "Erwartete Stunden": 8784 if calendar.isleap(jahr) else 8760}
    k.update({f"Stunden mit Daten: {c}": int(d[c].notna().sum()) for c in FILTER})
    k.update({
        "Day-Ahead-Mittel [€/MWh]": p.mean(),
        "Day-Ahead-Median [€/MWh]": p.median(),
        "Day-Ahead-Minimum [€/MWh]": p.min(),
        "Day-Ahead-Maximum [€/MWh]": p.max(),
        "Stunden < 0 €/MWh": int((p < 0).sum()),
        "Stunden > 200 €/MWh": int((p > 200).sum()),
        "Stunden > 300 €/MWh": int((p > 300).sum()),
        "Netzlast [TWh]": d["netzlast"].sum() / 1e6,
        "Photovoltaik [TWh]": d["pv"].sum() / 1e6,
        "Wind an Land [TWh]": d["wind_on"].sum() / 1e6,
        "Wind auf See [TWh]": d["wind_off"].sum() / 1e6,
        "Wind+PV in % der Netzlast": 100 * (d["pv"].sum() + d["wind_on"].sum()
                                             + d["wind_off"].sum()) / d["netzlast"].sum(),
        "Residuallast max [GW]": res.max() / 1000,
        "Residuallast 100-h-Wert [GW]": res.sort_values(ascending=False).iloc[min(99, len(res) - 1)] / 1000,
        "Residuallast min [GW]": res.min() / 1000,
        "Residuallast Mittel [GW]": res.mean() / 1000,
        "Stunden Residuallast < 0": int((res < 0).sum()),
        "Marktwert Solar [€/MWh]": mw_pv,
        "Marktwert Wind [€/MWh]": mw_wind,
        "Capture Rate Solar [%]": 100 * mw_pv / base,
        "Capture Rate Wind [%]": 100 * mw_wind / base,
    })
    return k


def marktwerte_monat(df: pd.DataFrame, jahr: int) -> pd.DataFrame:
    d = df.loc[str(jahr)].dropna(subset=["preis", "pv", "wind_on", "wind_off"])
    wind = d["wind_on"] + d["wind_off"]
    m = d.index.month
    erg = pd.DataFrame({
        "Base [€/MWh]": d["preis"].groupby(m).mean(),
        "Marktwert Solar [€/MWh]": (d["pv"] * d["preis"]).groupby(m).sum() / d["pv"].groupby(m).sum(),
        "Marktwert Wind [€/MWh]": (wind * d["preis"]).groupby(m).sum() / wind.groupby(m).sum(),
    })
    erg["Capture Rate Solar [%]"] = 100 * erg["Marktwert Solar [€/MWh]"] / erg["Base [€/MWh]"]
    erg["Capture Rate Wind [%]"] = 100 * erg["Marktwert Wind [€/MWh]"] / erg["Base [€/MWh]"]
    erg.index.name = "Monat"
    return erg


def validiere(k: dict) -> list[str]:
    jahr = k["Jahr"]
    zeilen = [f"{jahr}"]
    luecken = [c for c in FILTER if k[f"Stunden mit Daten: {c}"] < k["Erwartete Stunden"]]
    if luecken:
        for c in luecken:
            zeilen.append(f"  Datenlücke {c}: {k[f'Stunden mit Daten: {c}']} von "
                          f"{k['Erwartete Stunden']} Stunden")
    else:
        zeilen.append("  Datenvollständigkeit: alle Reihen vollständig")
    ref = REFERENZ.get(jahr)
    if not ref:
        zeilen.append("  Keine BNetzA-Referenzwerte hinterlegt.")
        return zeilen
    for schluessel, soll in ref.items():
        ist = k[schluessel]
        zeilen.append(f"  {schluessel:<26} eigen {ist:>9.2f} | BNetzA {soll:>9.2f} | Δ {ist - soll:+.2f}")
    if jahr >= 2025:
        zeilen.append("  Hinweis: Seit 01.10.2025 wird der Day-Ahead-Markt in Viertelstunden gehandelt. "
                      "Stundenmittel weichen deshalb geringfügig von der BNetzA-Zählweise ab.")
    return zeilen


# ---------------------------------------------------------------------------
# Grafiken
# ---------------------------------------------------------------------------
def _stil(ax) -> None:
    ax.set_facecolor("white")
    for seite in ("top", "right"):
        ax.spines[seite].set_visible(False)
    for seite in ("left", "bottom"):
        ax.spines[seite].set_color(CI["grau"])
    ax.tick_params(colors=CI["navy"], labelsize=9)
    ax.grid(axis="y", color=CI["linie"], lw=0.8)
    ax.set_axisbelow(True)


def _rahmen(fig, titel: str, untertitel: str, demo: bool) -> None:
    fig.patch.set_facecolor(CI["hell"])
    fig.text(0.06, 0.955, titel, fontsize=16, weight="bold", color=CI["navy"], va="top")
    fig.text(0.06, 0.900, untertitel, fontsize=10.5, color=CI["grau"], va="top")
    fig.text(0.06, 0.022, QUELLE, fontsize=8, color=CI["grau"])
    if demo:
        fig.text(0.5, 0.5, "SYNTHETISCHE DEMODATEN – NICHT ZITIEREN", rotation=22,
                 fontsize=30, color=CI["rot"], alpha=0.2, ha="center", va="center", weight="bold")


def plot_dauerlinie(df: pd.DataFrame, jahre: list[int], ziel: Path, demo: bool) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6.75))
    palette = [CI["grau"], CI["teal"], CI["blau"], CI["navy"]]
    farben = palette[-len(jahre):] if len(jahre) <= 4 else plt.cm.Blues(np.linspace(0.35, 1, len(jahre)))
    for jahr, farbe in zip(jahre, farben):
        r = residuallast(df.loc[str(jahr)]).dropna().sort_values(ascending=False).to_numpy() / 1000
        ax.plot(np.arange(1, len(r) + 1), r, color=farbe,
                lw=2.4 if jahr == jahre[-1] else 1.4, label=str(jahr))

    letztes = jahre[-1]
    r = residuallast(df.loc[str(letztes)]).dropna().sort_values(ascending=False).to_numpy() / 1000
    x = np.arange(1, len(r) + 1)
    spanne = r[0] - r[-1]
    ax.fill_between(x, r, 0, where=r < 0, color=CI["rot"], alpha=0.25, interpolate=True,
                    label="Überschuss: Wind + PV > Netzlast")
    ax.fill_between(x, r, 0, where=r >= 0, color=CI["navy"], alpha=0.05, interpolate=True)
    ax.axhline(0, color=CI["navy"], lw=0.9)

    n_neg = int((r < 0).sum())
    r100 = r[min(99, len(r) - 1)]
    pfeil = dict(arrowstyle="-|>", color=CI["navy"], lw=0.9)
    ax.annotate(f"Spitze {letztes}: {de(r[0])} GW", xy=(1, r[0]),
                xytext=(len(r) * 0.08, r[0] - 0.02 * spanne), fontsize=10,
                color=CI["navy"], arrowprops=pfeil, va="center")
    ax.annotate(f"Die obersten {de(r[0] - r100)} GW\nwerden nur in 100 Stunden gebraucht",
                xy=(100, r100), xytext=(len(r) * 0.10, r100 - 0.30 * spanne),
                fontsize=10, color=CI["navy"], arrowprops=pfeil)
    if n_neg:
        ax.annotate(f"{de(n_neg, 0)} Stunden mit Überschuss",
                    xy=(len(r) - n_neg / 2, r[-1] * 0.4),
                    xytext=(len(r) * 0.55, r[-1] * 0.75), fontsize=10,
                    color=CI["rot"], weight="bold", arrowprops=dict(pfeil, color=CI["rot"]))

    _stil(ax)
    ax.set_xlim(0, len(r))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: de(v, 0)))
    ax.set_xlabel("Stunden des Jahres, absteigend nach Residuallast sortiert", color=CI["navy"])
    ax.set_ylabel("Residuallast [GW]", color=CI["navy"])
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    titel = (f"{letztes}: In {de(n_neg, 0)} Stunden übertrafen Wind und PV die gesamte Netzlast"
             if n_neg else f"{letztes}: Residuallast zwischen {de(r[-1])} und {de(r[0])} GW")
    _rahmen(fig, titel,
            "Residuallast = Netzlast − Wind an Land − Wind auf See − Photovoltaik · stündlich · "
            "PV ohne Eigenverbrauch hinter dem Zähler", demo)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.07, right=0.97)
    pfad = ziel / "01_residuallast_dauerlinie.png"
    fig.savefig(pfad, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return pfad


def plot_heatmap(df: pd.DataFrame, jahr: int, ziel: Path, demo: bool) -> Path:
    p = df.loc[str(jahr), "preis"].dropna()
    stunden, monate = range(24), range(1, 13)
    mittel_tab = (p.groupby([p.index.hour, p.index.month]).mean().unstack()
                  .reindex(index=stunden, columns=monate))
    neg_tab = ((p < 0).groupby([p.index.hour, p.index.month]).sum().unstack()
               .reindex(index=stunden, columns=monate).fillna(0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7.4))
    jahresmittel = p.mean()
    norm = TwoSlopeNorm(vmin=min(np.nanmin(mittel_tab.values), jahresmittel - 1),
                        vcenter=jahresmittel,
                        vmax=max(np.nanmax(mittel_tab.values), jahresmittel + 1))
    im1 = ax1.imshow(mittel_tab.values, aspect="auto", cmap="RdBu_r", norm=norm)
    rot_skala = LinearSegmentedColormap.from_list("neg", ["#FFFFFF", CI["rot"]])
    im2 = ax2.imshow(neg_tab.values, aspect="auto", cmap=rot_skala, vmin=0)

    for (i, j), wert in np.ndenumerate(mittel_tab.values):
        if not np.isnan(wert):
            hell = abs(norm(wert) - 0.5) > 0.32
            ax1.text(j, i, f"{wert:.0f}", ha="center", va="center", fontsize=6,
                     color="white" if hell else CI["navy"])
    neg_max = max(neg_tab.values.max(), 1)
    for (i, j), wert in np.ndenumerate(neg_tab.values):
        if wert > 0:
            ax2.text(j, i, f"{wert:.0f}", ha="center", va="center", fontsize=6,
                     color="white" if wert > 0.6 * neg_max else CI["navy"])

    for ax, untertitel in ((ax1, "Ø Day-Ahead-Preis [€/MWh]"),
                           (ax2, "Anzahl Stunden mit negativem Preis")):
        ax.set_xticks(range(12), MONATE, fontsize=9, color=CI["navy"])
        ax.set_yticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)],
                      fontsize=9, color=CI["navy"])
        ax.set_title(untertitel, fontsize=11, color=CI["navy"], loc="left", weight="bold")
        for seite in ax.spines.values():
            seite.set_visible(False)
    ax1.set_ylabel("Stunde (Ortszeit MEZ/MESZ)", color=CI["navy"])
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02).set_label(
        f"weiß = Jahresmittel {de(jahresmittel)} €/MWh", fontsize=8)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02).set_label("Stunden", fontsize=8)

    n_neg = int(neg_tab.values.sum())
    if n_neg:
        anteil = neg_tab.loc[10:15].values.sum() / n_neg
        titel = (f"{jahr}: {de(100 * anteil, 0)} % der {de(n_neg, 0)} negativen Preisstunden "
                 f"fielen zwischen 10 und 16 Uhr")
    else:
        titel = f"{jahr}: Day-Ahead-Preis nach Tageszeit und Monat"
    _rahmen(fig, titel, "Stündliche Day-Ahead-Preise DE/LU · Mittelwert bzw. Anzahl je Zelle", demo)
    fig.subplots_adjust(top=0.82, bottom=0.10, left=0.07, right=0.97, wspace=0.18)
    pfad = ziel / f"02_preis_heatmap_{jahr}.png"
    fig.savefig(pfad, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return pfad


def plot_marktwert(mw: pd.DataFrame, k: dict, jahr: int, ziel: Path, demo: bool) -> Path:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})
    x = np.arange(len(mw))
    ax1.plot(x, mw["Base [€/MWh]"], color=CI["navy"], lw=2.4, marker="o", ms=4,
             label="Base (Ø aller Stunden)")
    ax1.plot(x, mw["Marktwert Solar [€/MWh]"], color=CI["gold"], lw=2.2, marker="o", ms=4,
             label="Marktwert Solar")
    ax1.plot(x, mw["Marktwert Wind [€/MWh]"], color=CI["teal"], lw=2.2, marker="o", ms=4,
             label="Marktwert Wind")
    ax1.fill_between(x, mw["Marktwert Solar [€/MWh]"], mw["Base [€/MWh]"],
                     where=mw["Marktwert Solar [€/MWh]"] < mw["Base [€/MWh]"],
                     color=CI["gold"], alpha=0.15, interpolate=True)
    ax1.set_ylabel("€/MWh", color=CI["navy"])
    ax1.legend(frameon=False, ncol=3, fontsize=9, loc="upper center")

    breite = 0.38
    ax2.bar(x - breite / 2, mw["Capture Rate Solar [%]"], breite, color=CI["gold"], label="Solar")
    ax2.bar(x + breite / 2, mw["Capture Rate Wind [%]"], breite, color=CI["teal"], label="Wind")
    ax2.axhline(100, color=CI["navy"], lw=1, ls="--")
    ax2.text(len(mw) - 0.5, 101, "100 % = Base-Preis", ha="right", va="bottom",
             fontsize=8, color=CI["navy"])
    ax2.set_ylabel("Capture Rate [%]", color=CI["navy"])
    ax2.set_xticks(x, [MONATE[m - 1] for m in mw.index])
    ax2.legend(frameon=False, ncol=2, fontsize=9, loc="lower right", bbox_to_anchor=(1, 1.0))
    for ax in (ax1, ax2):
        _stil(ax)

    titel = (f"{jahr}: Solarstrom erzielte {de(k['Capture Rate Solar [%]'], 0)} % des Durchschnittspreises, "
             f"Wind {de(k['Capture Rate Wind [%]'], 0)} %")
    _rahmen(fig, titel,
            "Mengengewichteter Marktwert je Monat vs. Base-Preis · Näherung aus SMARD-Stundenwerten, "
            "kein offizieller EEG-Marktwert (netztransparenz.de)", demo)
    fig.subplots_adjust(top=0.84, bottom=0.10, left=0.08, right=0.97, hspace=0.22)
    pfad = ziel / f"03_marktwert_solar_wind_{jahr}.png"
    fig.savefig(pfad, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return pfad


# ---------------------------------------------------------------------------
# Hypothesentests H1–H3
# ---------------------------------------------------------------------------
def _spearman(x: pd.Series, y: pd.Series) -> float:
    """Rangkorrelation nach Spearman ohne SciPy."""
    paar = pd.concat([x, y], axis=1).dropna()
    if len(paar) < 3:
        return float("nan")
    return float(paar.iloc[:, 0].rank().corr(paar.iloc[:, 1].rank()))


def monatsbasis(df: pd.DataFrame, jahre: list[int]) -> pd.DataFrame:
    """Je Monat: PV-Anteil an der Netzlast, Capture Rate Solar, mittlere Tagesspreizung."""
    zeilen = []
    for jahr in jahre:
        d = df.loc[str(jahr)].dropna(subset=["preis", "pv", "netzlast"])
        for monat, m in d.groupby(d.index.month):
            tage = m["preis"].groupby(m.index.date)
            spreizung = (tage.max() - tage.min()).mean()
            zeilen.append({
                "Jahr": jahr, "Monat": monat,
                "PV-Anteil an Netzlast [%]": 100 * m["pv"].sum() / m["netzlast"].sum(),
                "Capture Rate Solar [%]": 100 * (m["pv"] * m["preis"]).sum() / m["pv"].sum() / m["preis"].mean(),
                "Tagesspreizung [€/MWh]": spreizung,
            })
    return pd.DataFrame(zeilen)


def hypothesentests(df: pd.DataFrame, jahre: list[int], kz: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    mb = monatsbasis(df, jahre)
    rho1 = _spearman(mb["PV-Anteil an Netzlast [%]"], mb["Capture Rate Solar [%]"])
    rho3 = _spearman(mb["PV-Anteil an Netzlast [%]"], mb["Tagesspreizung [€/MWh]"])
    n = len(mb)
    z = [f"Hypothesentests auf Monatsbasis (n = {n} Monate, Jahre: {', '.join(map(str, jahre))})",
         "=" * 60,
         "H1 Kannibalisierung: Capture Rate Solar sinkt mit steigendem PV-Anteil.",
         f"   Spearman rho = {rho1:+.2f}  ->  "
         + ("gestützt (negativer Zusammenhang)" if rho1 < 0 else "nicht gestützt (kein negativer Zusammenhang)"),
         "H3 Flexibilitätswert: Tagesspreizung steigt mit steigendem PV-Anteil.",
         f"   Spearman rho = {rho3:+.2f}  ->  "
         + ("gestützt (positiver Zusammenhang)" if rho3 > 0 else "nicht gestützt (kein positiver Zusammenhang)"),
         "H2 Gesicherte Leistung: Spitzen-Residuallast sinkt kaum trotz Wind- und PV-Zubau."]
    for k in kz:
        wpv = k["Photovoltaik [TWh]"] + k["Wind an Land [TWh]"] + k["Wind auf See [TWh]"]
        z.append(f"   {k['Jahr']}: Wind+PV {wpv:6.1f} TWh | Residuallast max {k['Residuallast max [GW]']:5.1f} GW"
                 f" | 100-h-Wert {k['Residuallast 100-h-Wert [GW]']:5.1f} GW")
    if len(kz) >= 2:
        a, b = kz[0], kz[-1]
        d_wpv = ((b["Photovoltaik [TWh]"] + b["Wind an Land [TWh]"] + b["Wind auf See [TWh]"])
                 / (a["Photovoltaik [TWh]"] + a["Wind an Land [TWh]"] + a["Wind auf See [TWh]"]) - 1)
        d_max = b["Residuallast max [GW]"] / a["Residuallast max [GW]"] - 1
        z.append(f"   {a['Jahr']}->{b['Jahr']}: Wind+PV {100 * d_wpv:+.0f} %, Spitzen-Residuallast {100 * d_max:+.0f} %  ->  "
                 + ("nicht prüfbar (kein Zuwachs bei Wind+PV)" if d_wpv <= 0
                    else "gestützt" if d_max > -0.5 * d_wpv else "nicht gestützt"))
    else:
        z.append("   Für H2 mindestens zwei Jahre angeben, z. B. --jahre 2015 2020 2025.")
    z += ["",
          "Hinweis: Rangkorrelationen auf Monatsbasis zeigen Zusammenhänge, keine Kausalität.",
          "Saison wirkt als Störgröße (Sommer = viel PV und andere Nachfrage). Belastbarer:",
          "mehrere Jahre und Vergleich gleicher Kalendermonate über die Jahre."]
    return mb, z


def plot_hypothesen(mb: pd.DataFrame, ziel: Path, demo: bool) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.4))
    jahre = sorted(mb["Jahr"].unique())
    palette = [CI["grau"], CI["teal"], CI["blau"], CI["navy"]]
    farben = palette[-len(jahre):] if len(jahre) <= 4 else plt.cm.Blues(np.linspace(0.35, 1, len(jahre)))
    for jahr, farbe in zip(jahre, farben):
        t = mb[mb["Jahr"] == jahr]
        ax1.scatter(t["PV-Anteil an Netzlast [%]"], t["Capture Rate Solar [%]"], color=farbe, s=46, label=str(jahr))
        ax2.scatter(t["PV-Anteil an Netzlast [%]"], t["Tagesspreizung [€/MWh]"], color=farbe, s=46, label=str(jahr))
    rho1 = _spearman(mb["PV-Anteil an Netzlast [%]"], mb["Capture Rate Solar [%]"])
    rho3 = _spearman(mb["PV-Anteil an Netzlast [%]"], mb["Tagesspreizung [€/MWh]"])
    ax1.axhline(100, color=CI["navy"], lw=1, ls="--")
    ax1.set_title(f"H1 · Capture Rate Solar   (Spearman ρ = {de(rho1, 2)})",
                  loc="left", fontsize=11, color=CI["navy"], weight="bold")
    ax2.set_title(f"H3 · Mittlere Tagesspreizung   (Spearman ρ = {de(rho3, 2)})",
                  loc="left", fontsize=11, color=CI["navy"], weight="bold")
    ax1.set_ylabel("Capture Rate Solar [%]", color=CI["navy"])
    ax2.set_ylabel("Ø tägliche Preisspreizung Max − Min [€/MWh]", color=CI["navy"])
    for ax in (ax1, ax2):
        _stil(ax)
        ax.set_xlabel("PV-Anteil an der Netzlast im Monat [%]", color=CI["navy"])
        ax.legend(frameon=False, fontsize=9)
    _rahmen(fig, "Hypothesentest: Mehr PV senkt den Solar-Marktwert und erhöht den Wert von Flexibilität?",
            "Ein Punkt = ein Monat · Rangkorrelation nach Spearman · Zusammenhang, keine Kausalität", demo)
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.07, right=0.98, wspace=0.22)
    pfad = ziel / "04_hypothesen_h1_h3.png"
    fig.savefig(pfad, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return pfad


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def csv_de(obj: pd.DataFrame, pfad: Path, **kw) -> None:
    obj.to_csv(pfad, sep=";", decimal=",", encoding="utf-8-sig", float_format="%.3f", **kw)


# ---------------------------------------------------------------------------
# Hauptprogramm
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="SMARD-Auswertungen · THE PATTERN MIND")
    ap.add_argument("--jahre", nargs="+", type=int, default=[2025], help="z. B. 2024 2025")
    ap.add_argument("--quelle", choices=["api", "csv", "demo"], default="api")
    ap.add_argument("--csv", nargs="*", default=[], help="Downloadcenter-CSV-Dateien")
    ap.add_argument("--aufloesung", choices=["hour", "quarterhour"], default="hour",
                    help="nur API: Auflösung der Rohdaten")
    ap.add_argument("--ausgabe", default="smard_ergebnisse")
    ap.add_argument("--cache", default=".smard_cache")
    a = ap.parse_args()

    jahre = sorted(set(a.jahre))
    ziel = Path(a.ausgabe)
    ziel.mkdir(parents=True, exist_ok=True)
    demo = a.quelle == "demo"

    print(f"Datenquelle: {a.quelle} · Jahre: {', '.join(map(str, jahre))}")
    if a.quelle == "api":
        reihen = {}
        for name in FILTER:
            print(f"  lade {name} …")
            reihen[name] = lade_reihe_api(name, jahre, a.aufloesung, Path(a.cache))
    elif a.quelle == "csv":
        if not a.csv:
            ap.error("--quelle csv benötigt --csv <dateien>")
        reihen = lade_csv(a.csv)
    else:
        print("  ACHTUNG: synthetische Demodaten – Ergebnisse nicht verwenden.")
        reihen = demo_daten(jahre)

    df = auf_stundenbasis(reihen)
    start = pd.Timestamp(f"{jahre[0]}-01-01", tz=TZ)
    ende = pd.Timestamp(f"{jahre[-1] + 1}-01-01", tz=TZ)
    df = df[(df.index >= start) & (df.index < ende)].copy()
    df["residuallast"] = residuallast(df)

    alle_k, validierung, dateien = [], [], []
    for jahr in jahre:
        if df.loc[str(jahr)].empty:
            print(f"  {jahr}: keine Daten – übersprungen")
            continue
        k = kennzahlen(df, jahr)
        alle_k.append(k)
        validierung += validiere(k)
        mw = marktwerte_monat(df, jahr)
        csv_de(mw.rename(index=lambda m: MONATE[m - 1]), ziel / f"marktwerte_monat_{jahr}.csv")
        dateien.append(plot_heatmap(df, jahr, ziel, demo))
        dateien.append(plot_marktwert(mw, k, jahr, ziel, demo))
    if not alle_k:
        print("Keine auswertbaren Daten.")
        return 1
    gueltige_jahre = [k["Jahr"] for k in alle_k]
    dateien.insert(0, plot_dauerlinie(df, gueltige_jahre, ziel, demo))
    mb, hyp = hypothesentests(df, gueltige_jahre, alle_k)
    csv_de(mb.set_index(["Jahr", "Monat"]), ziel / "hypothesen_monatsbasis.csv")
    (ziel / "hypothesen.txt").write_text("\n".join(hyp) + "\n", encoding="utf-8")
    dateien.append(plot_hypothesen(mb, ziel, demo))

    tabelle = pd.DataFrame(alle_k).set_index("Jahr").T
    tabelle.index.name = "Kennzahl"
    csv_de(tabelle, ziel / "kennzahlen.csv")
    export = df.copy()
    export.index = export.index.tz_convert(TZ).strftime("%Y-%m-%d %H:%M%z")
    export.index.name = "zeit_ortszeit"
    csv_de(export, ziel / "stundenwerte.csv")
    kopf = ["Validierung gegen BNetzA-Referenzwerte", "=" * 40]
    if demo:
        kopf.append("SYNTHETISCHE DEMODATEN – Abweichungen sind erwartbar und bedeutungslos.")
    (ziel / "validierung.txt").write_text("\n".join(kopf + validierung) + "\n", encoding="utf-8")

    print("\n" + "\n".join(kopf + validierung))
    print("\n" + "\n".join(hyp))
    print(f"\nErgebnisse in: {ziel.resolve()}")
    for d in dateien:
        print(f"  {d.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
