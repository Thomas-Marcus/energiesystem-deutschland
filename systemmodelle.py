#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Systemmodelle · THE PATTERN MIND · Energiesystem Deutschland
============================================================

Ergänzt smard_analyse.py um sechs Systemmodelle und fünf Ergänzungsrechnungen.
smard_analyse.py muss im selben Ordner liegen (Datenzugriff, Stil, Hilfsfunktionen).

Modelle
  szenario2030      C1  Profil-Skalierung auf die Ausbauziele 2030, Lastvarianten, Speicherwirkung
  spitzen           C2  Spitzenzerlegung der Residuallast und Kapazitätskredit (prüft H2)
  dunkelflaute      C3  Dunkelflauten-Statistik und Speicherbedarf bei gesicherter Leistung
  batterie          C4  Batterie-Backtest mit realen Day-Ahead-Preisen, LCOS, Kapitalwert
  meritorder        C5  Merit-Order-Regression: Preis gegen Residuallast je Jahr
  kannibalisierung  C6  Robuste Kannibalisierung: Jahreswerte, Monats-Kontrollvariablen, t-Werte, Bootstrap
  flexwert          C8  Flexibilitätswert: tägliche Preisspreizung gegen PV-Anteil mit Saisonkontrolle (H3)
  aussenhandel      C7  Außenhandel und Pumpspeicher in Spitzenstunden, Dunkelflauten und negativen Preisstunden
Robustheit
  spitzen_robust    R1  Spitzenzerlegung für alle Jahre und Top-50/100/200/500-Stunden
  netzlast          R4  SMARD-Netzlast gegen Bruttostromverbrauch (AGEB/BDEW)
  (R2 Bootstrap in C6, R3 Zyklengrenze und Verschleiß-Sensitivität in C4)
Ergänzungen
  strompreisreal    D1  Haushaltsstrompreis 2017 nominal und real gegen aktuell
  europa            D2  EU-Haushaltsstrompreise in Euro und in Kaufkraftstandards (PPS)
  hpc               D3  HPC-Standortmodell mit Leistungspreis, Hochlauf, Batterie, Tornado
  heimspeicher      D4  Heimspeicher mit Alterung, Strompreispfaden, Solarspitzengesetz
  kommune           D5  Modellkommune: Pufferdauer und Überschuss am Sommertag

Aufruf
  python systemmodelle.py --module alle
  python systemmodelle.py --module szenario2030 spitzen --quelle demo
  python systemmodelle.py --module batterie --batteriejahre 2024 2025 --speicherdauer 1 2 4
  python systemmodelle.py --module alle --quelle csv --csv erzeugung.csv verbrauch.csv preise.csv

Datenbedarf
  SMARD (API, CSV oder Demo):  Preis, Netzlast, PV, Wind an Land, Wind auf See
  eingaben/installierte_leistung.csv   PV/Wind-Leistung je Jahresende (C1, C2, C3, D5)
  eingaben/strompreis_haushalt.csv     BDEW-Komponenten 2017 und aktuell (D1)
  eingaben/vpi.csv                     Destatis-Verbraucherpreisindex (D1)
  eingaben/eurostat_haushalt.csv       Fallback ohne Netz (D2)
  eingaben/ageb_bruttostromverbrauch.csv  Bruttostromverbrauch je Jahr (R4)
  eingaben/smard_aussenhandel_<jahr>.csv  optional: Handelsdaten aus dem SMARD-Downloadcenter (C7, sonst Abruf)
  Fehlende Dateien legt das Skript als Vorlage an; das betroffene Modul wird übersprungen.

Ausgaben (Ordner --ausgabe, Standard: systemmodelle_ergebnisse)
  10_szenario_2030_dauerlinien.png       11_szenario_2030_speicherwirkung.png
  12_spitzenzerlegung.png                13_kapazitaetskredit.png
  14_dunkelflauten.png                   15_speicherbedarf_gesicherte_leistung.png
  16_batterie_backtest.png               17_batterie_kapitalwert.png
  18_merit_order.png                     19_kannibalisierung_robust.png
  20_hpc_tornado.png                     21_heimspeicher_sensitivitaet.png
  22_strompreis_real.png                 23_strompreise_europa_eur_pps.png
  24_robust_spitzenzerlegung.png         25_kannibalisierung_bootstrap.png
  26_netzlast_plausibilisierung.png      27_batterie_verschleiss.png
  28_aussenhandel_kritische_stunden.png  29_flexibilitaet_tagesprofil.png
  30_flexibilitaetswert_h3.png
  je Modul: <modul>_kennzahlen.csv und <modul>_methodik.txt
  plausibilitaet.txt                     Ergebnis aller eingebauten Plausibilitätstests

Alle Annahmen stehen im Block KONFIGURATION und sind vom Nutzer zu bestätigen oder zu ersetzen.
"""
from __future__ import annotations

import argparse
import math
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

import smard_analyse as sa  # noqa: E402

# ===========================================================================
# KONFIGURATION – alle Annahmen an einer Stelle, jeweils mit Quelle
# ===========================================================================
# C1 · Ausbauziele 2030 in GW. EEG 2023 (PV, Wind an Land), WindSeeG (Wind auf See).
#      Geprüft 09.11.2026: BMWE-Dossier „Erneuerbare Energien“ nennt 215 / 115 / 30 GW.
ZIEL_2030_GW = {"pv": 215.0, "wind_on": 115.0, "wind_off": 30.0}
LASTVARIANTEN = {"V0": 0.00, "V1": 0.10, "V2": 0.20, "V3": 0.20}  # V3 = +20 %, Hälfte flexibel
SPEICHER_GWH = [0, 25, 50, 100]          # C1-Speichergrößen, Leistung = Energie / 4 h
ETA_RUNDLAUF = 0.90                       # Rundlauf-Wirkungsgrad Batterie

# C1-Sensitivität · Sektorkopplung 2030 (stilisierte Annahmen, als Sensitivität zu verstehen)
#   Energie: rund 6 Mio. Wärmepumpen × ~4 MWh/a ≈ 25 TWh; rund 8 Mio. E-Pkw × ~2,5 MWh/a ≈ 20 TWh
SEKTOR_2030_TWH = {"waermepumpen": 25.0, "e_mobilitaet": 20.0}
WP_MONATSANTEIL = [0.17, 0.15, 0.12, 0.08, 0.04, 0.01, 0.01, 0.01, 0.03, 0.08, 0.13, 0.17]  # Heizbedarf je Monat (Annahme)
WP_TAGESPROFIL = [0.9, 0.9, 0.9, 0.9, 0.95, 1.1, 1.3, 1.3, 1.2, 1.0, 0.9, 0.85,
                  0.85, 0.85, 0.85, 0.9, 1.0, 1.2, 1.25, 1.25, 1.2, 1.1, 1.0, 0.95]
EV_UNGESTEUERT = [0.4, 0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6,
                  0.6, 0.6, 0.7, 0.8, 1.0, 1.8, 2.4, 2.6, 2.2, 1.6, 1.0, 0.6]   # Laden nach Ankunft am Abend
EV_GESTEUERT = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.4, 0.8, 1.6, 2.4, 2.6,
                2.6, 2.4, 1.6, 0.8, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]     # Laden in die Mittagsstunden verschoben
NW_LAGS_STUNDE = 24                       # Newey-West-Verzögerungen für Stundendaten (ein Tag)
NW_LAGS_MONAT = 3                         # Newey-West-Verzögerungen für Monatsdaten (Faustregel 4·(n/100)^(2/9))

# C3 · Dunkelflauten-Definition
DUNKEL_SCHWELLEN = [0.10, 0.15]           # Kapazitätsfaktor Wind+PV
DUNKEL_MINDESTDAUER_H = [24, 48]
BATTERIEBESTAND_GWH = 25.0                 # knapp 25 GWh Ende 2025 (Fraunhofer ISE, Presseinformation Januar 2026)

# C4 · Batterie
BATT_LEISTUNG_MW = 10.0
BATT_DEGRADATION_EUR_MWH = 5.0            # Verschleißkosten je MWh Entladung
BATT_CAPEX_EUR_KWH = [160.0, 250.0, 350.0, 450.0]  # 160 = europ. Systempreis 2025 (BloombergNEF, 177 USD/kWh); 250–450 konservative Spanne
#   Systempreis schlüsselfertig, ohne projektspezifische Kosten (Netzanschluss, Genehmigung, Finanzierung).
BATT_OPEX_ANTEIL = 0.015                  # OPEX pro Jahr in % des CAPEX
BATT_LEBENSDAUER = 15
BATT_ZINS = 0.07                          # risikoadjustierte Kapitalkosten Großspeicher
BATT_MAX_ZYKLEN = 6000
BATT_ERLOESRUECKGANG = 0.03               # Variante: −3 % Erlös pro Jahr (Kannibalisierung)
BATT_VERSCHLEISS_SENS = [0.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0]  # €/MWh, Sensitivität der Einsatzoptimierung
BOOTSTRAP_N = 2000                        # Wiederholungen Bootstrap C6
TOP_N_VARIANTEN = [50, 100, 200, 500]     # Robustheit C2

# D3 · HPC-Standort (alle Werte Annahmen; Basis = Deck Folie 47, Deckungsbeitrag 0,20 €/kWh)
HPC = dict(ladepunkte=4, leistung_kw=300, kwh_je_ladung=35.0, ladungen_tag=150,
           capex=1_400_000.0, opex=80_000.0, jahre=10, zins=0.08,
           verkaufspreis=0.48, beschaffung=0.18, netz_arbeitspreis=0.07, abwicklung=0.03,
           leistungspreis_kw=100.0, gleichzeitigkeit=1.0,
           hochlauf=(0.50, 0.75, 1.00),
           batterie_kappung_kw=600.0, batterie_kwh=300.0, batterie_capex_kwh=450.0)

# D4 · Heimspeicher (Basis = Deck Folie 52; Einspeisevergütung 7,94 ct/kWh, Teileinspeisung bis 10 kWp;
#      Heimspeicherpreise: Verbraucherzentrale 300–700 €/kWh inkl. Installation, Stand 07/2026)
HEIM = dict(ladung_kwh=2500.0, eta=0.90, jahre=12, zins=0.04, strompreis=0.34,
            einspeiseverguetung=0.0794, alterung=0.02, preispfade=(0.0, 0.02, 0.04),
            preise=(4000, 5000, 6000, 7000, 8000))

# D5 · Modellkommune (Deck Folie 37)
KOMMUNE = dict(haushalte=10_000, verbrauch_kwh=3500.0, pv_mwp=20.0, speicher_mwh=5.0,
               ladepunkte=50, ladeleistung_kw=150.0)

QUELLE_SMARD = "Quelle: Bundesnetzagentur | SMARD.de · eigenes Modell THE PATTERN MIND"
CI = sa.CI
JAHRES_PALETTE = ["#9AA3AE", "#6B7A90", "#1A8C9E", "#3A78B5", "#13294B"]

_PLAUSI: list[str] = []


# ===========================================================================
# Hilfsfunktionen
# ===========================================================================
def de(x: float, nd: int = 1) -> str:
    return sa.de(x, nd)


def rahmen(fig, titel: str, untertitel: str, demo: bool, quelle: str = QUELLE_SMARD) -> None:
    fig.patch.set_facecolor(CI["hell"])
    fig.text(0.06, 0.955, titel, fontsize=15, weight="bold", color=CI["navy"], va="top")
    fig.text(0.06, 0.900, untertitel, fontsize=10, color=CI["grau"], va="top")
    fig.text(0.06, 0.022, quelle, fontsize=8, color=CI["grau"])
    if demo:
        fig.text(0.5, 0.5, "SYNTHETISCHE DEMODATEN – NICHT ZITIEREN", rotation=22, fontsize=30,
                 color=CI["rot"], alpha=0.2, ha="center", va="center", weight="bold")


def speichern(fig, pfad: Path) -> Path:
    fig.savefig(pfad, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    return pfad


def methodik(ziel: Path, name: str, zeilen: list[str]) -> None:
    (ziel / f"{name}_methodik.txt").write_text("\n".join(zeilen) + "\n", encoding="utf-8")


def plausi(name: str, ok: bool, detail: str = "") -> None:
    _PLAUSI.append(f"[{'OK ' if ok else 'FEHLER'}] {name}{' – ' + detail if detail else ''}")


def farben_fuer(n: int) -> list:
    if n <= len(JAHRES_PALETTE):
        return JAHRES_PALETTE[-n:]
    return list(plt.cm.Blues(np.linspace(0.35, 1, n)))


def annuitaet(i: float, t: int) -> float:
    return i / (1 - (1 + i) ** (-t))


def barwertfaktor(i: float, t: int) -> float:
    return (1 - (1 + i) ** (-t)) / i


def vorlage(pfad: Path, inhalt: str) -> bool:
    """Legt eine Vorlage an, falls sie fehlt. Rückgabe True, wenn die Datei vorhanden war."""
    if pfad.exists():
        return True
    pfad.parent.mkdir(parents=True, exist_ok=True)
    pfad.write_text(inhalt, encoding="utf-8")
    print(f"  Vorlage angelegt: {pfad} – bitte füllen und erneut starten.")
    return False


def lies_csv_de(pfad: Path) -> pd.DataFrame:
    return pd.read_csv(pfad, sep=";", decimal=",", encoding="utf-8-sig")


# ===========================================================================
# Daten
# ===========================================================================
def _wachstum(jahr: int, a2015: float) -> float:
    """Linearer Demo-Faktor: a2015 im Jahr 2015, 1,0 im Jahr 2025."""
    return a2015 + (1 - a2015) * (jahr - 2015) / 10


def lade_daten(quelle: str, jahre: list[int], csv: list[str], cache: Path) -> pd.DataFrame:
    if quelle == "api":
        reihen = {}
        for name in sa.FILTER:
            print(f"  lade {name} ({min(jahre)}–{max(jahre)}) …")
            reihen[name] = sa.lade_reihe_api(name, jahre, "hour", cache)
    elif quelle == "csv":
        reihen = sa.lade_csv(csv)
    else:
        reihen = sa.demo_daten(jahre)
        jahr = reihen["pv"].index.year
        for k, a in (("pv", 0.40), ("wind_on", 0.60), ("wind_off", 0.25), ("netzlast", 1.08)):
            faktor = np.array([_wachstum(j, a) for j in jahr])
            reihen[k] = reihen[k] * faktor
        krise = (jahr == 2022)
        reihen["preis"] = reihen["preis"].where(~krise, reihen["preis"] * 2.5)
    df = sa.auf_stundenbasis(reihen)
    start = pd.Timestamp(f"{min(jahre)}-01-01", tz=sa.TZ)
    ende = pd.Timestamp(f"{max(jahre) + 1}-01-01", tz=sa.TZ)
    df = df[(df.index >= start) & (df.index < ende)].copy()
    df["residuallast"] = sa.residuallast(df)
    return df


def lade_kapazitaet(quelle: str, df: pd.DataFrame, eingaben: Path) -> pd.DataFrame | None:
    """Installierte Leistung in MW je Stunde (pv, wind_on, wind_off), linear interpoliert."""
    jahre = sorted(set(df.index.year))
    if quelle == "demo":
        werte = {j: {"pv_gw": 70 * _wachstum(j, 0.40), "wind_on_gw": 56 * _wachstum(j, 0.60),
                     "wind_off_gw": 14 * _wachstum(j, 0.25)} for j in range(min(jahre) - 1, max(jahre) + 1)}
        tab = pd.DataFrame(werte).T
    else:
        pfad = eingaben / "installierte_leistung.csv"
        kopf = "jahr;pv_gw;wind_on_gw;wind_off_gw;quelle\n"
        zeilen = "".join(f"{j};;;;\n" for j in range(min(jahre) - 1, max(jahre) + 1))
        if not vorlage(pfad, kopf + zeilen):
            return None
        roh = lies_csv_de(pfad).set_index("jahr")
        tab = roh[["pv_gw", "wind_on_gw", "wind_off_gw"]].apply(pd.to_numeric, errors="coerce")
        if tab.loc[[j for j in jahre if j in tab.index]].isna().any().any():
            print("  installierte_leistung.csv unvollständig – Module mit Kapazitätsbezug übersprungen.")
            return None
    stuetz = pd.DataFrame({
        "pv": tab["pv_gw"].values * 1000, "wind_on": tab["wind_on_gw"].values * 1000,
        "wind_off": tab["wind_off_gw"].values * 1000},
        index=pd.DatetimeIndex([pd.Timestamp(f"{j}-12-31 23:00", tz=sa.TZ) for j in tab.index]))
    kombi = stuetz.reindex(stuetz.index.union(df.index)).sort_index()
    kombi = kombi.interpolate(method="time").bfill().ffill()
    return kombi.reindex(df.index)


# ===========================================================================
# C1 · Profil-Skalierung 2030
# ===========================================================================
def speicher_fahren(r: np.ndarray, e_mwh: float, p_mw: float, eta: float = ETA_RUNDLAUF):
    """Einfache Fahrweise: laden bei r<0, entladen bei r>0. Rückgabe: Residuallast nach Speicher."""
    if e_mwh <= 0:
        return r.copy()
    eta_c = eta_d = np.sqrt(eta)
    soc, aus = 0.0, np.empty_like(r)
    for t, x in enumerate(r):
        if x < 0:
            c = min(p_mw, -x, (e_mwh - soc) / eta_c)
            soc += c * eta_c
            aus[t] = x + c
        else:
            d = min(p_mw, x, soc * eta_d)
            soc -= d / eta_d
            aus[t] = x - d
    return aus


def spitze_mit_speicher(r: np.ndarray, e_mwh: float, p_mw: float, eta: float = ETA_RUNDLAUF) -> float:
    """Niedrigste erreichbare Spitze θ bei spitzenorientierter Fahrweise (Speicher zu Jahresbeginn voll):
    entladen nur oberhalb θ, laden aus freier Kapazität unterhalb θ. Binäre Suche über θ."""
    if e_mwh <= 0:
        return float(r.max())
    eta_c = eta_d = np.sqrt(eta)

    def machbar(theta: float) -> bool:
        soc = e_mwh
        for x in r:
            if x > theta:
                bedarf = x - theta
                if bedarf > p_mw + 1e-9 or bedarf > soc * eta_d + 1e-9:
                    return False
                soc -= bedarf / eta_d
            else:
                soc = min(e_mwh, soc + min(p_mw, theta - x) * eta_c)
        return True

    lo, hi = float(r.max()) - p_mw, float(r.max())
    for _ in range(22):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if machbar(mid) else (mid, hi)
    return hi


def flex_last(r0: pd.Series, zusatz: pd.Series, grenze: pd.Series) -> pd.Series:
    """Verteilt die Tagesenergie 'zusatz' auf die Stunden mit niedrigster Residuallast (gierig, mit Obergrenze)."""
    verteilt = pd.Series(0.0, index=r0.index)
    for _, idx in r0.groupby(r0.index.date).groups.items():
        rest = zusatz.loc[idx].sum()
        for t in r0.loc[idx].sort_values().index:
            if rest <= 0:
                break
            menge = min(grenze.loc[t], rest)
            verteilt.loc[t] = menge
            rest -= menge
    return verteilt


def kennzahlen_residual(r: np.ndarray) -> dict:
    s = np.sort(r)[::-1]
    return {"Überschussstunden": int((r < 0).sum()),
            "Überschuss [TWh]": float(-r[r < 0].sum() / 1e6),
            "Defizit [TWh]": float(r[r > 0].sum() / 1e6),
            "Residuallast max [GW]": float(s[0] / 1000),
            "Top-100-Mittel [GW]": float(s[:100].mean() / 1000)}


def zusatzlast(index: pd.DatetimeIndex, energie_twh: float, monatsanteil, tagesprofil) -> pd.Series:
    """Stündliche Zusatzlast in MW, deren Jahressumme genau energie_twh ergibt."""
    monat = np.asarray(index.month) - 1
    stunden_je_monat = pd.Series(1, index=index).groupby(index.month).transform("sum").to_numpy()
    w = np.asarray(monatsanteil)[monat] / stunden_je_monat * np.asarray(tagesprofil)[np.asarray(index.hour)]
    return pd.Series(energie_twh * 1e6 * w / w.sum(), index=index)


def modell_c1_sektor(df, kap, jahr, ziel, demo):
    """Sensitivität: Wie verändert die zeitliche Form der zusätzlichen Last 2030 Überschuss und Spitze?"""
    d, k = df.loc[str(jahr)].dropna(subset=["netzlast", "pv", "wind_on", "wind_off"]), kap.loc[str(jahr)].reindex(df.loc[str(jahr)].dropna(subset=["netzlast"]).index)
    k = kap.loc[str(jahr)].reindex(d.index)
    erz = sum((d[t] / k[t]).clip(0, 1) * ZIEL_2030_GW[t] * 1000 for t in ("pv", "wind_on", "wind_off"))
    e_wp, e_ev = SEKTOR_2030_TWH["waermepumpen"], SEKTOR_2030_TWH["e_mobilitaet"]
    flach = [1 / 12] * 12
    gleich = d["netzlast"] * (e_wp + e_ev) * 1e6 / d["netzlast"].sum()
    wp = zusatzlast(d.index, e_wp, WP_MONATSANTEIL, WP_TAGESPROFIL)
    ev_u = zusatzlast(d.index, e_ev, flach, EV_UNGESTEUERT)
    ev_g = zusatzlast(d.index, e_ev, flach, EV_GESTEUERT)
    plausi("C1 Sektorkopplung: Zusatzenergie exakt", abs((wp + ev_u).sum() / 1e6 - (e_wp + e_ev)) < 1e-6,
           f"{(wp + ev_u).sum() / 1e6:.3f} TWh")
    varianten = {
        "S0 Heutige Last": d["netzlast"],
        "S1 +45 TWh gleichmäßig": d["netzlast"] + gleich,
        "S2 Wärmepumpen + Laden ungesteuert": d["netzlast"] + wp + ev_u,
        "S3 Wärmepumpen + Laden gesteuert": d["netzlast"] + wp + ev_g,
        "S4 wie S2, Energie × 1,5": d["netzlast"] + 1.5 * (wp + ev_u),
    }
    zeilen = []
    for name, last in varianten.items():
        r = (last - erz).to_numpy()
        zeilen.append({"Variante": name, "Zusatzenergie [TWh]": (last.sum() - d["netzlast"].sum()) / 1e6, **kennzahlen_residual(r)})
    tab = pd.DataFrame(zeilen).set_index("Variante")
    sa.csv_de(tab, ziel / "c1_sektorkopplung.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.33, 7.5))
    farben = [CI["grau"], CI["blau"], CI["rot"], CI["teal"], CI["navy"]]
    kurz = ["S0", "S1", "S2", "S3", "S4"]
    ax1.bar(kurz, tab["Residuallast max [GW]"], color=farben)
    ax2.bar(kurz, tab["Überschuss [TWh]"], color=farben)
    for ax, sp, nd in ((ax1, "Residuallast max [GW]", 1), (ax2, "Überschuss [TWh]", 1)):
        for i, v in enumerate(tab[sp]):
            ax.text(i, v * 1.01, de(v, nd), ha="center", va="bottom", fontsize=9, color=CI["navy"])
        sa._stil(ax)
    ax1.set_title("Spitzenresiduallast 2030 [GW]", loc="left", fontsize=11, color=CI["navy"], weight="bold")
    ax2.set_title("Überschussenergie 2030 [TWh]", loc="left", fontsize=11, color=CI["navy"], weight="bold")
    ax1.set_ylim(0, tab["Residuallast max [GW]"].max() * 1.15)
    ax2.set_ylim(0, tab["Überschuss [TWh]"].max() * 1.15)
    s1, s2, s3 = (tab.iloc[i]["Residuallast max [GW]"] for i in (1, 2, 3))
    rahmen(fig, f"Form der Zusatzlast entscheidet: Spitze {de(s2)} GW ungesteuert, {de(s3)} GW gesteuert "
                f"(gleichmäßig {de(s1)} GW)",
           f"S1–S3 je +{de(e_wp + e_ev, 0)} TWh (Wärmepumpen {de(e_wp, 0)}, E-Mobilität {de(e_ev, 0)}) · S4 = S2 × 1,5 · "
           f"Wetterjahr {jahr}, Ausbauziele 2030 · stilisierte Lastprofile (Annahmen)", demo)
    fig.text(0.07, 0.04, "S0 heutige Last · S1 gleichmäßig · S2 Wärmepumpen + Laden ungesteuert · "
                         "S3 Wärmepumpen + Laden gesteuert · S4 wie S2 mit 1,5-facher Energie", fontsize=9, color=CI["grau"])
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.07, right=0.98, wspace=0.22)
    speichern(fig, ziel / "31_sektorkopplung_2030.png")
    methodik(ziel, "c1s", [
        "C1-Sensitivität · Zeitliche Form der Zusatzlast durch Sektorkopplung",
        f"Zusatzenergie 2030: Wärmepumpen {e_wp} TWh, E-Mobilität {e_ev} TWh (Annahmen: ~6 Mio. WP × ~4 MWh; ~8 Mio. E-Pkw × ~2,5 MWh).",
        "S1 verteilt die Energie proportional zur heutigen Netzlast; S2/S3 nutzen stilisierte Profile:",
        "Wärmepumpen mit Monatsanteilen nach Heizbedarf (Dez–Feb 49 %) und Morgen-/Abendspitze;",
        "E-Mobilität ungesteuert mit Abendspitze 17–21 Uhr, gesteuert mit Laden in den Mittagsstunden.",
        "Temperaturabhängigkeit innerhalb eines Monats (Kältewellen) ist nicht abgebildet; die Spitze ist damit eher unterschätzt.",
        "Ergebnis ist eine Sensitivität, keine Prognose."])


def modell_c1(df, kap, basisjahre, ziel, demo):
    if kap is None:
        print("  C1 übersprungen: installierte Leistung fehlt.")
        return None
    zeilen, residuen = [], {}
    for jahr in basisjahre:
        d, k = df.loc[str(jahr)].dropna(subset=["netzlast", "pv", "wind_on", "wind_off"]), kap.loc[str(jahr)]
        k = k.reindex(d.index)
        cf = {t: (d[t] / k[t]).clip(0, 1) for t in ("pv", "wind_on", "wind_off")}
        # Plausibilität: Kapazitätsfaktor × installierte Leistung reproduziert die Ist-Erzeugung
        rep = d["netzlast"] - sum(cf[t] * k[t] for t in cf)
        abw = float((rep - d["residuallast"]).abs().max() / d["netzlast"].mean())
        plausi(f"C1 Reproduktion Ist-Residuallast {jahr}", abw < 0.001, f"max. Abweichung {abw:.2e}")
        erz = sum(cf[t] * ZIEL_2030_GW[t] * 1000 for t in cf)
        for var, zuwachs in LASTVARIANTEN.items():
            if var == "V3":
                fest = d["netzlast"] * (1 + zuwachs / 2)
                r0 = fest - erz
                zusatz = d["netzlast"] * zuwachs / 2
                grenze = 2 * d["netzlast"].groupby(d.index.date).transform("mean") * zuwachs
                last = fest + flex_last(r0, zusatz, grenze)
                plausi(f"C1 V3 Tagesenergie {jahr}",
                       abs(last.sum() - (d['netzlast'] * (1 + zuwachs)).sum()) / last.sum() < 1e-6)
            else:
                last = d["netzlast"] * (1 + zuwachs)
            r = (last - erz).to_numpy()
            residuen[(jahr, var)] = r
            for e in SPEICHER_GWH:
                rs = speicher_fahren(r, e * 1000, e * 1000 / 4)
                zeilen.append({"Wetterjahr": jahr, "Lastvariante": var, "Speicher [GWh]": e,
                               **kennzahlen_residual(rs),
                               "Spitze spitzenorientiert [GW]": spitze_mit_speicher(r, e * 1000, e * 1000 / 4) / 1000})
    tab = pd.DataFrame(zeilen)
    sa.csv_de(tab.set_index(["Wetterjahr", "Lastvariante", "Speicher [GWh]"]), ziel / "c1_kennzahlen.csv")

    # Grafik 10: Dauerlinien
    bj = basisjahre[-1]
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ist = np.sort(df.loc[str(bj), "residuallast"].dropna().to_numpy())[::-1] / 1000
    ax.plot(np.arange(1, len(ist) + 1), ist, color=CI["grau"], lw=2, label=f"Ist {bj}")
    farben = {"V0": CI["navy"], "V1": CI["blau"], "V2": CI["teal"], "V3": CI["gold"]}
    namen = {"V0": "2030 · Last unverändert", "V1": "2030 · Last +10 %",
             "V2": "2030 · Last +20 %", "V3": "2030 · Last +20 %, Hälfte flexibel"}
    for var in LASTVARIANTEN:
        s = np.sort(residuen[(bj, var)])[::-1] / 1000
        ax.plot(np.arange(1, len(s) + 1), s, color=farben[var], lw=1.8, label=namen[var])
    s0 = np.sort(residuen[(bj, "V0")])[::-1] / 1000
    ax.fill_between(np.arange(1, len(s0) + 1), s0, 0, where=s0 < 0, color=CI["rot"], alpha=0.15)
    ax.axhline(0, color=CI["navy"], lw=0.9)
    sa._stil(ax)
    ax.set_xlim(0, len(s0))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: de(v, 0)))
    ax.set_xlabel("Stunden des Jahres, absteigend sortiert", color=CI["navy"])
    ax.set_ylabel("Residuallast [GW]", color=CI["navy"])
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    v0 = tab[(tab["Wetterjahr"] == bj) & (tab["Lastvariante"] == "V0")].set_index("Speicher [GWh]")
    h0, twh0 = v0.loc[0, "Überschussstunden"], v0.loc[0, "Überschuss [TWh]"]
    rahmen(fig, f"2030: {de(h0, 0)} Überschussstunden und {de(twh0)} TWh Überschuss bei heutiger Last",
           f"Profil-Skalierung Wetterjahr {bj} auf PV {de(ZIEL_2030_GW['pv'], 0)} GW, Wind an Land "
           f"{de(ZIEL_2030_GW['wind_on'], 0)} GW, Wind auf See {de(ZIEL_2030_GW['wind_off'], 0)} GW · "
           "ohne Netz, Export, Abregelung", demo)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.07, right=0.97)
    speichern(fig, ziel / "10_szenario_2030_dauerlinien.png")

    # Grafik 11: Speicherwirkung (Linie = letztes Wetterjahr, Band = Spanne über Wetterjahre)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.33, 7.5))
    for var in LASTVARIANTEN:
        t = tab[tab["Lastvariante"] == var]
        g = t.groupby("Speicher [GWh]")
        for ax, sp in ((ax1, "Überschuss [TWh]"), (ax2, "Spitze spitzenorientiert [GW]")):
            ax.fill_between(g[sp].min().index, g[sp].min(), g[sp].max(), color=farben[var], alpha=0.12)
            haupt = t[t["Wetterjahr"] == bj].set_index("Speicher [GWh]")[sp]
            ax.plot(haupt.index, haupt.values, color=farben[var], marker="o", lw=2, label=namen[var])
    ax1.set_title("Überschussorientierte Fahrweise", loc="left", fontsize=11, color=CI["navy"], weight="bold")
    ax2.set_title("Spitzenorientierte Fahrweise", loc="left", fontsize=11, color=CI["navy"], weight="bold")
    for ax, yl in ((ax1, "Verbleibender Überschuss [TWh]"), (ax2, "Erreichbare Spitzen-Residuallast [GW]")):
        sa._stil(ax)
        ax.set_xlabel("Speicherenergie [GWh] (Leistung = Energie / 4 h)", color=CI["navy"])
        ax.set_ylabel(yl, color=CI["navy"])
        ax.set_xticks(SPEICHER_GWH)
    ax1.legend(frameon=False, fontsize=8)
    e_ref = 50 if 50 in v0.index else SPEICHER_GWH[-1]
    anteil = 100 * (1 - v0.loc[e_ref, "Überschuss [TWh]"] / twh0) if twh0 > 0 else 0
    spitze_red = v0.loc[0, "Spitze spitzenorientiert [GW]"] - v0.loc[e_ref, "Spitze spitzenorientiert [GW]"]
    rahmen(fig, f"{e_ref} GWh Speicher nehmen {de(anteil, 0)} % des Überschusses auf oder senken die Spitze "
                f"um {de(spitze_red)} GW",
           f"Variante V0, Wetterjahr {bj}; Band = Spanne der Wetterjahre {', '.join(map(str, basisjahre))} · "
           f"zwei Fahrweisen, nicht gleichzeitig · Leistung = Energie/4 h, η = {de(100 * ETA_RUNDLAUF, 0)} %", demo)
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.07, right=0.98, wspace=0.22)
    speichern(fig, ziel / "11_szenario_2030_speicherwirkung.png")

    modell_c1_sektor(df, kap, bj, ziel, demo)

    methodik(ziel, "c1", [
        "C1 · Profil-Skalierung 2030",
        "Stündlicher Kapazitätsfaktor je Technologie = SMARD-Erzeugung / installierte Leistung (linear interpoliert).",
        f"Erzeugung 2030 = Kapazitätsfaktor × Zielleistung (PV {ZIEL_2030_GW['pv']} GW, Wind an Land "
        f"{ZIEL_2030_GW['wind_on']} GW, Wind auf See {ZIEL_2030_GW['wind_off']} GW).",
        "Last: V0 unverändert, V1 +10 %, V2 +20 %, V3 +20 % mit der Hälfte des Zuwachses tagesintern in Stunden",
        "niedrigster Residuallast verschoben (Obergrenze je Stunde = doppelter gleichmäßiger Zuwachs).",
        f"Speicher, überschussorientiert: laden bei negativer, entladen bei positiver Residuallast; Leistung = Energie/4 h; η = {ETA_RUNDLAUF}.",
        "Speicher, spitzenorientiert: niedrigste Schwelle θ, über der der Speicher (zu Jahresbeginn voll) jede Stunde deckt;",
        "Nachladung aus freier Kapazität unterhalb θ (binäre Suche). Beide Fahrweisen schließen sich gegenseitig aus.",
        "Grenzen: kein Netz, kein europäischer Austausch, keine Abregelungslogik, Wetterjahre der Vergangenheit.",
        "Das Modell zeigt Größenordnungen, keine Prognose."])
    return residuen.get((bj, "V0"))


# ===========================================================================
# C2 · Spitzenzerlegung und Kapazitätskredit
# ===========================================================================
def modell_c2(df, kap, jahre, ziel, demo):
    zeilen = []
    for jahr in jahre:
        d = df.loc[str(jahr)].dropna(subset=["netzlast", "pv", "wind_on", "wind_off"])
        top = d.nlargest(100, "residuallast")
        spitze = d.loc[d["residuallast"].idxmax()]
        z = {"Jahr": jahr,
             "Top-100 Residuallast [GW]": top["residuallast"].mean() / 1000,
             "Top-100 Netzlast [GW]": top["netzlast"].mean() / 1000,
             "Top-100 Wind an Land [GW]": top["wind_on"].mean() / 1000,
             "Top-100 Wind auf See [GW]": top["wind_off"].mean() / 1000,
             "Top-100 PV [GW]": top["pv"].mean() / 1000,
             "Spitze Residuallast [GW]": spitze["residuallast"] / 1000,
             "Spitze Netzlast [GW]": spitze["netzlast"] / 1000,
             "Spitze Wind+PV [GW]": (spitze["wind_on"] + spitze["wind_off"] + spitze["pv"]) / 1000,
             "Spitze Zeitpunkt": d["residuallast"].idxmax().strftime("%d.%m.%Y %H:%M"),
             "Netzlast Jahressumme [TWh]": d["netzlast"].sum() / 1e6,
             "Wind+PV Jahressumme [TWh]": (d["wind_on"] + d["wind_off"] + d["pv"]).sum() / 1e6,
             "Top-100 Anteil Dez–Feb [%]": 100 * top.index.month.isin([12, 1, 2]).mean(),
             "Top-100 Anteil 16–21 Uhr [%]": 100 * ((top.index.hour >= 16) & (top.index.hour <= 21)).mean()}
        if kap is not None:
            k = kap.reindex(d.index)
            for t, n in (("wind_on", "Wind an Land"), ("wind_off", "Wind auf See"), ("pv", "PV")):
                z[f"CF Jahr {n} [%]"] = 100 * d[t].sum() / k[t].sum()
                z[f"CF Top-100 {n} [%]"] = 100 * top[t].sum() / k.loc[top.index, t].sum()
        zeilen.append(z)
    tab = pd.DataFrame(zeilen).set_index("Jahr")
    sa.csv_de(tab, ziel / "c2_kennzahlen.csv")

    a, b = tab.iloc[0], tab.iloc[-1]
    d_r = b["Top-100 Residuallast [GW]"] - a["Top-100 Residuallast [GW]"]
    beitr = {"Netzlast": b["Top-100 Netzlast [GW]"] - a["Top-100 Netzlast [GW]"],
             "Wind an Land": -(b["Top-100 Wind an Land [GW]"] - a["Top-100 Wind an Land [GW]"]),
             "Wind auf See": -(b["Top-100 Wind auf See [GW]"] - a["Top-100 Wind auf See [GW]"]),
             "PV": -(b["Top-100 PV [GW]"] - a["Top-100 PV [GW]"])}
    plausi("C2 Zerlegung summiert sich exakt", abs(sum(beitr.values()) - d_r) < 1e-9,
           f"ΔR = {d_r:.4f} GW, Summe = {sum(beitr.values()):.4f} GW")
    erz = beitr["Wind an Land"] + beitr["Wind auf See"] + beitr["PV"]
    anteil_last = abs(beitr["Netzlast"]) / (abs(beitr["Netzlast"]) + abs(erz)) if (beitr["Netzlast"] or erz) else 0
    wachstum = 100 * (b["Wind+PV Jahressumme [TWh]"] / a["Wind+PV Jahressumme [TWh]"] - 1)
    urteil = ("Rückgang überwiegend lastgetrieben" if anteil_last > 0.5 else "Rückgang überwiegend erzeugungsgetrieben")
    if d_r >= 0:
        urteil = "kein Rückgang der Top-100-Residuallast"
    text = [f"C2 · Spitzenzerlegung {a.name} → {b.name} (Mittel der 100 Stunden mit höchster Residuallast)",
            f"ΔResiduallast = {d_r:+.2f} GW",
            *[f"  Beitrag {k:<13} {v:+.2f} GW" for k, v in beitr.items()],
            f"Anteil Netzlast am Betrag der Veränderung: {100 * anteil_last:.0f} %",
            f"Wachstum Wind+PV-Jahreserzeugung: {wachstum:+.0f} %",
            f"Urteil H2: {urteil}",
            "Hinweis: SMARD-Netzlast auf Definitions- oder Abdeckungsänderungen zwischen den Jahren prüfen",
            "(Jahressummen in c2_kennzahlen.csv)."]
    (ziel / "c2_ergebnis.txt").write_text("\n".join(text) + "\n", encoding="utf-8")

    # Grafik 12: Wasserfall
    fig, ax = plt.subplots(figsize=(12, 6.75))
    stufen = [(f"Top-100\n{a.name}", a["Top-100 Residuallast [GW]"], "summe")]
    stufen += [(f"Δ {k}", v, "delta") for k, v in beitr.items()]
    stufen += [(f"Top-100\n{b.name}", b["Top-100 Residuallast [GW]"], "summe")]
    stand = 0.0
    for i, (name, wert, art) in enumerate(stufen):
        if art == "summe":
            ax.bar(i, wert, color=CI["navy"], width=0.6)
            ax.text(i, wert + 0.5, f"{de(wert)} GW", ha="center", fontsize=10, color=CI["navy"], weight="bold")
            stand = wert
        else:
            unten = stand + min(wert, 0)
            ax.bar(i, abs(wert), bottom=unten, width=0.6, color=CI["rot"] if wert > 0 else CI["gruen"])
            ax.text(i, stand + max(wert, 0) + 0.5, f"{wert:+.1f}".replace(".", ","), ha="center", fontsize=10,
                    color=CI["navy"])
            stand += wert
    ax.set_xticks(range(len(stufen)), [s[0] for s in stufen], fontsize=9)
    sa._stil(ax)
    ax.set_ylabel("Residuallast, Mittel der Top-100-Stunden [GW]", color=CI["navy"])
    lo = min(a["Top-100 Residuallast [GW]"], b["Top-100 Residuallast [GW]"])
    ax.set_ylim(max(0, lo - 12), max(a["Top-100 Residuallast [GW]"], b["Top-100 Residuallast [GW]"]) + 5)
    rahmen(fig, f"Spitzen-Residuallast {de(d_r)} GW: {de(100 * anteil_last, 0)} % der Veränderung stammen aus der Netzlast",
           f"Zerlegung ΔR = ΔNetzlast − ΔWind an Land − ΔWind auf See − ΔPV in den 100 kritischsten Stunden · "
           f"Wind+PV-Jahreserzeugung {de(wachstum, 0)} %", demo)
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.08, right=0.97)
    speichern(fig, ziel / "12_spitzenzerlegung.png")

    # Grafik 13: Kapazitätskredit
    if kap is not None:
        fig, ax = plt.subplots(figsize=(12, 6.75))
        techs = ["Wind an Land", "Wind auf See", "PV"]
        n = len(tab)
        breite = 0.8 / (2 * n)
        farben_j = farben_fuer(n)
        jahres_pos = []
        for j, (jahr, z) in enumerate(tab.iterrows()):
            for i, t in enumerate(techs):
                x = i - 0.4 + j * 2 * breite
                ax.bar(x + breite / 2, z[f"CF Jahr {t} [%]"], breite, color=CI["linie"], edgecolor=CI["grau"])
                ax.bar(x + 1.5 * breite, z[f"CF Top-100 {t} [%]"], breite, color=farben_j[j])
                jahres_pos.append((x + breite, str(jahr)))
        ymax = ax.get_ylim()[1]
        for x, txt in jahres_pos:
            ax.text(x, -0.03 * ymax, txt, ha="center", va="top", fontsize=7, color=CI["grau"], clip_on=False)
        ax.set_xticks(range(len(techs)), techs, fontsize=10)
        ax.tick_params(axis="x", pad=18, length=0)
        sa._stil(ax)
        ax.set_ylabel("Kapazitätsfaktor [%]", color=CI["navy"])
        ax.bar(0, 0, color=CI["linie"], edgecolor=CI["grau"], label="Jahresmittel")
        ax.bar(0, 0, color=CI["navy"], label="Top-100-Stunden (farbig je Jahr)")
        ax.legend(frameon=False, fontsize=9)
        cf_w = tab.iloc[-1]["CF Top-100 Wind an Land [%]"]
        cf_pv = tab.iloc[-1]["CF Top-100 PV [%]"]
        rahmen(fig, f"In den kritischsten Stunden liefert 1 GW Wind an Land {de(10 * cf_w, 0)} MW, 1 GW PV "
                    f"{de(10 * cf_pv, 0)} MW",
               f"Kapazitätsfaktor im Jahresmittel vs. in den 100 Stunden höchster Residuallast · Näherung des "
               f"Kapazitätskredits, {tab.index[-1]}", demo)
        fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.97)
        speichern(fig, ziel / "13_kapazitaetskredit.png")

    methodik(ziel, "c2", [
        "C2 · Spitzenzerlegung und Kapazitätskredit",
        "Je Jahr die 100 Stunden mit höchster Residuallast; Mittelwerte von Netzlast, Wind an Land, Wind auf See, PV.",
        "Zerlegung: ΔR = ΔNetzlast − ΔWind an Land − ΔWind auf See − ΔPV (exakt, da R = L − W − PV).",
        "Kapazitätskredit-Näherung: Kapazitätsfaktor in den Top-100-Stunden = Erzeugung / installierte Leistung.",
        "Grenzen: deskriptive Zerlegung, keine Kausalaussage; mögliche Änderungen der SMARD-Netzlastdefinition",
        "zwischen den Jahren beeinflussen den Lastbeitrag."])


# ===========================================================================
# C3 · Dunkelflauten und Speicherbedarf
# ===========================================================================
def perioden(maske: pd.Series, mindest: int) -> list[tuple]:
    werte, idx = maske.to_numpy(), maske.index
    erg, start = [], None
    for i, m in enumerate(np.append(werte, False)):
        if m and start is None:
            start = i
        elif not m and start is not None:
            if i - start >= mindest:
                erg.append((idx[start], idx[i - 1], i - start))
            start = None
    return erg


def speicherbedarf(r_mw: np.ndarray, f_mw: float, eta: float = ETA_RUNDLAUF) -> tuple[float, int, int]:
    """Benötigte Speicherenergie [MWh] bei gesicherter Leistung F; Rückgabe (E, Endindex, Dauer h)."""
    d, best, best_t, start, best_dauer, laenge = 0.0, 0.0, 0, 0, 0, 0
    for t, r in enumerate(r_mw):
        if r > f_mw:
            if d == 0:
                start = t
            d += r - f_mw
        else:
            d = max(0.0, d - (f_mw - r) * eta)
        if d > best:
            best, best_t, best_dauer = d, t, t - start + 1
        laenge = t
    return best, best_t, best_dauer


def modell_c3(df, kap, jahre, f_werte, r2030, basisjahr2030, ziel, demo):
    if kap is None:
        print("  C3 übersprungen: installierte Leistung fehlt.")
        return
    alle_perioden, stat = [], []
    for jahr in jahre:
        d = df.loc[str(jahr)].dropna(subset=["pv", "wind_on", "wind_off"])
        k = kap.reindex(d.index)
        cf = (d["pv"] + d["wind_on"] + d["wind_off"]) / (k["pv"] + k["wind_on"] + k["wind_off"])
        for s in DUNKEL_SCHWELLEN:
            for m in DUNKEL_MINDESTDAUER_H:
                ps = perioden(cf < s, m)
                for p in ps:
                    alle_perioden.append({"Jahr": jahr, "Schwelle": s, "Mindestdauer [h]": m,
                                          "Start": p[0].strftime("%d.%m.%Y %H:%M"),
                                          "Ende": p[1].strftime("%d.%m.%Y %H:%M"), "Dauer [h]": p[2],
                                          "Mittl. Kapazitätsfaktor [%]": 100 * cf.loc[p[0]:p[1]].mean()})
                stat.append({"Jahr": jahr, "Schwelle": s, "Mindestdauer [h]": m, "Anzahl": len(ps),
                             "Stunden gesamt": sum(p[2] for p in ps),
                             "Längste [h]": max((p[2] for p in ps), default=0),
                             "Längste Start": max(ps, key=lambda p: p[2])[0].strftime("%d.%m.%Y") if ps else ""})
    st = pd.DataFrame(stat)
    sa.csv_de(pd.DataFrame(alle_perioden), ziel / "c3_dunkelflauten.csv", index=False)

    bedarf = []
    reihen = {str(j): df.loc[str(j), "residuallast"].dropna().to_numpy() for j in jahre}
    if r2030 is not None:
        reihen[f"2030 (Wetter {basisjahr2030})"] = r2030
    for name, r in reihen.items():
        for f in f_werte:
            e, t, dauer = speicherbedarf(r, f * 1000)
            bedarf.append({"Reihe": name, "Gesicherte Leistung [GW]": f, "Speicherbedarf [TWh]": e / 1e6,
                           "Dauer Entladephase [h]": dauer})
        e_hoch, _, _ = speicherbedarf(r, r.max() + 1)
        plausi(f"C3 E(F) = 0 für F > max. Residuallast ({name})", e_hoch == 0)
    bt = pd.DataFrame(bedarf)
    sa.csv_de(bt.set_index(["Reihe", "Gesicherte Leistung [GW]"]), ziel / "c3_speicherbedarf.csv")
    sa.csv_de(st.set_index(["Jahr", "Schwelle", "Mindestdauer [h]"]), ziel / "c3_kennzahlen.csv")

    # Grafik 14
    haupt = st[(st["Schwelle"] == DUNKEL_SCHWELLEN[0]) & (st["Mindestdauer [h]"] == DUNKEL_MINDESTDAUER_H[0])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.33, 7.5))
    ax1.bar(haupt["Jahr"].astype(str), haupt["Längste [h]"], color=CI["navy"])
    ax2.bar(haupt["Jahr"].astype(str), haupt["Anzahl"], color=CI["teal"])
    for ax, yl, t in ((ax1, "Stunden", "Längste Dunkelflaute je Jahr"), (ax2, "Anzahl", "Anzahl Perioden je Jahr")):
        sa._stil(ax)
        ax.set_ylabel(yl, color=CI["navy"])
        ax.set_title(t, loc="left", fontsize=11, color=CI["navy"], weight="bold")
        ax.tick_params(axis="x", rotation=45)
    rek = haupt.loc[haupt["Längste [h]"].idxmax()]
    rahmen(fig, f"Längste Dunkelflaute: {de(rek['Längste [h]'], 0)} Stunden ({int(rek['Jahr'])}, ab {rek['Längste Start']})",
           f"Dunkelflaute = Wind+PV unter {de(100 * DUNKEL_SCHWELLEN[0], 0)} % ihrer installierten Leistung für mindestens "
           f"{DUNKEL_MINDESTDAUER_H[0]} h · weitere Varianten in c3_kennzahlen.csv", demo)
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.07, right=0.98, wspace=0.2)
    speichern(fig, ziel / "14_dunkelflauten.png")

    # Grafik 15
    fig, ax = plt.subplots(figsize=(12, 6.75))
    zeige = [str(jahre[-1])] + ([k for k in reihen if k.startswith("2030")])
    raster = np.arange(min(f_werte), max(f_werte) + 0.01, 1.0)
    log = False
    kurven = {}
    for name in zeige:
        kurven[name] = np.array([speicherbedarf(reihen[name], f * 1000)[0] / 1e6 for f in raster])
    pos = np.concatenate([k[k > 0] for k in kurven.values()]) if kurven else np.array([])
    if pos.size and pos.max() / pos.min() > 50:
        log = True
        ax.set_yscale("log")
    for name, farbe in zip(zeige, (CI["navy"], CI["gold"])):
        k = kurven[name]
        ax.plot(raster, np.where(k > 0, k, np.nan) if log else k, lw=2.2, color=farbe, label=f"Residuallast {name}")
        t = bt[bt["Reihe"] == name]
        ax.plot(t["Gesicherte Leistung [GW]"], t["Speicherbedarf [TWh]"].where(t["Speicherbedarf [TWh]"] > 0) if log
                else t["Speicherbedarf [TWh]"], "o", color=farbe)
        if (k == 0).any():
            f0 = raster[np.argmax(k == 0)]
            ax.annotate(f"0 ab {de(f0, 0)} GW", xy=(f0, ax.get_ylim()[0] if log else 0), xytext=(4, 8),
                        textcoords="offset points", fontsize=8, color=farbe)
    ax.set_xlim(raster[0], raster[-1])
    ax.axhline(BATTERIEBESTAND_GWH / 1000, color=CI["rot"], ls="--", lw=1)
    ax.text(raster[-1], BATTERIEBESTAND_GWH / 1000, f"Batteriebestand {de(BATTERIEBESTAND_GWH, 0)} GWh  ",
            va="bottom", ha="right", fontsize=8, color=CI["rot"])
    sa._stil(ax)
    ax.set_xlabel("Gesicherte Leistung F [GW]", color=CI["navy"])
    ax.set_ylabel("Benötigte Speicherenergie [TWh]", color=CI["navy"])
    ax.legend(frameon=False, fontsize=9)
    f_ref = 50 if 50 in f_werte else f_werte[len(f_werte) // 2]
    e_ref = bt[(bt["Reihe"] == str(jahre[-1])) & (bt["Gesicherte Leistung [GW]"] == f_ref)]["Speicherbedarf [TWh]"].iloc[0]
    rahmen(fig, f"Bei {f_ref} GW gesicherter Leistung: {de(e_ref, 2)} TWh Speicherbedarf "
                f"({de(e_ref * 1000 / BATTERIEBESTAND_GWH, 0)}× Batteriebestand)",
           f"Größtes kumuliertes Defizit über F, Nachladung aus freier gesicherter Leistung (η = {ETA_RUNDLAUF}) · "
           "ohne Importe, Biomasse und Wasser nicht als gesichert gerechnet", demo)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.97)
    speichern(fig, ziel / "15_speicherbedarf_gesicherte_leistung.png")

    methodik(ziel, "c3", [
        "C3 · Dunkelflauten und Speicherbedarf",
        "Kapazitätsfaktor Wind+PV = (Wind an Land + Wind auf See + PV) / installierte Leistung dieser Technologien.",
        f"Dunkelflaute = zusammenhängende Periode mit Kapazitätsfaktor < Schwelle ({DUNKEL_SCHWELLEN}) über "
        f"mindestens {DUNKEL_MINDESTDAUER_H} Stunden.",
        "Speicherbedarf E(F): Defizitzähler D steigt um (R − F), wenn die Residuallast R über der gesicherten Leistung F",
        f"liegt, und sinkt um (F − R) × η sonst (η = {ETA_RUNDLAUF}); E(F) = Maximum von D.",
        "Grenzen: Importe nicht berücksichtigt (Obergrenze); Biomasse/Wasserkraft nicht als gesichert gerechnet;",
        "Speicher- und Kraftwerksstandorte sowie Netzrestriktionen nicht abgebildet."])


# ===========================================================================
# C4 · Batterie-Backtest
# ===========================================================================
def dp_optimum(preise: np.ndarray, stufen: int, p_mw: float, eta: float, verschleiss: float,
               start: int = 0) -> np.ndarray:
    """Dynamische Programmierung: Aktionen je Stunde (−1 laden, 0 halten, +1 entladen).
    Füllstandsstufen 0..stufen, eine Stufe = p_mw MWh gespeichert."""
    eta_c = eta_d = np.sqrt(eta)
    n = len(preise)
    wert = np.zeros(stufen + 1)
    wahl = np.zeros((n, stufen + 1), dtype=np.int8)
    lv = np.arange(stufen + 1)
    for t in range(n - 1, -1, -1):
        p = preise[t]
        halten = wert.copy()
        laden = np.full(stufen + 1, -np.inf)
        laden[:-1] = -p * p_mw / eta_c + wert[1:]
        entladen = np.full(stufen + 1, -np.inf)
        entladen[1:] = (p - verschleiss) * p_mw * eta_d + wert[:-1]
        alle = np.vstack([laden, halten, entladen])
        beste = alle.argmax(axis=0)
        wahl[t] = beste - 1
        wert = alle[beste, lv]
    aktionen, l = np.zeros(n, dtype=np.int8), start
    for t in range(n):
        a = wahl[t, l]
        aktionen[t] = a
        l = l + 1 if a == -1 else (l - 1 if a == 1 else l)
    return aktionen


def abrechnen(aktionen: np.ndarray, preise: np.ndarray, p_mw: float, eta: float, verschleiss: float,
              brutto: bool = False):
    """Erlös je Stunde. netto = nach kalkulatorischem Verschleißansatz (Zielfunktion der Optimierung),
    brutto = tatsächlicher Handelserlös (zahlungswirksam)."""
    eta_c = eta_d = np.sqrt(eta)
    lade = (aktionen == -1) * p_mw / eta_c
    entl = (aktionen == 1) * p_mw * eta_d
    erloes = entl * (preise - (0.0 if brutto else verschleiss)) - lade * preise
    return erloes, entl


def lebensdauer_eff(zyklen_pro_jahr: float) -> int:
    """Lebensdauer begrenzt durch kalendarische Lebensdauer und Zyklengrenze."""
    if zyklen_pro_jahr <= 0:
        return BATT_LEBENSDAUER
    return int(max(1, min(BATT_LEBENSDAUER, np.floor(BATT_MAX_ZYKLEN / zyklen_pro_jahr))))


def break_even_capex(erl_eur_a: float, e_kwh: float, t: int, rueckgang: float = 0.0) -> float:
    """CAPEX [€/kWh], bei dem der Kapitalwert null ist (OPEX als Anteil des CAPEX)."""
    bwf_r = sum((1 - rueckgang) ** (k - 1) / (1 + BATT_ZINS) ** k for k in range(1, t + 1))
    bwf = barwertfaktor(BATT_ZINS, t)
    return erl_eur_a * bwf_r / (1 + BATT_OPEX_ANTEIL * bwf) / e_kwh


def realistisch(preise: pd.Series, stufen: int, p_mw: float, eta: float, verschleiss: float) -> np.ndarray:
    """Fahrplan für Tag t mit den Preisen von Tag t−1 optimieren, mit Preisen von Tag t abrechnen."""
    aktionen = np.zeros(len(preise), dtype=np.int8)
    tage = preise.groupby(preise.index.date).indices
    vor, l = None, 0
    for tag in sorted(tage):
        idx = tage[tag]
        if vor is None:
            vor = preise.iloc[idx].to_numpy()
            continue
        prog = np.resize(vor, len(idx))
        akt = dp_optimum(prog, stufen, p_mw, eta, verschleiss, start=l)
        # Zulässigkeit mit tatsächlichem Füllstand sichern
        for i, a in enumerate(akt):
            if a == -1 and l >= stufen or a == 1 and l <= 0:
                akt[i] = 0
            l = l + 1 if akt[i] == -1 else (l - 1 if akt[i] == 1 else l)
        aktionen[idx] = akt
        vor = preise.iloc[idx].to_numpy()
    return aktionen


def modell_c4(df, jahre, dauern, ziel, demo):
    zeilen, monate = [], []
    for jahr in jahre:
        pr = df.loc[str(jahr), "preis"].dropna()
        if len(pr) < 8000:
            print(f"  C4: {jahr} hat zu wenige Preisstunden – übersprungen.")
            continue
        for h in dauern:
            akt_o = dp_optimum(pr.to_numpy(), h, BATT_LEISTUNG_MW, ETA_RUNDLAUF, BATT_DEGRADATION_EUR_MWH)
            akt_r = realistisch(pr, h, BATT_LEISTUNG_MW, ETA_RUNDLAUF, BATT_DEGRADATION_EUR_MWH)
            for art, akt in (("Obergrenze", akt_o), ("Realistisch", akt_r)):
                er_n, entl = abrechnen(akt, pr.to_numpy(), BATT_LEISTUNG_MW, ETA_RUNDLAUF, BATT_DEGRADATION_EUR_MWH)
                er, _ = abrechnen(akt, pr.to_numpy(), BATT_LEISTUNG_MW, ETA_RUNDLAUF, BATT_DEGRADATION_EUR_MWH, brutto=True)
                e_mwh = h * BATT_LEISTUNG_MW
                zyklen = (akt == 1).sum() * BATT_LEISTUNG_MW / e_mwh
                zeilen.append({"Jahr": jahr, "Dauer [h]": h, "Variante": art,
                               "Erlös [€/MW/a]": er.sum() / BATT_LEISTUNG_MW,
                               "Erlös netto Verschleißansatz [€/MW/a]": er_n.sum() / BATT_LEISTUNG_MW,
                               "Zyklen/a": zyklen, "Entladung [MWh/a]": entl.sum()})
                m = pd.Series(er, index=pr.index).groupby(pr.index.month).sum() / BATT_LEISTUNG_MW
                for mon, v in m.items():
                    monate.append({"Jahr": jahr, "Dauer [h]": h, "Variante": art, "Monat": mon, "Erlös [€/MW]": v})
            ro = zeilen[-2]["Erlös netto Verschleißansatz [€/MW/a]"]
            rr = zeilen[-1]["Erlös netto Verschleißansatz [€/MW/a]"]
            plausi(f"C4 realistisch ≤ Obergrenze ({jahr}, {h} h)", rr <= ro + 1e-6)
    if not zeilen:
        return
    # Plausibilität konstante Preise
    akt_k = dp_optimum(np.full(48, 50.0), 2, BATT_LEISTUNG_MW, ETA_RUNDLAUF, BATT_DEGRADATION_EUR_MWH)
    er_k, _ = abrechnen(akt_k, np.full(48, 50.0), BATT_LEISTUNG_MW, ETA_RUNDLAUF, BATT_DEGRADATION_EUR_MWH)
    plausi("C4 konstante Preise → Erlös ≤ 0", er_k.sum() <= 1e-9)

    tab = pd.DataFrame(zeilen)
    anf, bwf = annuitaet(BATT_ZINS, BATT_LEBENSDAUER), barwertfaktor(BATT_ZINS, BATT_LEBENSDAUER)
    bwf_rueck = sum((1 - BATT_ERLOESRUECKGANG) ** (t - 1) / (1 + BATT_ZINS) ** t for t in range(1, BATT_LEBENSDAUER + 1))
    erg = []
    for _, z in tab.iterrows():
        e_kwh = z["Dauer [h]"] * BATT_LEISTUNG_MW * 1000
        erl = z["Erlös [€/MW/a]"] * BATT_LEISTUNG_MW
        z = z.to_dict()
        t_eff = lebensdauer_eff(z["Zyklen/a"])
        bwf_eff = barwertfaktor(BATT_ZINS, t_eff)
        z["Lebensdauer effektiv [a]"] = t_eff
        z["Break-even CAPEX 15 a ohne Zyklengrenze [€/kWh]"] = break_even_capex(erl, e_kwh, BATT_LEBENSDAUER)
        z["Break-even CAPEX [€/kWh]"] = break_even_capex(erl, e_kwh, t_eff)
        z["Break-even CAPEX, −3 %/a [€/kWh]"] = break_even_capex(erl, e_kwh, t_eff, BATT_ERLOESRUECKGANG)
        for c in BATT_CAPEX_EUR_KWH:
            capex = c * e_kwh
            z[f"KW bei {c:.0f} €/kWh [Mio. €]"] = (-capex + (erl - BATT_OPEX_ANTEIL * capex) * bwf_eff) / 1e6
            z[f"LCOS bei {c:.0f} €/kWh [ct/kWh]"] = (
                (capex * annuitaet(BATT_ZINS, t_eff) + BATT_OPEX_ANTEIL * capex) / max(z["Entladung [MWh/a]"] * 1000, 1) * 100)
        z["Zyklen über Lebensdauer"] = z["Zyklen/a"] * t_eff
        erg.append(z)
    erg = pd.DataFrame(erg)
    sa.csv_de(erg.set_index(["Jahr", "Dauer [h]", "Variante"]), ziel / "c4_kennzahlen.csv")
    mt = pd.DataFrame(monate)

    lj = int(erg["Jahr"].max())
    href = 2 if 2 in dauern else dauern[0]
    # Grafik 16
    fig, ax = plt.subplots(figsize=(12, 6.75))
    jahre_ok = sorted(mt["Jahr"].unique())
    breite = 0.8 / len(jahre_ok)
    for j, jahr in enumerate(jahre_ok):
        for art, alpha in (("Obergrenze", 0.35), ("Realistisch", 1.0)):
            t = mt[(mt["Jahr"] == jahr) & (mt["Dauer [h]"] == href) & (mt["Variante"] == art)]
            ax.bar(t["Monat"] + (j - (len(jahre_ok) - 1) / 2) * breite, t["Erlös [€/MW]"] / 1000, breite,
                   color=farben_fuer(len(jahre_ok))[j], alpha=alpha,
                   label=f"{jahr} · {art}")
    ax.set_xticks(range(1, 13), sa.MONATE)
    sa._stil(ax)
    ax.set_ylabel("Arbitrage-Erlös [Tsd. €/MW]", color=CI["navy"])
    ax.legend(frameon=False, fontsize=8, ncol=2)
    zl = erg[(erg["Jahr"] == lj) & (erg["Dauer [h]"] == href)].set_index("Variante")
    rahmen(fig, f"{lj} brachte Day-Ahead-Arbitrage {de(zl.loc['Realistisch', 'Erlös [€/MW/a]'] / 1000, 0)}–"
                f"{de(zl.loc['Obergrenze', 'Erlös [€/MW/a]'] / 1000, 0)} Tsd. €/MW ({href}-h-Speicher)",
           "Obergrenze: vollständige Voraussicht · Realistisch: Fahrplan aus Vortagespreisen · nur Day-Ahead, "
           "ohne Regelleistung und Intraday", demo)
    fig.subplots_adjust(top=0.84, bottom=0.10, left=0.08, right=0.97)
    speichern(fig, ziel / "16_batterie_backtest.png")

    # Grafik 17
    fig, ax = plt.subplots(figsize=(12, 6.75))
    capex_achse = np.linspace(100, 700, 61)
    farbe_h = dict(zip(dauern, [CI["teal"], CI["blau"], CI["navy"], CI["gold"]][:len(dauern)]))
    for h in dauern:
        for art, ls in (("Realistisch", "-"), ("Obergrenze", "--")):
            z = erg[(erg["Jahr"] == lj) & (erg["Dauer [h]"] == h) & (erg["Variante"] == art)].iloc[0]
            e_kwh = h * BATT_LEISTUNG_MW * 1000
            erl = z["Erlös [€/MW/a]"] * BATT_LEISTUNG_MW
            bwf_l = barwertfaktor(BATT_ZINS, int(z["Lebensdauer effektiv [a]"]))
            kw = (-capex_achse * e_kwh + (erl - BATT_OPEX_ANTEIL * capex_achse * e_kwh) * bwf_l) / 1e6
            ax.plot(capex_achse, kw, ls=ls, color=farbe_h[h], lw=2 if ls == "-" else 1.3,
                    label=f"{h} h · {art}")
        be = erg[(erg["Jahr"] == lj) & (erg["Dauer [h]"] == h) & (erg["Variante"] == "Realistisch")][
            "Break-even CAPEX [€/kWh]"].iloc[0]
        if 100 <= be <= 700:
            ax.plot(be, 0, "o", color=farbe_h[h])
    ax.axhline(0, color=CI["navy"], lw=0.9)
    for c in BATT_CAPEX_EUR_KWH:
        ax.axvline(c, color=CI["grau"], ls=":", lw=0.8)
    sa._stil(ax)
    ax.set_xlabel("Investition [€/kWh]", color=CI["navy"])
    ax.set_ylabel(f"Kapitalwert {BATT_LEISTUNG_MW:.0f}-MW-Speicher [Mio. €]", color=CI["navy"])
    ax.legend(frameon=False, fontsize=8, ncol=2)
    be_ref = zl.loc["Realistisch", "Break-even CAPEX [€/kWh]"]
    rahmen(fig, f"Break-even bei {de(be_ref, 0)} €/kWh für den {href}-h-Speicher (Erlöse {lj}, realistisch)",
           f"Lebensdauer = min({BATT_LEBENSDAUER} Jahre; {BATT_MAX_ZYKLEN} Zyklen), Zins {de(100 * BATT_ZINS, 0)} %, OPEX "
           f"{de(100 * BATT_OPEX_ANTEIL, 1)} % p. a., konstante Erlöse · gepunktet: Annahmen zum Batteriepreis", demo)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.97)
    speichern(fig, ziel / "17_batterie_kapitalwert.png")

    # Sensitivität: kalkulatorischer Verschleißpreis steuert Zyklen und Lebensdauer
    pr_l = df.loc[str(lj), "preis"].dropna()
    sens = []
    for vs in BATT_VERSCHLEISS_SENS:
        for art in ("Obergrenze", "Realistisch"):
            akt = (dp_optimum(pr_l.to_numpy(), href, BATT_LEISTUNG_MW, ETA_RUNDLAUF, vs) if art == "Obergrenze"
                   else realistisch(pr_l, href, BATT_LEISTUNG_MW, ETA_RUNDLAUF, vs))
            er, entl = abrechnen(akt, pr_l.to_numpy(), BATT_LEISTUNG_MW, ETA_RUNDLAUF, vs, brutto=True)
            e_kwh = href * BATT_LEISTUNG_MW * 1000
            zyk = (akt == 1).sum() / href
            t_eff = lebensdauer_eff(zyk)
            sens.append({"Verschleißpreis [€/MWh]": vs, "Variante": art, "Zyklen/a": zyk,
                         "Erlös brutto [€/MW/a]": er.sum() / BATT_LEISTUNG_MW, "Lebensdauer effektiv [a]": t_eff,
                         "Break-even CAPEX [€/kWh]": break_even_capex(er.sum(), e_kwh, t_eff)})
    st = pd.DataFrame(sens)
    sa.csv_de(st.set_index(["Verschleißpreis [€/MWh]", "Variante"]), ziel / "c4_verschleiss_sensitivitaet.csv")
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax2 = ax.twinx()
    for art, ls in (("Realistisch", "-"), ("Obergrenze", "--")):
        t = st[st["Variante"] == art]
        ax.plot(t["Verschleißpreis [€/MWh]"], t["Break-even CAPEX [€/kWh]"], ls=ls, marker="o", color=CI["navy"],
                lw=2.2, label=f"Break-even · {art}")
        ax2.plot(t["Verschleißpreis [€/MWh]"], t["Zyklen/a"], ls=ls, marker="s", color=CI["gold"], lw=1.5,
                 label=f"Zyklen/a · {art}")
    sa._stil(ax)
    ax.set_xlabel("Kalkulatorischer Verschleißpreis in der Einsatzoptimierung [€/MWh]", color=CI["navy"])
    ax.set_ylabel("Break-even-Investition [€/kWh]", color=CI["navy"])
    ax2.set_ylabel("Zyklen pro Jahr", color=CI["gold"])
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper right")
    best = st[st["Variante"] == "Realistisch"].sort_values("Break-even CAPEX [€/kWh]").iloc[-1]
    rahmen(fig, f"Bester Verschleißpreis {de(best['Verschleißpreis [€/MWh]'], 0)} €/MWh: Break-even "
                f"{de(best['Break-even CAPEX [€/kWh]'], 0)} €/kWh ({href}-h-Speicher, {lj})",
           f"Höherer Verschleißpreis = weniger Zyklen und längere Lebensdauer, aber weniger Handelserlös · "
           f"Lebensdauer = min({BATT_LEBENSDAUER} a; {BATT_MAX_ZYKLEN} Zyklen)", demo)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.92)
    speichern(fig, ziel / "27_batterie_verschleiss.png")

    methodik(ziel, "c4", [
        "C4 · Batterie-Backtest",
        f"Speicher {BATT_LEISTUNG_MW:.0f} MW, Dauer {dauern} h, Rundlauf-Wirkungsgrad {ETA_RUNDLAUF}, Verschleiß "
        f"{BATT_DEGRADATION_EUR_MWH} €/MWh Entladung.",
        "Obergrenze: dynamische Programmierung über das ganze Jahr mit vollständiger Voraussicht (Stufe = 1 h Leistung).",
        "Realistisch: Fahrplan je Tag mit den Preisen des Vortags optimiert und mit tatsächlichen Preisen abgerechnet.",
        "Ab 01.10.2025 Stundenmittel der Viertelstundenpreise; Viertelstunden-Arbitrage bietet zusätzliche Erlöse.",
        "LCOS = (CAPEX × Annuitätenfaktor + OPEX) / jährlich abgegebene Energie.",
        f"Erlös = zahlungswirksamer Handelserlös (brutto); der Verschleißpreis wirkt nur als Steuergröße der Optimierung.",
        f"Effektive Lebensdauer = min({BATT_LEBENSDAUER} Jahre; {BATT_MAX_ZYKLEN} Zyklen / Zyklen pro Jahr), abgerundet.",
        "Sensitivität: Verschleißpreis 0–50 €/MWh steuert Zyklenzahl, Lebensdauer und Break-even (Grafik 27).",
        f"Kapitalwert: {BATT_LEBENSDAUER} Jahre, Zins {BATT_ZINS}, OPEX {BATT_OPEX_ANTEIL} × CAPEX; Variante −3 %/a Erlös.",
        "Nicht enthalten: Regelleistung, Intraday, Netzentgelte, Preiswirkung des eigenen Einsatzes.",
        "Batteriepreise sind Annahmen und durch aktuelle Angebote zu ersetzen."])


# ===========================================================================
# C5 · Merit-Order-Regression
# ===========================================================================
def ols_inferenz(x: np.ndarray, y: np.ndarray, spalte: int, nw_lags: int) -> dict:
    """OLS mit klassischen und Newey-West-Standardfehlern (Bartlett-Kern); p-Werte zweiseitig,
    Normalverteilungsapproximation. Zeilen müssen chronologisch geordnet sein."""
    koef, *_ = np.linalg.lstsq(x, y, rcond=None)
    res = y - x @ koef
    n, k = x.shape
    dof = max(n - k, 1)
    xtx_inv = np.linalg.pinv(x.T @ x)
    se_ols = float(np.sqrt(res @ res / dof * xtx_inv[spalte, spalte]))
    u = x * res[:, None]
    s = u.T @ u
    for lag in range(1, nw_lags + 1):
        g = u[lag:].T @ u[:-lag]
        s += (1 - lag / (nw_lags + 1)) * (g + g.T)
    kov_nw = xtx_inv @ s @ xtx_inv * n / dof
    se_nw = float(np.sqrt(max(kov_nw[spalte, spalte], 0)))
    b = float(koef[spalte])
    p = lambda t: float(math.erfc(abs(t) / math.sqrt(2)))
    r2 = float(1 - (res @ res) / ((y - y.mean()) @ (y - y.mean())))
    return {"b": b, "SE OLS": se_ols, "t OLS": b / se_ols if se_ols else np.nan, "p OLS": p(b / se_ols) if se_ols else np.nan,
            "SE Newey-West": se_nw, "t Newey-West": b / se_nw if se_nw else np.nan,
            "p Newey-West": p(b / se_nw) if se_nw else np.nan, "R²": r2, "n": int(n), "koef": koef}


def modell_c5(df, jahre, ziel, demo):
    erg, kurven = [], {}
    for jahr in jahre:
        d = df.loc[str(jahr)].dropna(subset=["preis", "residuallast"])
        if len(d) < 8000:
            continue
        r, p = d["residuallast"] / 1000, d["preis"]
        klassen = (r // 5) * 5 + 2.5
        g = p.groupby(klassen)
        med = g.median()[g.size() >= 50]
        kurven[jahr] = med
        m = (r >= 20) & (r <= 50)
        x = np.column_stack([np.ones(m.sum()), r[m]])
        inf = ols_inferenz(x, p[m].to_numpy(), 1, NW_LAGS_STUNDE)
        koef, r2 = inf["koef"], inf["R²"]
        erg.append({"Jahr": jahr, "Steigung [€/MWh je GW]": koef[1], "Achsenabschnitt [€/MWh]": koef[0],
                    "R²": r2, "Median-Preis bei 40 GW [€/MWh]": float(med.get(42.5, np.nan)),
                    "Stunden": len(d), "n (20–50 GW)": inf["n"], "SE OLS": inf["SE OLS"], "p OLS": inf["p OLS"],
                    "SE Newey-West": inf["SE Newey-West"], "t Newey-West": inf["t Newey-West"],
                    "p Newey-West": inf["p Newey-West"]})
    if not erg:
        print("  C5 übersprungen: keine Jahre mit vollständigen Preisdaten.")
        return
    tab = pd.DataFrame(erg).set_index("Jahr")
    sa.csv_de(tab, ziel / "c5_kennzahlen.csv")
    fig, ax = plt.subplots(figsize=(12, 6.75))
    js = list(kurven)
    for jahr, farbe in zip(js, farben_fuer(len(js))):
        hervor = jahr == 2022
        ax.plot(kurven[jahr].index, kurven[jahr].values, marker="o", ms=3,
                color=CI["rot"] if hervor else farbe, lw=2.6 if hervor or jahr == js[-1] else 1.4, label=str(jahr))
    ax.axhline(0, color=CI["navy"], lw=0.9)
    sa._stil(ax)
    ax.set_xlabel("Residuallast [GW] (5-GW-Klassen)", color=CI["navy"])
    ax.set_ylabel("Median Day-Ahead-Preis [€/MWh]", color=CI["navy"])
    ax.legend(frameon=False, fontsize=8, ncol=2)
    lj = js[-1]
    titel = f"Je GW Residuallast steigt der Preis {lj} um {de(tab.loc[lj, 'Steigung [€/MWh je GW]'])} €/MWh"
    if 2022 in tab.index and not np.isnan(tab.loc[2022, "Median-Preis bei 40 GW [€/MWh]"]):
        diff = tab.loc[2022, "Median-Preis bei 40 GW [€/MWh]"] - tab.loc[lj, "Median-Preis bei 40 GW [€/MWh]"]
        titel += f"; 2022 bei 40 GW +{de(diff, 0)} €/MWh"
    rahmen(fig, titel, "Median je 5-GW-Klasse (mind. 50 Stunden); Steigung aus linearer Regression im Bereich "
                       "20–50 GW · deskriptiv, kein Kausalmodell", demo)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.97)
    speichern(fig, ziel / "18_merit_order.png")
    methodik(ziel, "c5", [
        "C5 · Merit-Order-Regression",
        "Stündliche Paare aus Residuallast und Day-Ahead-Preis je Jahr; Median und Klassen à 5 GW (mind. 50 Stunden).",
        "Lineare Regression Preis = a + b × Residuallast im Bereich 20–50 GW; b in €/MWh je GW, R² ausgewiesen.",
        f"Standardfehler klassisch (OLS) und nach Newey-West mit {NW_LAGS_STUNDE} Verzögerungen (Autokorrelation der Stundenwerte);",
        "p-Werte zweiseitig, Normalverteilungsapproximation.",
        "Deskriptiv: Brennstoff- und CO₂-Preise, Must-Run, Export und Viertelstundenhandel wirken mit.",
        "Preise vor 10/2018 (Zone DE-AT-LU) sind über die SMARD-Chart-API nicht verfügbar."])


# ===========================================================================
# C6 · Robuste Kannibalisierung
# ===========================================================================
def t_wert(rho: float, n: int) -> float:
    return rho * np.sqrt((n - 2) / (1 - rho ** 2)) if abs(rho) < 1 else np.inf


def modell_c6(df, jahre, ziel, demo):
    jahr_z, mon_z = [], []
    for jahr in jahre:
        d = df.loc[str(jahr)].dropna(subset=["preis", "pv", "wind_on", "wind_off", "netzlast"])
        if len(d) < 8000:
            continue
        wind = d["wind_on"] + d["wind_off"]
        base = d["preis"].mean()
        jahr_z.append({"Jahr": jahr, "PV-Anteil [%]": 100 * d["pv"].sum() / d["netzlast"].sum(),
                       "Wind-Anteil [%]": 100 * wind.sum() / d["netzlast"].sum(),
                       "Capture Rate Solar [%]": 100 * (d["pv"] * d["preis"]).sum() / d["pv"].sum() / base,
                       "Capture Rate Wind [%]": 100 * (wind * d["preis"]).sum() / wind.sum() / base})
        for mon, m in d.groupby(d.index.month):
            mon_z.append({"Jahr": jahr, "Monat": mon, "PV-Anteil [%]": 100 * m["pv"].sum() / m["netzlast"].sum(),
                          "Capture Rate Solar [%]": 100 * (m["pv"] * m["preis"]).sum() / m["pv"].sum() / m["preis"].mean()})
    if len(jahr_z) < 3:
        print("  C6 übersprungen: weniger als drei Jahre mit vollständigen Preisdaten.")
        return
    jt, mt = pd.DataFrame(jahr_z).set_index("Jahr"), pd.DataFrame(mon_z)
    rho_j = sa._spearman(jt["PV-Anteil [%]"], jt["Capture Rate Solar [%]"])
    rho_m = sa._spearman(mt["PV-Anteil [%]"], mt["Capture Rate Solar [%]"])
    y = mt["Capture Rate Solar [%]"].to_numpy()
    dummies = pd.get_dummies(mt["Monat"], prefix="m", drop_first=True).astype(float).to_numpy()
    x = np.column_stack([np.ones(len(mt)), mt["PV-Anteil [%]"].to_numpy(), dummies])
    koef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ koef
    dof = max(len(y) - x.shape[1], 1)
    kov = (resid @ resid / dof) * np.linalg.pinv(x.T @ x)
    se = float(np.sqrt(kov[1, 1]))
    r2 = 1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean()))
    x0 = np.column_stack([np.ones(len(mt)), mt["PV-Anteil [%]"].to_numpy()])
    k0, *_ = np.linalg.lstsq(x0, y, rcond=None)

    # Bootstrap: (a) Monate unabhängig ziehen, (b) ganze Jahre ziehen (berücksichtigt Autokorrelation)
    rng = np.random.default_rng(42)
    monats_dummy = pd.get_dummies(pd.Categorical(mt["Monat"], categories=range(1, 13)), drop_first=True).astype(float).to_numpy()
    x_voll = np.column_stack([np.ones(len(mt)), mt["PV-Anteil [%]"].to_numpy(), monats_dummy])
    jahre_mt = mt["Jahr"].to_numpy()
    uniq = np.unique(jahre_mt)
    boot = {"Monate i.i.d.": [], "Jahresblöcke": []}
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, len(mt), len(mt))
        boot["Monate i.i.d."].append(np.linalg.lstsq(x_voll[idx], y[idx], rcond=None)[0][1])
        gew = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([np.where(jahre_mt == j)[0] for j in gew])
        boot["Jahresblöcke"].append(np.linalg.lstsq(x_voll[idx], y[idx], rcond=None)[0][1])
    bt_tab = pd.DataFrame([{"Verfahren": k, "b Median": np.median(v), "KI 2,5 %": np.percentile(v, 2.5),
                            "KI 97,5 %": np.percentile(v, 97.5), "Anteil b < 0 [%]": 100 * np.mean(np.array(v) < 0)}
                           for k, v in boot.items()]).set_index("Verfahren")
    sa.csv_de(bt_tab, ziel / "c6_bootstrap.csv")
    inf6 = ols_inferenz(x_voll, y, 1, NW_LAGS_MONAT)
    inf6_0 = ols_inferenz(x0, y, 1, NW_LAGS_MONAT)
    reg6 = pd.DataFrame([{k: v for k, v in i.items() if k != "koef"} | {"Modell": name}
                         for name, i in (("Ohne Monatskontrollen", inf6_0), ("Mit Monatskontrollen", inf6))]).set_index("Modell")
    reg6["KI 2,5 % (Jahresblöcke)"] = [np.nan, bt_tab.loc["Jahresblöcke", "KI 2,5 %"]]
    reg6["KI 97,5 % (Jahresblöcke)"] = [np.nan, bt_tab.loc["Jahresblöcke", "KI 97,5 %"]]
    sa.csv_de(reg6, ziel / "c6_regression.csv")
    plausi("C6 ohne Monats-Dummies gleiches Vorzeichen wie Spearman-ρ (Monat)", np.sign(k0[1]) == np.sign(rho_m))
    text = ["C6 · Robuste Kannibalisierung",
            f"Spearman ρ Jahresbasis (n = {len(jt)}): {rho_j:+.2f}",
            f"Spearman ρ Monatsbasis (n = {len(mt)}): {rho_m:+.2f}, t = {t_wert(rho_m, len(mt)):.1f}",
            f"Regression mit Monats-Dummies: b = {koef[1]:+.2f} Prozentpunkte Capture Rate je Prozentpunkt PV-Anteil "
            f"(Standardfehler {se:.2f}, t = {koef[1] / se:.1f}, R² = {r2:.2f})",
            f"Ohne Monats-Dummies: b = {k0[1]:+.2f}",
            "Bestehende Deck-Werte: ρ = −0,93 (n = 24) → t = "
            f"{t_wert(-0.93, 24):.1f}; ρ = +0,73 (n = 24) → t = {t_wert(0.73, 24):.1f}",
            "Hinweis: Saisonautokorrelation verkleinert die effektive Stichprobe; t-Werte sind optimistisch.",
            f"Bootstrap ({BOOTSTRAP_N} Wiederholungen), 95-%-Intervall für b:",
            *[f"  {k:<14} [{z['KI 2,5 %']:+.2f}; {z['KI 97,5 %']:+.2f}], Median {z['b Median']:+.2f}, "
              f"Anteil b < 0: {z['Anteil b < 0 [%]']:.1f} %" for k, z in bt_tab.iterrows()]]
    (ziel / "c6_ergebnis.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    sa.csv_de(jt, ziel / "c6_kennzahlen.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.33, 7.5))
    ax1.plot(jt["PV-Anteil [%]"], jt["Capture Rate Solar [%]"], "-o", color=CI["gold"], lw=1.5)
    for jahr, z in jt.iterrows():
        ax1.annotate(str(jahr), (z["PV-Anteil [%]"], z["Capture Rate Solar [%]"]), textcoords="offset points",
                     xytext=(5, 4), fontsize=8, color=CI["navy"])
    ax1.set_xlabel("PV-Anteil an der Netzlast im Jahr [%]", color=CI["navy"])
    ax1.set_ylabel("Capture Rate Solar [%]", color=CI["navy"])
    ax1.set_title(f"Jahreswerte (ρ = {de(rho_j, 2)})", loc="left", fontsize=11, color=CI["navy"], weight="bold")
    for mon, farbe, name in ((6, CI["gold"], "Juni"), (12, CI["navy"], "Dezember")):
        t = mt[mt["Monat"] == mon]
        ax2.plot(t["Jahr"], t["Capture Rate Solar [%]"], "-o", color=farbe, label=name)
    ax2.set_xlabel("Jahr", color=CI["navy"])
    ax2.set_ylabel("Capture Rate Solar [%]", color=CI["navy"])
    ax2.set_title("Gleicher Monat über die Jahre", loc="left", fontsize=11, color=CI["navy"], weight="bold")
    ax2.legend(frameon=False, fontsize=9)
    ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for ax in (ax1, ax2):
        sa._stil(ax)
        ax.axhline(100, color=CI["grau"], ls="--", lw=0.8)
    t_b = koef[1] / se
    if koef[1] < 0 and abs(t_b) >= 2:
        titel6 = f"Auch ohne Saisoneffekt: Jeder Prozentpunkt PV-Anteil kostet {de(abs(koef[1]), 1)} Prozentpunkte Capture Rate"
    elif koef[1] < 0:
        titel6 = f"Mit Saisonkontrolle bleibt der Zusammenhang negativ, aber nicht signifikant (t = {de(t_b, 1)})"
    else:
        titel6 = f"Mit Saisonkontrolle kein negativer Zusammenhang (b = {de(koef[1], 2)})"
    rahmen(fig, titel6,
           f"Regression mit Monats-Dummies über {len(mt)} Monatswerte, t = {de(koef[1] / se, 1)} · "
           "Capture Rate = mengengewichteter Marktwert / Base-Preis", demo)
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.07, right=0.98, wspace=0.22)
    speichern(fig, ziel / "19_kannibalisierung_robust.png")

    fig, ax = plt.subplots(figsize=(12, 6.75))
    for (k, v), farbe in zip(boot.items(), (CI["blau"], CI["gold"])):
        ax.hist(v, bins=60, alpha=0.55, color=farbe, label=k)
        z = bt_tab.loc[k]
        for q in ("KI 2,5 %", "KI 97,5 %"):
            ax.axvline(z[q], color=farbe, ls="--", lw=1)
    ax.axvline(koef[1], color=CI["navy"], lw=2, label=f"Punktschätzer b = {de(koef[1], 2)}")
    ax.axvline(0, color=CI["rot"], lw=1)
    sa._stil(ax)
    ax.set_xlabel("Koeffizient b [Prozentpunkte Capture Rate je Prozentpunkt PV-Anteil]", color=CI["navy"])
    ax.set_ylabel("Häufigkeit", color=CI["navy"])
    ax.legend(frameon=False, fontsize=9)
    zb = bt_tab.loc["Jahresblöcke"]
    rahmen(fig, f"95-%-Intervall des Kannibalisierungseffekts: {de(zb['KI 2,5 %'], 1)} bis {de(zb['KI 97,5 %'], 1)} "
                "(Jahresblock-Bootstrap)",
           f"{BOOTSTRAP_N} Bootstrap-Wiederholungen der Regression mit Monats-Dummies · Jahresblöcke berücksichtigen "
           "die Abhängigkeit der Monate innerhalb eines Jahres", demo)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.97)
    speichern(fig, ziel / "25_kannibalisierung_bootstrap.png")
    methodik(ziel, "c6", [
        "C6 · Robuste Kannibalisierung",
        "Capture Rate = Σ(Erzeugung × Preis) / Σ Erzeugung / Base-Preis; PV-Anteil = PV / Netzlast.",
        "Jahreswerte schließen den Saisoneffekt aus; Monatsregression mit 11 Monats-Dummies (OLS, numpy).",
        "t-Wert für Spearman: t = ρ × √((n − 2) / (1 − ρ²)); Saisonautokorrelation macht ihn optimistisch.",
        f"Bootstrap mit {BOOTSTRAP_N} Wiederholungen: Monate i.i.d. und Jahresblöcke (Blockbootstrap); 95-%-Perzentilintervall.",
        "SMARD-Proxy weicht vom offiziellen EEG-Marktwert (netztransparenz.de) geringfügig ab."])


# ===========================================================================
# D1 · Haushaltsstrompreis real
# ===========================================================================
BLOECKE = {"Markt / Vertrieb": ["erzeugung", "beschaffung", "vertrieb"],
           "Netz / Messung": ["netz", "mess"],
           "Steuern, Abgaben, Umlagen": ["konzession", "umlage", "steuer", "eeg"]}


def modell_d1(eingaben, ziel, demo):
    pfad_p, pfad_v = eingaben / "strompreis_haushalt.csv", eingaben / "vpi.csv"
    ok_p = vorlage(pfad_p, "bestandteil;2017;aktuell;quelle_aktuell\n"
                           "Erzeugung, Beschaffung, Vertrieb;5,63;;\nNetzentgelte, Messung, Abrechnung;7,48;;\n"
                           "Konzessionsabgabe;1,66;;\nSonstige Umlagen;0,80;;\nEEG-Umlage;6,88;;\n"
                           "Stromsteuer;2,05;;\nUmsatzsteuer;4,66;;\n")
    ok_v = vorlage(pfad_v, "jahr;index;quelle\n2017;;Destatis 61111-0001\n2025;;Destatis 61111-0001\n")
    if not (ok_p and ok_v):
        return
    p = lies_csv_de(pfad_p)
    p.columns = [str(c).strip() for c in p.columns]
    if "aktuell" not in p.columns:
        jahres_sp = [c for c in p.columns if c.isdigit() and c != "2017"]
        if not jahres_sp or "2017" not in p.columns:
            print(f"  D1 übersprungen: Spalten '2017' und 'aktuell' erwartet, gefunden: {list(p.columns)}")
            return
        p = p.rename(columns={max(jahres_sp): "aktuell"})
    p["2017"] = pd.to_numeric(p["2017"], errors="coerce")
    p["aktuell"] = pd.to_numeric(p["aktuell"], errors="coerce")
    v = pd.to_numeric(lies_csv_de(pfad_v).set_index("jahr")["index"], errors="coerce")
    if p["aktuell"].isna().all() or v.isna().any() or 2017 not in v.index:
        print("  D1 übersprungen: Eingaben unvollständig.")
        return
    jahr_akt = int(v.index.max())
    faktor = v.loc[jahr_akt] / v.loc[2017]
    blk = {}
    for name, schl in BLOECKE.items():
        m = p["bestandteil"].str.lower().apply(lambda s: any(k in s for k in schl))
        blk[name] = (p.loc[m, "2017"].sum(), p.loc[m, "aktuell"].fillna(0).sum())
        p = p[~m]
    if len(p):
        print(f"  D1: nicht zugeordnete Bestandteile: {list(p['bestandteil'])}")
    s17, sakt = sum(a for a, _ in blk.values()), sum(b for _, b in blk.values())
    plausi("D1 Summe 2017 = 29,16 ct/kWh", abs(s17 - 29.16) <= 0.05, f"{s17:.2f}")
    real17 = s17 * faktor
    fig, ax = plt.subplots(figsize=(12, 6.75))
    reihen = [("2017 nominal", [a for a, _ in blk.values()]), (f"2017 real ({jahr_akt})", [a * faktor for a, _ in blk.values()]),
              (f"{jahr_akt}", [b for _, b in blk.values()])]
    farben = [CI["navy"], CI["blau"], CI["gold"]]
    for i, (name, werte) in enumerate(reihen[::-1]):
        links = 0
        for w, f in zip(werte, farben):
            ax.barh(i, w, left=links, color=f, height=0.55)
            if w > 1.5:
                ax.text(links + w / 2, i, de(w), ha="center", va="center", color="white", fontsize=9, weight="bold")
            links += w
        ax.text(links + 0.4, i, f"{de(links, 2)} ct/kWh", va="center", fontsize=10, color=CI["navy"], weight="bold")
    ax.set_yticks(range(3), [r[0] for r in reihen[::-1]])
    for f, n in zip(farben, BLOECKE):
        ax.barh(0, 0, color=f, label=n)
    ax.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3)
    sa._stil(ax)
    ax.set_xlabel("ct/kWh, Haushalt 3.500 kWh/a", color=CI["navy"])
    rahmen(fig, f"Nominal +{de(sakt - s17, 2)} ct/kWh seit 2017, real +{de(sakt - real17, 2)} ct/kWh",
           f"Preisbasis {jahr_akt} über Verbraucherpreisindex (Faktor {de(faktor, 3)}) · BDEW-Komponenten in drei Blöcken",
           demo, quelle="Quelle: BDEW Strompreisanalyse; Destatis VPI · eigene Berechnung THE PATTERN MIND")
    ax.set_xlim(0, max(s17 * faktor, sakt) * 1.18)
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.14, right=0.95)
    speichern(fig, ziel / "22_strompreis_real.png")


# ===========================================================================
# D2 · EU-Strompreise in EUR und PPS
# ===========================================================================
EU27 = ["BE", "BG", "CZ", "DK", "DE", "EE", "IE", "EL", "ES", "FR", "HR", "IT", "CY", "LV", "LT", "LU", "HU",
        "MT", "NL", "AT", "PL", "PT", "RO", "SI", "SK", "FI", "SE"]
EUROSTAT_URL = ("https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/nrg_pc_204"
                "?format=JSON&lang=DE&nrg_cons=KWH2500-4999&tax=I_TAX&unit=KWH&currency={cur}")  # Parameter prüfen


def jsonstat_zu_df(daten: dict) -> pd.DataFrame:
    ids, groessen = daten["id"], daten["size"]
    kat = [daten["dimension"][d]["category"]["index"] for d in ids]
    kat_inv = [{v: k for k, v in (c.items() if isinstance(c, dict) else enumerate(c))} for c in kat]
    zeilen = []
    for flach, wert in daten["value"].items():
        rest, pos = int(flach), []
        for g in reversed(groessen):
            pos.append(rest % g)
            rest //= g
        pos = pos[::-1]
        zeilen.append({**{d: kat_inv[i][pos[i]] for i, d in enumerate(ids)}, "wert": wert})
    return pd.DataFrame(zeilen)


def modell_d2(eingaben, ziel, demo):
    daten = None
    if not demo:
        try:
            teile = []
            for cur, sp in (("EUR", "eur_kwh"), ("PPS", "pps_kwh")):
                with urllib.request.urlopen(EUROSTAT_URL.format(cur=cur), timeout=30) as r:
                    t = jsonstat_zu_df(json.loads(r.read().decode("utf-8")))
                teile.append(t.rename(columns={"wert": sp})[["geo", "time", sp]])
            daten = teile[0].merge(teile[1], on=["geo", "time"]).rename(columns={"geo": "land", "time": "zeitraum"})
        except Exception as fehler:  # noqa: BLE001
            print(f"  D2: Eurostat-API nicht erreichbar ({fehler}) – nutze Eingabedatei.")
    if daten is None:
        pfad = eingaben / "eurostat_haushalt.csv"
        if demo:
            rng = np.random.default_rng(3)
            daten = pd.DataFrame([{"land": l, "zeitraum": z, "eur_kwh": rng.uniform(0.1, 0.38),
                                   "pps_kwh": rng.uniform(0.12, 0.40)} for l in EU27 for z in ("2017-S2", "2026-S1")])
        elif not vorlage(pfad, "land;zeitraum;eur_kwh;pps_kwh\nDE;2017-S2;;\nDE;2026-S1;;\n"):
            return
        else:
            daten = lies_csv_de(pfad)
    daten = daten[daten["land"].isin(EU27)]
    zeiten = sorted(daten["zeitraum"].unique())
    alt = "2017-S2" if "2017-S2" in zeiten else zeiten[0]
    neu = zeiten[-1]
    fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
    raenge = {}
    for ax, sp, name in ((axes[0], "eur_kwh", "Euro"), (axes[1], "pps_kwh", "Kaufkraftstandard (PPS)")):
        t = daten.pivot_table(index="land", columns="zeitraum", values=sp).dropna(subset=[neu])
        t = t.sort_values(neu)
        y = np.arange(len(t))
        for i, (land, z) in enumerate(t.iterrows()):
            farbe = CI["gold"] if land == "DE" else CI["navy"]
            if alt in t.columns and not np.isnan(z.get(alt, np.nan)):
                ax.plot([z[alt] * 100, z[neu] * 100], [i, i], color=CI["linie"], lw=2, zorder=1)
                ax.scatter(z[alt] * 100, i, color=CI["grau"], s=18, zorder=2)
            ax.scatter(z[neu] * 100, i, color=farbe, s=34, zorder=3)
        ax.set_yticks(y, t.index, fontsize=7)
        sa._stil(ax)
        ax.set_xlabel(f"ct/kWh in {name}", color=CI["navy"])
        rang = int((t[neu].rank(ascending=False)).get("DE", np.nan)) if "DE" in t.index else None
        raenge[name] = rang
        ax.set_title(f"{name} · Deutschland Rang {rang}", loc="left", fontsize=11, color=CI["navy"], weight="bold")
    rahmen(fig, f"Deutschland {neu}: Rang {raenge['Euro']} in Euro, Rang {raenge['Kaufkraftstandard (PPS)']} in Kaufkraft",
           f"Haushalte 2.500–4.999 kWh/a, inkl. Steuern und Abgaben · grau {alt}, farbig {neu} · EU-27",
           demo, quelle="Quelle: Eurostat nrg_pc_204 · eigene Darstellung THE PATTERN MIND")
    fig.subplots_adjust(top=0.84, bottom=0.10, left=0.06, right=0.98, wspace=0.18)
    speichern(fig, ziel / "23_strompreise_europa_eur_pps.png")


# ===========================================================================
# D3 · HPC-Standortmodell
# ===========================================================================
def hpc_kapitalwert(p: dict, batterie: bool = False) -> dict:
    db_kwh = p["verkaufspreis"] - p["beschaffung"] - p["netz_arbeitspreis"] - p["abwicklung"]
    spitze = p["ladepunkte"] * p["leistung_kw"] * p["gleichzeitigkeit"]
    capex = p["capex"]
    if batterie:
        spitze = min(spitze, p["batterie_kappung_kw"])
        capex += p["batterie_kwh"] * p["batterie_capex_kwh"]
    kw = -capex
    for t in range(1, p["jahre"] + 1):
        ausl = p["hochlauf"][min(t - 1, len(p["hochlauf"]) - 1)]
        menge = p["ladungen_tag"] * ausl * p["kwh_je_ladung"] * 365
        cf = menge * db_kwh - p["opex"] - spitze * p["leistungspreis_kw"]
        kw += cf / (1 + p["zins"]) ** t
    lade_je_lp = p["ladungen_tag"] / p["ladepunkte"]
    auslastung = p["ladungen_tag"] * p["kwh_je_ladung"] / (p["ladepunkte"] * p["leistung_kw"] * 24)
    return {"Kapitalwert [€]": kw, "Deckungsbeitrag [€/kWh]": db_kwh, "Spitze [kW]": spitze,
            "Leistungspreis [€/a]": spitze * p["leistungspreis_kw"], "Ladungen je Ladepunkt und Tag": lade_je_lp,
            "Zeitauslastung [%]": 100 * auslastung}


def break_even_ladungen(p: dict, batterie: bool = False) -> float:
    lo, hi = 1.0, 2000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if hpc_kapitalwert({**p, "ladungen_tag": mid}, batterie)["Kapitalwert [€]"] > 0:
            hi = mid
        else:
            lo = mid
    return hi


def modell_d3(ziel, demo):
    basis = hpc_kapitalwert(HPC)
    varianten = {
        "Ohne Leistungspreis (Deck Folie 47)": {**HPC, "leistungspreis_kw": 0.0, "hochlauf": (1.0,)},
        "Mit Leistungspreis, ohne Hochlauf": {**HPC, "hochlauf": (1.0,)},
        "Mit Leistungspreis und Hochlauf": HPC}
    zeilen = []
    for name, p in varianten.items():
        for batt in (False, True):
            if batt and p["leistungspreis_kw"] == 0:
                continue
            r = hpc_kapitalwert(p, batt)
            zeilen.append({"Variante": name + (" + Batterie" if batt else ""), **r,
                           "Break-even Ladungen/Tag": break_even_ladungen(p, batt)})
    tab = pd.DataFrame(zeilen).set_index("Variante")
    sa.csv_de(tab, ziel / "d3_kennzahlen.csv")
    ref = tab.loc["Ohne Leistungspreis (Deck Folie 47)", "Break-even Ladungen/Tag"]
    plausi("D3 Reproduktion Deck-Break-even ≈ 113 Ladungen/Tag (ohne Leistungspreis, DB 0,20)",
           abs(ref - 113) < 2, f"{ref:.1f}")
    params = [("ladungen_tag", "Ladungen pro Tag"), ("verkaufspreis", "Verkaufspreis"), ("beschaffung", "Beschaffung"),
              ("leistungspreis_kw", "Leistungspreis"), ("capex", "CAPEX"), ("zins", "Zins")]
    tor = []
    for k, n in params:
        lo = hpc_kapitalwert({**HPC, k: HPC[k] * 0.8})["Kapitalwert [€]"] - basis["Kapitalwert [€]"]
        hi = hpc_kapitalwert({**HPC, k: HPC[k] * 1.2})["Kapitalwert [€]"] - basis["Kapitalwert [€]"]
        tor.append((n, lo, hi))
    tor.sort(key=lambda x: abs(x[2] - x[1]))
    fig, ax = plt.subplots(figsize=(12, 6.75))
    for i, (n, lo, hi) in enumerate(tor):
        ax.barh(i, lo / 1e6, color=CI["rot"] if lo < 0 else CI["gruen"], height=0.55)
        ax.barh(i, hi / 1e6, color=CI["rot"] if hi < 0 else CI["gruen"], height=0.55)
        ax.text(min(lo, hi) / 1e6, i, f"{de(min(lo, hi) / 1e6, 2)}  ", ha="right", va="center", fontsize=8)
        ax.text(max(lo, hi) / 1e6, i, f"  {de(max(lo, hi) / 1e6, 2)}", ha="left", va="center", fontsize=8)
    ax.set_yticks(range(len(tor)), [t[0] for t in tor])
    ax.axvline(0, color=CI["navy"], lw=1)
    sa._stil(ax)
    ax.set_xlabel("Änderung des Kapitalwerts bei ±20 % [Mio. €]", color=CI["navy"])
    mit = tab.loc["Mit Leistungspreis und Hochlauf", "Break-even Ladungen/Tag"]
    rahmen(fig, f"Mit Leistungspreis und Hochlauf liegt der Break-even bei {de(mit, 0)} statt {de(ref, 0)} Ladungen/Tag",
           f"Basis: {HPC['ladepunkte']} × {HPC['leistung_kw']} kW, {HPC['ladungen_tag']} Ladungen/Tag, Kapitalwert "
           f"{de(basis['Kapitalwert [€]'] / 1e6, 2)} Mio. €, Zeitauslastung {de(basis['Zeitauslastung [%]'], 0)} % · "
           "alle Werte Annahmen", demo, quelle="Eigenes Standortmodell THE PATTERN MIND · Annahmen im Konfigurationsblock")
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.16, right=0.95)
    speichern(fig, ziel / "20_hpc_tornado.png")


# ===========================================================================
# D4 · Heimspeicher
# ===========================================================================
def anteil_negativ(df: pd.DataFrame) -> float:
    d = df.dropna(subset=["pv", "preis"])
    if d.empty:
        return 0.0
    letzt = d.loc[str(d.index.year.max())]
    return float(letzt.loc[letzt["preis"] < 0, "pv"].sum() / letzt["pv"].sum())


def heim_kw(preis_anschaffung: float, pfad: float, s_neg: float, h: dict = HEIM) -> float:
    kw = -preis_anschaffung
    for t in range(1, h["jahre"] + 1):
        kap = (1 - h["alterung"]) ** (t - 1)
        ladung = h["ladung_kwh"] * kap
        strompreis = h["strompreis"] * (1 + pfad) ** (t - 1)
        opportunitaet = h["einspeiseverguetung"] * (1 - s_neg)
        kw += (ladung * h["eta"] * strompreis - ladung * opportunitaet) / (1 + h["zins"]) ** t
    return kw


def modell_d4(df, ziel, demo):
    s_neg = anteil_negativ(df)
    zeilen = []
    for pfad in HEIM["preispfade"]:
        for p in HEIM["preise"]:
            zeilen.append({"Strompreispfad [%/a]": 100 * pfad, "Anschaffung [€]": p, "Kapitalwert [€]": heim_kw(p, pfad, s_neg)})
    tab = pd.DataFrame(zeilen)
    sa.csv_de(tab.set_index(["Strompreispfad [%/a]", "Anschaffung [€]"]), ziel / "d4_kennzahlen.csv")
    deck = heim_kw(6000, 0.0, 0.0, {**HEIM, "alterung": 0.0, "einspeiseverguetung": 0.08})
    plausi("D4 Reproduktion Deck: 6.000 € → ca. −700 € (ohne Alterung, Pfad 0, ohne Solarspitzen)",
           abs(deck + 698) < 15, f"{deck:.0f} €")
    fig, ax = plt.subplots(figsize=(12, 6.75))
    for pfad, farbe in zip(HEIM["preispfade"], (CI["grau"], CI["blau"], CI["navy"])):
        t = tab[tab["Strompreispfad [%/a]"] == 100 * pfad]
        ax.plot(t["Anschaffung [€]"], t["Kapitalwert [€]"], "-o", color=farbe, lw=2,
                label=f"Strompreis +{de(100 * pfad, 0)} % pro Jahr")
    ax.axhline(0, color=CI["navy"], lw=0.9)
    sa._stil(ax)
    ax.set_xlabel("Anschaffungspreis [€]", color=CI["navy"])
    ax.set_ylabel("Kapitalwert [€]", color=CI["navy"])
    ax.legend(frameon=False, fontsize=9)
    be = {}
    for pfad in HEIM["preispfade"]:
        lo, hi = 0.0, 20000.0
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if heim_kw(mid, pfad, s_neg) > 0 else (lo, mid)
        be[pfad] = lo
    rahmen(fig, f"Break-even zwischen {de(be[HEIM['preispfade'][0]], 0)} € und {de(be[HEIM['preispfade'][-1]], 0)} € "
                "– je nach Strompreisentwicklung",
           f"Alterung {de(100 * HEIM['alterung'], 0)} %/a, {HEIM['jahre']} Jahre, Zins {de(100 * HEIM['zins'], 0)} % · "
           f"{de(100 * s_neg, 1)} % der PV-Energie fallen in negative Preisstunden (ohne Vergütung)", demo,
           quelle="Eigene Kapitalwertrechnung THE PATTERN MIND · Preise als Modellannahmen")
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.97)
    speichern(fig, ziel / "21_heimspeicher_sensitivitaet.png")


# ===========================================================================
# D5 · Modellkommune
# ===========================================================================
def modell_d5(df, kap, ziel):
    k = KOMMUNE
    puffer_min = k["speicher_mwh"] / k["pv_mwp"] * 60
    zeilen = [f"D5 · Modellkommune ({k['haushalte']:,} Haushalte)".replace(",", "."),
              f"Pufferdauer: {k['speicher_mwh']} MWh / {k['pv_mwp']} MW = {puffer_min:.0f} Minuten PV-Spitzenleistung.",
              f"Gleichzeitige Ladeleistung: {k['ladepunkte']} × {k['ladeleistung_kw']:.0f} kW = "
              f"{k['ladepunkte'] * k['ladeleistung_kw'] / 1000:.1f} MW."]
    if kap is not None:
        d = df.dropna(subset=["pv", "netzlast"])
        letzt = d.loc[str(d.index.year.max())]
        tage = letzt["pv"].groupby(letzt.index.date).sum()
        tag = tages = tage.idxmax()
        t = letzt[letzt.index.date == tag]
        pv_k = t["pv"] / kap.loc[t.index, "pv"] * k["pv_mwp"]
        last_k = t["netzlast"] / letzt["netzlast"].mean() * k["haushalte"] * k["verbrauch_kwh"] / 8760 / 1000
        ueber = (pv_k - last_k).clip(lower=0).sum()
        zeilen += [f"Sonnigster Tag {tag:%d.%m.%Y}: PV {pv_k.sum():.1f} MWh, Verbrauch {last_k.sum():.1f} MWh, "
                   f"Überschuss {ueber:.1f} MWh.",
                   f"Der Speicher nimmt höchstens {min(k['speicher_mwh'], ueber):.1f} MWh auf = "
                   f"{100 * min(k['speicher_mwh'], ueber) / ueber if ueber else 0:.0f} % des Tagesüberschusses.",
                   "Lastprofil: Form der nationalen Netzlast, skaliert auf den Haushaltsverbrauch (Näherung)."]
    (ziel / "d5_modellkommune.txt").write_text("\n".join(zeilen) + "\n", encoding="utf-8")


# ===========================================================================
# R1 · Robustheit der Spitzenzerlegung
# ===========================================================================
def modell_r1(df, jahre, ziel, demo):
    basis = jahre[0]
    zeilen = []
    for n in TOP_N_VARIANTEN:
        ref = None
        for jahr in jahre:
            d = df.loc[str(jahr)].dropna(subset=["netzlast", "pv", "wind_on", "wind_off"])
            if len(d) < 8000:
                continue
            top = d.nlargest(n, "residuallast")
            w = {k: top[k].mean() / 1000 for k in ("residuallast", "netzlast", "wind_on", "wind_off", "pv")}
            if jahr == basis:
                ref = w
                continue
            if ref is None:
                continue
            d_r = w["residuallast"] - ref["residuallast"]
            d_l = w["netzlast"] - ref["netzlast"]
            d_ee = -sum(w[k] - ref[k] for k in ("wind_on", "wind_off", "pv"))
            plausi_ok = abs(d_l + d_ee - d_r) < 1e-9
            if not plausi_ok:
                plausi(f"R1 Zerlegung {jahr} Top-{n}", False)
            anteil = abs(d_l) / (abs(d_l) + abs(d_ee)) if (d_l or d_ee) else np.nan
            zeilen.append({"Top-N": n, "Jahr": jahr, "ΔResiduallast [GW]": d_r, "Beitrag Netzlast [GW]": d_l,
                           "Beitrag Wind+PV [GW]": d_ee, "Anteil Netzlast [%]": 100 * anteil,
                           "Lastgetrieben": bool(d_r < 0 and anteil > 0.5)})
    if not zeilen:
        print("  R1 übersprungen: zu wenige Jahre.")
        return
    tab = pd.DataFrame(zeilen)
    plausi("R1 Zerlegungen summieren sich exakt (alle Jahre, alle Top-N)", True)
    sa.csv_de(tab.set_index(["Top-N", "Jahr"]), ziel / "r1_kennzahlen.csv")
    rueck = tab[tab["ΔResiduallast [GW]"] < 0]
    n_last, n_ges = int(rueck["Lastgetrieben"].sum()), len(rueck)
    lj = tab["Jahr"].max()
    letzt = tab[tab["Jahr"] == lj].set_index("Top-N")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.33, 7.5), gridspec_kw={"width_ratios": [3, 2]})
    t = tab[tab["Top-N"] == 100]
    ax1.bar(t["Jahr"] - 0.2, t["Beitrag Netzlast [GW]"], 0.4, color=CI["navy"], label="Beitrag Netzlast")
    ax1.bar(t["Jahr"] + 0.2, t["Beitrag Wind+PV [GW]"], 0.4, color=CI["teal"], label="Beitrag Wind+PV")
    ax1.plot(t["Jahr"], t["ΔResiduallast [GW]"], "o-", color=CI["rot"], lw=2, label="ΔResiduallast")
    ax1.axhline(0, color=CI["navy"], lw=0.8)
    ax1.set_title(f"Top-100-Stunden, Veränderung gegenüber {basis}", loc="left", fontsize=11,
                  color=CI["navy"], weight="bold")
    ax1.set_ylabel("GW", color=CI["navy"])
    ax1.legend(frameon=False, fontsize=8)
    ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax2.bar([str(n) for n in letzt.index], letzt["Anteil Netzlast [%]"], color=CI["navy"])
    ax2.axhline(50, color=CI["rot"], ls="--", lw=1)
    for i, v in enumerate(letzt["Anteil Netzlast [%]"]):
        ax2.text(i, v + 1.5, f"{de(v, 0)} %", ha="center", fontsize=9, color=CI["navy"])
    ax2.set_title(f"Anteil Netzlast {basis}→{lj} je Stundenzahl", loc="left", fontsize=11,
                  color=CI["navy"], weight="bold")
    ax2.set_xlabel("Anzahl kritischster Stunden", color=CI["navy"])
    ax2.set_ylabel("Anteil am Betrag der Veränderung [%]", color=CI["navy"])
    ax2.set_ylim(0, 100)
    for ax in (ax1, ax2):
        sa._stil(ax)
    rahmen(fig, f"Robust: In {n_last} von {n_ges} Varianten mit sinkender Spitze überwiegt der Lastbeitrag",
           f"Varianten = Jahre {jahre[1]}–{lj} × Top-{', '.join(map(str, TOP_N_VARIANTEN))}-Stunden, jeweils gegenüber {basis} · "
           "Anteil = |ΔNetzlast| / (|ΔNetzlast| + |ΔWind+PV|)", demo)
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.07, right=0.98, wspace=0.22)
    speichern(fig, ziel / "24_robust_spitzenzerlegung.png")
    methodik(ziel, "r1", [
        "R1 · Robustheit der Spitzenzerlegung",
        f"Wiederholung der C2-Zerlegung für alle Jahre gegenüber {basis} und für die Top-{TOP_N_VARIANTEN}-Stunden.",
        "Beitrag Wind+PV = −(ΔWind an Land + ΔWind auf See + ΔPV); ΔR = ΔNetzlast + Beitrag Wind+PV (exakt).",
        "Lastgetrieben = Rückgang der Residuallast und Anteil der Netzlast am Betrag der Veränderung über 50 %.",
        f"Ergebnis: {n_last} von {n_ges} Varianten mit sinkender Spitze sind lastgetrieben."])


# ===========================================================================
# R4 · Plausibilisierung der SMARD-Netzlast
# ===========================================================================
def modell_r4(df, jahre, eingaben, ziel, demo):
    smard = {j: df.loc[str(j), "netzlast"].sum() / 1e6 for j in jahre
             if df.loc[str(j), "netzlast"].notna().sum() > 8000}
    if demo:
        ageb = pd.Series({j: v * 1.17 * (1 - 0.004 * (j - 2015)) for j, v in smard.items()})
    else:
        pfad = eingaben / "ageb_bruttostromverbrauch.csv"
        kopf = "jahr;bruttostromverbrauch_twh;quelle\n" + "".join(f"{j};;\n" for j in jahre)
        if not vorlage(pfad, kopf):
            return
        roh = lies_csv_de(pfad).set_index("jahr")
        ageb = pd.to_numeric(roh["bruttostromverbrauch_twh"], errors="coerce").dropna()
    gemeinsam = sorted(set(smard) & set(ageb.index))
    if len(gemeinsam) < 2:
        print("  R4 übersprungen: zu wenige gemeinsame Jahre.")
        return
    tab = pd.DataFrame({"SMARD-Netzlast [TWh]": [smard[j] for j in gemeinsam],
                        "Bruttostromverbrauch [TWh]": [ageb[j] for j in gemeinsam]}, index=gemeinsam)
    tab.index.name = "Jahr"
    tab["Verhältnis SMARD/Brutto"] = tab["SMARD-Netzlast [TWh]"] / tab["Bruttostromverbrauch [TWh]"]
    a, b = gemeinsam[0], gemeinsam[-1]
    for sp, kurz in (("SMARD-Netzlast [TWh]", "SMARD"), ("Bruttostromverbrauch [TWh]", "Brutto")):
        tab[f"Index {kurz} ({a}=100)"] = 100 * tab[sp] / tab.loc[a, sp]
    sa.csv_de(tab, ziel / "r4_kennzahlen.csv")
    d_s = 100 * (tab.loc[b, "SMARD-Netzlast [TWh]"] / tab.loc[a, "SMARD-Netzlast [TWh]"] - 1)
    d_b = 100 * (tab.loc[b, "Bruttostromverbrauch [TWh]"] / tab.loc[a, "Bruttostromverbrauch [TWh]"] - 1)
    gleiche_richtung = np.sign(d_s) == np.sign(d_b)
    urteil = ("bestätigt: Rückgang zeigt sich auch im Bruttostromverbrauch" if gleiche_richtung and abs(d_b) >= 0.5 * abs(d_s)
              else "nicht bestätigt: Reihen laufen auseinander, Definitionseffekt möglich")
    (ziel / "r4_ergebnis.txt").write_text("\n".join([
        "R4 · Plausibilisierung der SMARD-Netzlast",
        f"SMARD-Netzlast {a}→{b}: {d_s:+.1f} %",
        f"Bruttostromverbrauch {a}→{b}: {d_b:+.1f} %",
        f"Verhältnis SMARD/Brutto: {tab.loc[a, 'Verhältnis SMARD/Brutto']:.3f} → {tab.loc[b, 'Verhältnis SMARD/Brutto']:.3f}",
        f"Urteil: {urteil}"]) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.plot(tab.index, tab[f"Index SMARD ({a}=100)"], "o-", color=CI["navy"], lw=2.2, label="SMARD-Netzlast")
    ax.plot(tab.index, tab[f"Index Brutto ({a}=100)"], "s-", color=CI["gold"], lw=2.2,
            label="Bruttostromverbrauch (AGEB/BDEW)")
    ax.axhline(100, color=CI["grau"], lw=0.8, ls=":")
    ax2 = ax.twinx()
    ax2.bar(tab.index, tab["Verhältnis SMARD/Brutto"], color=CI["linie"], alpha=0.6, width=0.6, zorder=0)
    ax2.set_ylim(0, 1.6)
    ax2.set_ylabel("Verhältnis SMARD / Brutto (Balken)", color=CI["grau"])
    ax2.spines["top"].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    sa._stil(ax)
    ax.set_ylabel(f"Index ({a} = 100)", color=CI["navy"])
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    rahmen(fig, f"Netzlast {de(d_s, 1)} %, Bruttostromverbrauch {de(d_b, 1)} % seit {a}: "
                + ("der Rückgang ist real" if gleiche_richtung and abs(d_b) >= 0.5 * abs(d_s) else "Reihen laufen auseinander"),
           "SMARD-Netzlast (Netz der allgemeinen Versorgung) gegen Bruttostromverbrauch der Energiebilanz · "
           "unabhängige Plausibilisierung von Befund C2", demo,
           quelle="Quelle: Bundesnetzagentur | SMARD.de; AG Energiebilanzen / BDEW · eigene Berechnung THE PATTERN MIND")
    fig.subplots_adjust(top=0.84, bottom=0.10, left=0.08, right=0.92)
    speichern(fig, ziel / "26_netzlast_plausibilisierung.png")
    methodik(ziel, "r4", [
        "R4 · Plausibilisierung der SMARD-Netzlast",
        "Jahressumme der stündlichen SMARD-Netzlast gegen den Bruttostromverbrauch der Energiebilanz (AGEB/BDEW).",
        "Bruttostromverbrauch enthält zusätzlich Kraftwerkseigenverbrauch, Netzverluste, Pumparbeit, industrielle",
        "Eigenerzeugung und Bahnstrom; das Verhältnis beider Reihen ist deshalb kleiner als 1.",
        "Bestätigt, wenn beide Reihen in dieselbe Richtung laufen und der Rückgang im Brutto mindestens halb so groß ist.",
        "Bruttostromverbrauch 2025 vorläufig; Revisionen der Energiebilanz verändern einzelne Jahreswerte."])


# ===========================================================================
# C7 · Außenhandel und Pumpspeicher in kritischen Stunden
# ===========================================================================
SMARD_DOWNLOAD = "https://www.smard.de/nip-download-manager/nip/download/market-data"
# Modulkennungen „Kommerzieller Außenhandel“ (SMARD-Downloadcenter, dokumentiert in bundesAPI/smard-api, Issue 12).
# Die Zuordnung zu Ländern ergibt sich aus den Spaltennamen der gelieferten CSV, nicht aus diesen Kennungen.
TRADE_MODULE = [22004629, 22004722, 22004724, 22004998, 22004712, 22004404, 22004409, 22004545, 22004546,
                22004548, 22004550, 22004551, 22004552, 22004405, 22004547, 22004403, 22004406, 22004407,
                22004408, 22004410, 22004412, 22004549, 22004553]
PUMPSPEICHER = {"pump_erz": (4070, "DE"), "pump_verbr": (4387, "DE")}  # Erzeugung bzw. Verbrauch Pumpspeicher


def _zeitindex_aus_csv(df: pd.DataFrame) -> pd.DatetimeIndex:
    spalten = list(df.columns)
    if "Datum von" in spalten:
        roh = df["Datum von"].astype(str)
    elif "Datum" in spalten and "Anfang" in spalten:
        roh = df["Datum"].astype(str) + " " + df["Anfang"].astype(str)
    else:
        start = next(c for c in spalten if str(c).lower().startswith("datum"))
        roh = df[start].astype(str)
    try:
        naiv = pd.to_datetime(roh, format="%d.%m.%Y %H:%M")
    except (ValueError, TypeError):
        naiv = pd.to_datetime(roh, dayfirst=True)
    return sa._lokalisiere(pd.DatetimeIndex(naiv))


def handel_aus_csv(text_oder_pfad) -> pd.DataFrame | None:
    df = pd.read_csv(text_oder_pfad, sep=";", decimal=",", thousands=".", na_values=["-", ""],
                     encoding="utf-8-sig", low_memory=False)
    idx = _zeitindex_aus_csv(df)
    exp = [c for c in df.columns if "export" in str(c).lower() and "netto" not in str(c).lower()]
    imp = [c for c in df.columns if "import" in str(c).lower() and "netto" not in str(c).lower()]
    netto = [c for c in df.columns if "netto" in str(c).lower()]
    if not (exp and imp) and not netto:
        return None
    out = pd.DataFrame(index=idx)
    zahl = lambda c: pd.to_numeric(df[c], errors="coerce").to_numpy()
    if exp and imp:
        out["export"] = np.nansum(np.abs(np.column_stack([zahl(c) for c in exp])), axis=1)
        out["import"] = np.nansum(np.abs(np.column_stack([zahl(c) for c in imp])), axis=1)
        out["nettoexport"] = out["export"] - out["import"]
    if netto:
        out["nettoexport_csv"] = zahl(netto[0])
        if "nettoexport" not in out:
            out["nettoexport"] = out["nettoexport_csv"]
    out.attrs["spalten"] = {"export": exp, "import": imp, "netto": netto}
    return out


def lade_handel(jahre, quelle, eingaben: Path, cache: Path, df_basis: pd.DataFrame) -> pd.DataFrame | None:
    if quelle == "demo":
        r = df_basis["residuallast"]
        rng = np.random.default_rng(7)
        netto = -0.25 * (r - r.mean()) + rng.normal(0, 1500, len(r)) - 500
        out = pd.DataFrame({"nettoexport": netto, "export": netto.clip(lower=0) + 3000,
                            "import": (-netto).clip(lower=0) + 3000}, index=r.index)
        out.attrs["spalten"] = {"export": ["Demo"], "import": ["Demo"], "netto": []}
        return out
    teile = []
    dateien = sorted(eingaben.glob("smard_aussenhandel*.csv"))
    if dateien:
        for d in dateien:
            t = handel_aus_csv(d)
            if t is not None:
                teile.append(t)
                print(f"  C7: Handelsdaten aus {d.name}")
    else:
        cache.mkdir(parents=True, exist_ok=True)
        for jahr in jahre:
            datei = cache / f"aussenhandel_{jahr}.csv"
            if not datei.exists():
                von = int(pd.Timestamp(f"{jahr}-01-01", tz=sa.TZ).value // 10**6)
                bis = int(pd.Timestamp(f"{jahr + 1}-01-01", tz=sa.TZ).value // 10**6) - 1
                nutzlast = {"request_form": [{"format": "CSV", "moduleIds": TRADE_MODULE, "region": "DE",
                                              "timestamp_from": von, "timestamp_to": bis,
                                              "type": "discrete", "language": "de"}]}
                try:
                    anfrage = urllib.request.Request(SMARD_DOWNLOAD, data=json.dumps(nutzlast).encode("utf-8"),
                                                     headers={"Content-Type": "application/json",
                                                              "User-Agent": "THE-PATTERN-MIND-SMARD-Analyse/1.0"})
                    with urllib.request.urlopen(anfrage, timeout=120) as antwort:
                        inhalt = antwort.read().decode("utf-8-sig")
                    if "Datum" not in inhalt[:200]:
                        raise ValueError("Antwort ist keine SMARD-CSV")
                    datei.write_text(inhalt, encoding="utf-8")
                except Exception as fehler:  # noqa: BLE001
                    print(f"  C7: Download {jahr} fehlgeschlagen ({fehler}).")
                    continue
            t = handel_aus_csv(datei)
            if t is not None:
                teile.append(t)
    if not teile:
        print("  C7 übersprungen: keine Handelsdaten. Manuell: SMARD-Downloadcenter → Kategorie „Stromhandel“, "
              "„Kommerzieller Außenhandel“, Auflösung Stunde, je Jahr als CSV nach eingaben/smard_aussenhandel_<jahr>.csv.")
        return None
    spalten = teile[0].attrs.get("spalten", {})
    handel = pd.concat(teile).sort_index()
    handel = handel[~handel.index.duplicated(keep="last")]
    stunde = handel.resample("h").sum(min_count=1)
    stunde.attrs["spalten"] = spalten
    return stunde


def lade_pumpspeicher(jahre, quelle, cache: Path, df_basis: pd.DataFrame) -> pd.DataFrame | None:
    if quelle == "demo":
        r = df_basis["residuallast"]
        erz = ((r - r.quantile(0.7)).clip(lower=0) * 0.15).clip(upper=6000)
        verbr = ((r.quantile(0.3) - r).clip(lower=0) * 0.15).clip(upper=6000)
        return pd.DataFrame({"pump_erz": erz, "pump_verbr": verbr})
    if quelle != "api":
        return None
    reihen = {}
    for name, (fid, region) in PUMPSPEICHER.items():
        try:
            index = sa._hole_json(f"{sa.SMARD_BASIS}/{fid}/{region}/index_hour.json", None)
            start_ms = pd.Timestamp(f"{min(jahre)}-01-01", tz=sa.TZ).value // 10**6
            ende_ms = pd.Timestamp(f"{max(jahre) + 1}-01-01", tz=sa.TZ).value // 10**6
            punkte = []
            for ts in index["timestamps"]:
                if start_ms - 8 * 86_400_000 <= ts < ende_ms:
                    url = f"{sa.SMARD_BASIS}/{fid}/{region}/{fid}_{region}_hour_{ts}.json"
                    punkte.extend(sa._hole_json(url, cache / f"{fid}_{region}_hour_{ts}.json").get("series", []))
            zeit = pd.to_datetime([p[0] for p in punkte], unit="ms", utc=True).tz_convert(sa.TZ)
            s = pd.to_numeric(pd.Series([p[1] for p in punkte], index=zeit), errors="coerce")
            reihen[name] = s[~s.index.duplicated(keep="last")].sort_index()
        except Exception as fehler:  # noqa: BLE001
            print(f"  C7: Pumpspeicher-Reihe {name} nicht verfügbar ({fehler}).")
    return pd.DataFrame(reihen) if reihen else None


def modell_c7(df, kap, jahre, quelle, eingaben, cache, ziel, demo):
    handel = lade_handel(jahre, quelle, eingaben, cache, df)
    if handel is None:
        return
    pump = lade_pumpspeicher(jahre, quelle, cache, df)
    basis = df.join(handel[["nettoexport"] + [c for c in ("export", "import", "nettoexport_csv") if c in handel]], how="left")
    if pump is not None:
        basis = basis.join(pump, how="left")
    zeilen = []
    for jahr in jahre:
        d = basis.loc[str(jahr)]
        if d["nettoexport"].notna().sum() < 8000:
            continue
        top = d.nlargest(100, "residuallast")
        z = {"Jahr": jahr,
             "Stunden mit Handelsdaten": int(d["nettoexport"].notna().sum()),
             "Nettoexport Jahr [TWh]": d["nettoexport"].sum() / 1e6,
             "Nettoexport Mittel alle Stunden [GW]": d["nettoexport"].mean() / 1000,
             "Nettoimport Top-100-Residuallast [GW]": -top["nettoexport"].mean() / 1000,
             "Anteil Import an Top-100-Residuallast [%]": 100 * (-top["nettoexport"]).mean() / top["residuallast"].mean()}
        if d["preis"].notna().sum() > 8000:
            neg = d[d["preis"] < 0]
            z["Stunden mit negativem Preis"] = len(neg)
            z["Nettoexport in negativen Preisstunden [GW]"] = neg["nettoexport"].mean() / 1000 if len(neg) else np.nan
        if kap is not None:
            k = kap.reindex(d.index)
            cf = (d["pv"] + d["wind_on"] + d["wind_off"]) / (k["pv"] + k["wind_on"] + k["wind_off"])
            maske = pd.Series(False, index=d.index)
            for p in perioden(cf < DUNKEL_SCHWELLEN[0], DUNKEL_MINDESTDAUER_H[0]):
                maske.loc[p[0]:p[1]] = True
            if maske.any():
                z["Dunkelflaute-Stunden"] = int(maske.sum())
                z["Nettoimport in Dunkelflauten [GW]"] = -d.loc[maske, "nettoexport"].mean() / 1000
                z["Anteil Import an Residuallast in Dunkelflauten [%]"] = (
                    100 * (-d.loc[maske, "nettoexport"]).mean() / d.loc[maske, "residuallast"].mean())
        if "pump_erz" in d and d["pump_erz"].notna().sum() > 8000:
            z["Pumpspeicher Erzeugung [TWh]"] = d["pump_erz"].sum() / 1e6
            z["Pumpspeicher Top-100 [GW]"] = top["pump_erz"].mean() / 1000
            if "Stunden mit negativem Preis" in z and z["Stunden mit negativem Preis"]:
                z["Pumpspeicher Verbrauch neg. Preise [GW]"] = d.loc[d["preis"] < 0, "pump_verbr"].abs().mean() / 1000
        if "nettoexport_csv" in d and d["nettoexport_csv"].notna().sum() > 8000 and "export" in d:
            abw = (d["nettoexport"] - d["nettoexport_csv"]).abs().mean()
            plausi(f"C7 Netto = Export − Import ({jahr})", abw < 1.0, f"mittlere Abweichung {abw:.2f} MWh")
        zeilen.append(z)
    if not zeilen:
        print("  C7 übersprungen: keine vollständigen Handelsjahre.")
        return
    tab = pd.DataFrame(zeilen).set_index("Jahr")
    sa.csv_de(tab, ziel / "c7_kennzahlen.csv")
    plausi("C7 Handelsdaten je Jahr nahezu vollständig", bool((tab["Stunden mit Handelsdaten"] >= 8000).all()))

    lj = tab.index.max()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.33, 7.5), gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(tab))
    b = 0.27
    ax1.bar(x - b, tab["Nettoimport Top-100-Residuallast [GW]"], b, color=CI["navy"], label="Nettoimport in Top-100-Spitzenstunden")
    if "Nettoimport in Dunkelflauten [GW]" in tab:
        ax1.bar(x, tab["Nettoimport in Dunkelflauten [GW]"], b, color=CI["blau"], label="Nettoimport in Dunkelflauten")
    if "Nettoexport in negativen Preisstunden [GW]" in tab:
        ax1.bar(x + b, tab["Nettoexport in negativen Preisstunden [GW]"], b, color=CI["gold"],
                label="Nettoexport in negativen Preisstunden")
    ax1.axhline(0, color=CI["navy"], lw=0.8)
    ax1.set_xticks(x, [str(j) for j in tab.index])
    ax1.set_ylabel("GW (Mittel der jeweiligen Stunden)", color=CI["navy"])
    ax1.set_title("Austausch in kritischen Stunden", loc="left", fontsize=11, color=CI["navy"], weight="bold")
    ax1.legend(frameon=False, fontsize=8)
    ax2.bar([str(j) for j in tab.index], tab["Nettoexport Jahr [TWh]"],
            color=[CI["teal"] if v >= 0 else CI["rot"] for v in tab["Nettoexport Jahr [TWh]"]])
    ax2.axhline(0, color=CI["navy"], lw=0.8)
    ax2.set_ylabel("Nettoexport [TWh] (negativ = Nettoimport)", color=CI["navy"])
    ax2.set_title("Jahressaldo des Außenhandels", loc="left", fontsize=11, color=CI["navy"], weight="bold")
    ax2.tick_params(axis="x", rotation=45)
    for ax in (ax1, ax2):
        sa._stil(ax)
    t_l = tab.loc[lj]
    titel = (f"{lj}: In den 100 Spitzenstunden kamen im Mittel {de(t_l['Nettoimport Top-100-Residuallast [GW]'], 1)} GW "
             f"aus dem Ausland ({de(t_l['Anteil Import an Top-100-Residuallast [%]'], 0)} % der Residuallast)")
    rahmen(fig, titel, "Kommerzieller Außenhandel laut SMARD, stündlich · positive Balken links = Beitrag des Auslands "
                       "zur Deckung bzw. Abnahme · Dunkelflaute nach C3-Definition", demo)
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.07, right=0.98, wspace=0.25)
    speichern(fig, ziel / "28_aussenhandel_kritische_stunden.png")

    # Tagesprofil im letzten Jahr: Handel und Pumpspeicher als Flexibilität
    d = basis.loc[str(lj)]
    fig, ax = plt.subplots(figsize=(12, 6.75))
    h = d.groupby(d.index.hour)
    ax.plot(h["nettoexport"].mean().index, h["nettoexport"].mean() / 1000, "-o", color=CI["navy"], lw=2.2,
            label="Nettoexport (negativ = Import)")
    if "pump_erz" in d and d["pump_erz"].notna().any():
        ax.plot(h["pump_erz"].mean().index, h["pump_erz"].mean() / 1000, "-s", color=CI["teal"], lw=1.8,
                label="Pumpspeicher Erzeugung")
        ax.plot(h["pump_verbr"].mean().index, -h["pump_verbr"].mean().abs() / 1000, "-s", color=CI["gold"], lw=1.8,
                label="Pumpspeicher Verbrauch (negativ)")
    ax2 = ax.twinx()
    ax2.plot(h["residuallast"].mean().index, h["residuallast"].mean() / 1000, ls="--", color=CI["grau"], lw=1.2,
             label="Residuallast (rechte Achse)")
    ax2.set_ylabel("Residuallast [GW]", color=CI["grau"])
    ax2.spines["top"].set_visible(False)
    ax.axhline(0, color=CI["navy"], lw=0.8)
    sa._stil(ax)
    ax.set_xlabel("Stunde (Ortszeit)", color=CI["navy"])
    ax.set_ylabel("GW, Mittel je Stunde", color=CI["navy"])
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="lower left")
    rahmen(fig, f"{lj}: Mittags Export, abends Import – der Außenhandel folgt der Residuallast",
           "Mittleres Tagesprofil von Nettoexport, Pumpspeichern und Residuallast · Ausland und Pumpspeicher als "
           "heute verfügbare Flexibilität", demo)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.92)
    speichern(fig, ziel / "29_flexibilitaet_tagesprofil.png")

    sp = handel.attrs.get("spalten", {})
    methodik(ziel, "c7", [
        "C7 · Außenhandel und Pumpspeicher in kritischen Stunden",
        "Datenbasis: kommerzieller Außenhandel laut SMARD (stündlich, Summe aller Grenzen), Pumpspeicher-Erzeugung",
        "und -Verbrauch laut SMARD; Nettoexport = Export − Import.",
        f"Verwendete Export-Spalten: {len(sp.get('export', []))}, Import-Spalten: {len(sp.get('import', []))}, "
        f"Netto-Spalte vorhanden: {bool(sp.get('netto'))}.",
        "Kritische Stunden: 100 Stunden mit höchster Residuallast je Jahr; Stunden mit negativem Day-Ahead-Preis;",
        "Dunkelflauten nach C3 (Wind+PV < 10 % der installierten Leistung über ≥ 24 h).",
        "Kommerzielle Flüsse weichen von physikalischen Flüssen ab (Ringflüsse, Transit).",
        "Bis 30.09.2018 gehörte Österreich zur gemeinsamen Gebotszone; Handel mit AT ist davor nicht enthalten."])


# ===========================================================================
# C8 · Flexibilitätswert: Preisspreizung und PV-Anteil (Prüfung H3 mit Saisonkontrolle)
# ===========================================================================
def modell_c8(df, jahre, ziel, demo):
    zeilen = []
    for jahr in jahre:
        d = df.loc[str(jahr)].dropna(subset=["preis", "pv", "netzlast"])
        if len(d) < 8000:
            continue
        tage = d["preis"].groupby(d.index.date)
        spreiz_tag = (tage.max() - tage.min())
        spreiz_tag.index = pd.to_datetime(spreiz_tag.index)
        for mon, m in d.groupby(d.index.month):
            st = spreiz_tag[spreiz_tag.index.month == mon]
            zeilen.append({"Jahr": jahr, "Monat": mon,
                           "PV-Anteil [%]": 100 * m["pv"].sum() / m["netzlast"].sum(),
                           "Tagesspreizung [€/MWh]": st.mean(),
                           "Preisniveau [€/MWh]": m["preis"].mean()})
    if len(zeilen) < 24:
        print("  C8 übersprungen: zu wenige Monate mit Preisdaten.")
        return
    mt = pd.DataFrame(zeilen)
    jt = mt.groupby("Jahr").agg({"PV-Anteil [%]": "mean", "Tagesspreizung [€/MWh]": "mean", "Preisniveau [€/MWh]": "mean"})
    y = mt["Tagesspreizung [€/MWh]"].to_numpy()
    dum = pd.get_dummies(pd.Categorical(mt["Monat"], categories=range(1, 13)), drop_first=True).astype(float).to_numpy()
    pv = mt["PV-Anteil [%]"].to_numpy()
    niveau = mt["Preisniveau [€/MWh]"].to_numpy()

    def schaetze(idx, mit_niveau):
        spalten = [np.ones(len(idx)), pv[idx], dum[idx]]
        if mit_niveau:
            spalten.insert(2, niveau[idx])
        x = np.column_stack(spalten)
        k, *_ = np.linalg.lstsq(x, y[idx], rcond=None)
        return k, x

    alle = np.arange(len(mt))
    erg = {}
    for name, mit in (("Monatskontrollen", False), ("Monatskontrollen + Preisniveau", True)):
        k, x = schaetze(alle, mit)
        res = y - x @ k
        kov = (res @ res / max(len(y) - x.shape[1], 1)) * np.linalg.pinv(x.T @ x)
        r2 = 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean()))
        rng = np.random.default_rng(42)
        jahre_mt = mt["Jahr"].to_numpy()
        uniq = np.unique(jahre_mt)
        boot = []
        for _ in range(BOOTSTRAP_N):
            gew = rng.choice(uniq, len(uniq), replace=True)
            idx = np.concatenate([np.where(jahre_mt == j)[0] for j in gew])
            boot.append(schaetze(idx, mit)[0][1])
        inf8 = ols_inferenz(x, y, 1, NW_LAGS_MONAT)
        erg[name] = {"b [€/MWh je Prozentpunkt]": k[1], "Standardfehler": float(np.sqrt(kov[1, 1])),
                     "p OLS": inf8["p OLS"], "SE Newey-West": inf8["SE Newey-West"], "p Newey-West": inf8["p Newey-West"],
                     "n": inf8["n"],
                     "t": k[1] / np.sqrt(kov[1, 1]), "R²": r2,
                     "KI 2,5 % (Jahresblöcke)": np.percentile(boot, 2.5), "KI 97,5 % (Jahresblöcke)": np.percentile(boot, 97.5),
                     "Anteil b > 0 [%]": 100 * np.mean(np.array(boot) > 0)}
    tab = pd.DataFrame(erg).T
    tab.index.name = "Modell"
    sa.csv_de(tab, ziel / "c8_kennzahlen.csv")
    sa.csv_de(jt, ziel / "c8_jahreswerte.csv")
    sa.csv_de(mt.set_index(["Jahr", "Monat"]), ziel / "c8_monatswerte.csv")
    rho_j = sa._spearman(jt["PV-Anteil [%]"], jt["Tagesspreizung [€/MWh]"])
    rho_m = sa._spearman(mt["PV-Anteil [%]"], mt["Tagesspreizung [€/MWh]"])
    haupt = erg["Monatskontrollen + Preisniveau"]
    urteil = ("gestützt" if haupt["KI 2,5 % (Jahresblöcke)"] > 0 else
              "nicht eindeutig (Intervall schließt 0 ein)" if haupt["KI 97,5 % (Jahresblöcke)"] > 0 else "widerlegt")
    (ziel / "c8_ergebnis.txt").write_text("\n".join([
        "C8 · Flexibilitätswert (H3): Tagesspreizung gegen PV-Anteil",
        f"Spearman ρ Jahreswerte (n = {len(jt)}): {rho_j:+.2f}; Monatswerte (n = {len(mt)}): {rho_m:+.2f}",
        *[f"{n}: b = {e['b [€/MWh je Prozentpunkt]']:+.2f} €/MWh je Prozentpunkt PV (t = {e['t']:.1f}, R² = {e['R²']:.2f}), "
          f"95-%-KI Jahresblöcke [{e['KI 2,5 % (Jahresblöcke)']:+.2f}; {e['KI 97,5 % (Jahresblöcke)']:+.2f}]" for n, e in erg.items()],
        f"Urteil H3 (mit Preisniveau als Kontrolle): {urteil}",
        "Hinweis: Das Preisniveau kontrolliert den Einfluss von Brennstoffpreisen (u. a. Krise 2022) auf die Spreizung."]) + "\n",
        encoding="utf-8")
    plausi("C8 Monatswerte vollständig (12 je Jahr)", bool((mt.groupby("Jahr").size() == 12).all()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.33, 7.5))
    jahre_u = sorted(mt["Jahr"].unique())
    for jahr, farbe in zip(jahre_u, farben_fuer(len(jahre_u))):
        t = mt[mt["Jahr"] == jahr]
        ax1.scatter(t["PV-Anteil [%]"], t["Tagesspreizung [€/MWh]"], color=farbe if jahr != 2022 else CI["rot"], s=38,
                    label=str(jahr))
    ax1.set_xlabel("PV-Anteil an der Netzlast im Monat [%]", color=CI["navy"])
    ax1.set_ylabel("Mittlere tägliche Preisspreizung [€/MWh]", color=CI["navy"])
    ax1.set_title(f"Monatswerte (ρ = {de(rho_m, 2)})", loc="left", fontsize=11, color=CI["navy"], weight="bold")
    ax1.legend(frameon=False, fontsize=8, ncol=2)
    ax2.plot(jt.index, jt["Tagesspreizung [€/MWh]"], "-o", color=CI["navy"], lw=2, label="Tagesspreizung")
    ax2b = ax2.twinx()
    ax2b.plot(jt.index, jt["PV-Anteil [%]"], "-s", color=CI["gold"], lw=1.8, label="PV-Anteil")
    ax2b.set_ylabel("PV-Anteil an der Netzlast [%]", color=CI["gold"])
    ax2b.spines["top"].set_visible(False)
    ax2.set_ylabel("Mittlere tägliche Preisspreizung [€/MWh]", color=CI["navy"])
    ax2.set_title("Jahreswerte", loc="left", fontsize=11, color=CI["navy"], weight="bold")
    ax2.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    for ax in (ax1, ax2):
        sa._stil(ax)
    b_h = haupt['b [€/MWh je Prozentpunkt]']
    rahmen(fig, f"H3 {urteil}: {'+' if b_h >= 0 else ''}{de(b_h, 1)} €/MWh Tagesspreizung je Prozentpunkt PV-Anteil",
           f"Regression mit Monatskontrollen und Preisniveau über {len(mt)} Monatswerte, 95-%-KI (Jahresblöcke) "
           f"{de(haupt['KI 2,5 % (Jahresblöcke)'], 1)} bis {de(haupt['KI 97,5 % (Jahresblöcke)'], 1)}", demo)
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.07, right=0.93, wspace=0.3)
    speichern(fig, ziel / "30_flexibilitaetswert_h3.png")
    methodik(ziel, "c8", [
        "C8 · Flexibilitätswert (Prüfung H3)",
        "Tagesspreizung = Maximum minus Minimum der stündlichen Day-Ahead-Preise eines Tages; Monatsmittel.",
        "Regression: Spreizung_m = a + b · PV-Anteil_m (+ c · Preisniveau_m) + Monats-Dummies; OLS mit numpy.",
        f"Konfidenzintervall: Blockbootstrap mit Jahren als Ziehungseinheit, {BOOTSTRAP_N} Wiederholungen.",
        "Das Preisniveau kontrolliert Brennstoff- und CO2-Preiseffekte, die die Spreizung unabhängig vom PV-Anteil erhöhen."])


# ===========================================================================
# Hauptprogramm
# ===========================================================================
MODULE = ["szenario2030", "spitzen", "spitzen_robust", "netzlast", "dunkelflaute", "batterie", "meritorder",
          "kannibalisierung", "flexwert", "aussenhandel", "strompreisreal", "europa", "hpc", "heimspeicher", "kommune"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Systemmodelle · THE PATTERN MIND")
    ap.add_argument("--module", nargs="+", required=True, choices=MODULE + ["alle"])
    ap.add_argument("--quelle", choices=["api", "csv", "demo"], default="api")
    ap.add_argument("--csv", nargs="*", default=[])
    ap.add_argument("--jahre", nargs="+", type=int, default=list(range(2015, 2026)),
                    help="Zeitreihe für C3, C5, C6 (preisbasierte Module nutzen nur Jahre mit Preisdaten)")
    ap.add_argument("--spitzenjahre", nargs="+", type=int, default=[2015, 2020, 2025])
    ap.add_argument("--basisjahre", nargs="+", type=int, default=[2023, 2024, 2025])
    ap.add_argument("--batteriejahre", nargs="+", type=int, default=[2024, 2025])
    ap.add_argument("--gesicherte-leistung", nargs="+", type=float, default=[40, 50, 60, 70])
    ap.add_argument("--speicherdauer", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--ausgabe", default="systemmodelle_ergebnisse")
    ap.add_argument("--eingaben", default="eingaben")
    ap.add_argument("--cache", default=".smard_cache")
    a = ap.parse_args()
    module = MODULE if "alle" in a.module else a.module
    demo = a.quelle == "demo"
    ziel, eingaben = Path(a.ausgabe), Path(a.eingaben)
    ziel.mkdir(parents=True, exist_ok=True)

    braucht_smard = set(module) & {"szenario2030", "spitzen", "spitzen_robust", "netzlast", "dunkelflaute", "batterie",
                                   "meritorder", "kannibalisierung", "flexwert", "aussenhandel", "heimspeicher", "kommune"}
    df = kap = None
    if braucht_smard:
        jahre = sorted(set(a.jahre) | set(a.spitzenjahre) | set(a.basisjahre) | set(a.batteriejahre))
        if a.quelle == "csv" and not a.csv:
            ap.error("--quelle csv benötigt --csv <dateien>")
        print(f"Datenquelle: {a.quelle} · Jahre {min(jahre)}–{max(jahre)}")
        df = lade_daten(a.quelle, jahre, a.csv, Path(a.cache))
        if set(module) & {"szenario2030", "spitzen", "dunkelflaute", "kommune", "aussenhandel"}:
            kap = lade_kapazitaet(a.quelle, df, eingaben)

    r2030 = None
    schritte = [
        ("szenario2030", lambda: modell_c1(df, kap, sorted(a.basisjahre), ziel, demo)),
        ("spitzen", lambda: modell_c2(df, kap, sorted(a.spitzenjahre), ziel, demo)),
        ("spitzen_robust", lambda: modell_r1(df, sorted(a.jahre), ziel, demo)),
        ("netzlast", lambda: modell_r4(df, sorted(a.jahre), eingaben, ziel, demo)),
        ("dunkelflaute", lambda: modell_c3(df, kap, sorted(a.jahre), [int(f) if f.is_integer() else f for f in a.gesicherte_leistung],
                                           r2030, max(a.basisjahre), ziel, demo)),
        ("batterie", lambda: modell_c4(df, sorted(a.batteriejahre), sorted(a.speicherdauer), ziel, demo)),
        ("meritorder", lambda: modell_c5(df, sorted(a.jahre), ziel, demo)),
        ("kannibalisierung", lambda: modell_c6(df, sorted(a.jahre), ziel, demo)),
        ("flexwert", lambda: modell_c8(df, sorted(a.jahre), ziel, demo)),
        ("aussenhandel", lambda: modell_c7(df, kap, sorted(a.jahre), a.quelle, eingaben, Path(a.cache), ziel, demo)),
        ("strompreisreal", lambda: modell_d1(eingaben, ziel, demo)),
        ("europa", lambda: modell_d2(eingaben, ziel, demo)),
        ("hpc", lambda: modell_d3(ziel, demo)),
        ("heimspeicher", lambda: modell_d4(df, ziel, demo)),
        ("kommune", lambda: modell_d5(df, kap, ziel)),
    ]
    for name, f in schritte:
        if name not in module:
            continue
        print(f"→ {name}")
        try:
            erg = f()
        except Exception as fehler:  # noqa: BLE001 – ein Modul darf den Gesamtlauf nicht abbrechen
            import traceback
            plausi(f"Modul {name} abgebrochen", False, f"{type(fehler).__name__}: {fehler}")
            (ziel / f"fehler_{name}.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"  FEHLER in {name}: {fehler} – Details in fehler_{name}.txt, weiter mit nächstem Modul")
            continue
        if name == "szenario2030":
            r2030 = erg
    (ziel / "plausibilitaet.txt").write_text("\n".join(_PLAUSI) + "\n", encoding="utf-8")
    print("\nPlausibilitätstests:")
    for z in _PLAUSI:
        print("  " + z)
    print(f"\nErgebnisse in: {ziel.resolve()}")
    return 0 if all(z.startswith("[OK") for z in _PLAUSI) else 2


if __name__ == "__main__":
    sys.exit(main())
