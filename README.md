# Energiesystem Deutschland 2015–2025

Programme, Eingabedaten und Ergebnisse zur Arbeit **„Vom Mengenziel zur Systemfrage. Kosten- und Systemwirkungen der deutschen Stromwende 2015–2025“** von Thomas Marcus Köhler (THE PATTERN MIND, Leipzig).

*English summary:* This repository contains the code, input data and results of an empirical analysis of Germany's power system based on hourly open data from the Federal Network Agency (SMARD), 2015–2025. Eight system models and four robustness checks cover peak residual load, dark doldrums, battery economics, merit order, solar cannibalisation, the value of flexibility, cross-border trade and a 2030 scenario.

## Inhalt

| Datei / Ordner | Inhalt |
| --- | --- |
| `smard_analyse.py` | Datenabruf über die SMARD-Schnittstelle mit Zwischenspeicher, Aufbereitung, Validierung an Referenzwerten |
| `systemmodelle.py` | Modelle C1–C8, Robustheitsprüfungen R1–R4, Anwendungsfälle D1–D5, Grafiken, Kennzahlen, Plausibilitätstests |
| `eingaben/` | Ergänzende Jahreswerte: installierte Leistung (UBA/AGEE-Stat), Bruttostromverbrauch (BDEW/AGEB), Strompreisbestandteile (BDEW), Verbraucherpreisindex (Destatis) |
| `ergebnisse/` | Alle Grafiken, Kennzahlentabellen, Regressionstabellen, Methodennotizen und das Protokoll der Plausibilitätstests des finalen Laufs |

## Reproduktion

Voraussetzung: Python 3.10 oder neuer und eine Internetverbindung für den ersten Lauf.

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python systemmodelle.py --module alle
```

Der erste Lauf lädt die SMARD-Zeitreihen 2015–2025 und dauert je nach Verbindung 20 bis 50 Minuten; weitere Läufe nutzen den lokalen Zwischenspeicher. Ohne Internet zeigt `python systemmodelle.py --module alle --quelle demo` die Funktionsweise mit synthetischen, als Demodaten gekennzeichneten Werten.

Alle Annahmen stehen gebündelt im Konfigurationsblock am Anfang von `systemmodelle.py`, jeweils mit Quellenangabe. Jeder Lauf prüft sich mit eingebauten Plausibilitätstests selbst; das Protokoll steht in `ergebnisse/plausibilitaet.txt`.

## Methodische Hinweise

- Regressionen sind Schätzungen in reduzierter Form; Standardfehler werden zusätzlich nach Newey-West (autokorrelationsrobust) ausgewiesen, Konfidenzintervalle per Blockbootstrap.
- Szenario- und Investitionsrechnungen sind bedingte Aussagen auf Basis der ausgewiesenen Annahmen.
- Die Modelle betrachten Deutschland ohne Netzengpässe und berücksichtigen nur den Day-Ahead-Markt.

## Datenquellen

Strommarktdaten: Bundesnetzagentur | SMARD.de. Ergänzende Daten: UBA/AGEE-Stat, AG Energiebilanzen/BDEW, BDEW-Strompreisanalyse, Destatis, Eurostat. Die Nutzungsbedingungen der jeweiligen Datenquellen gelten; die Quellenangabe „Bundesnetzagentur | SMARD.de“ ist bei Weiterverwendung der SMARD-Daten anzugeben.

## Lizenz und Zitation

Programmcode: MIT-Lizenz (siehe `LICENSE`). Zitationsangaben stehen in `CITATION.cff`.

## Hinweis zur Entstehung

Die Programme wurden KI-gestützt mit Claude (Anthropic) entwickelt. Fragestellung, Analysedesign, Prüfung und Bewertung der Ergebnisse liegen beim Autor.
