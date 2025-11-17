from __future__ import annotations  # Erlaubt Typannotationen auch bei Klassen, die erst später definiert werden

import time  # Zum Messen der Ausführungszeit von Funktionen
from dataclasses import dataclass  # Ermöglicht das einfache Erstellen von Datencontainern
from typing import Any, Iterable, List, Tuple  # Typangaben für Lesbarkeit und Fehlervermeidung

import statistics as stats  # Für Mittelwert- und Standardabweichungsberechnung


# ------------------------- DATENSTRUKTUR: SearchResult -------------------------
#Speichert alle wichtigen Informationen eines einzelnen Suchdurchlaufs in einem klar strukturierten Objekt.

@dataclass
class SearchResult:
   

    solved: bool              # Gibt an, ob das Puzzle erfolgreich gelöst wurde (True/False)
    depth: int                # Tiefe der Lösung (Anzahl der Züge bis zum Ziel)
    expanded: int             # Anzahl der expandierten Zustände im Suchprozess
    runtime_s: float          # Laufzeit in Sekunden
    heuristic: str            # Name der verwendeten Heuristik (z. B. "manhattan" oder "hamming")
    start_state: tuple[int, ...]  # Startzustand des Puzzles (z. B. (1,2,3,4,5,6,7,8,0))


# ------------------------- FUNKTION: time_call -------------------------
#Misst, wie lange eine Funktion zur Ausführung braucht.

def time_call(fn, *args, **kwargs) -> Tuple[float, Any]:
   
    t0 = time.perf_counter()          # Startzeitpunkt hochpräzise messen
    out = fn(*args, **kwargs)         # Funktion ausführen und Ergebnis speichern
    t1 = time.perf_counter()          # Endzeitpunkt messen
    return (t1 - t0, out)             # Laufzeit + Ergebnis als Tupel zurückgeben


# ------------------------- FUNKTION: mean_std -------------------------
#Berechnet den Mittelwert (mean) und die Standardabweichung (std) einer Zahlenliste.
#Wird z. B. verwendet, um mehrere Durchläufe statistisch auszuwerten.


def mean_std(values: Iterable[float | int]) -> Tuple[float, float]:
    
    vals = list(values)                       # Werte in eine Liste umwandeln
    if not vals:                              # Prüfen, ob Liste leer ist
        raise ValueError("mean_std() requires at least one value")
    if len(vals) == 1:                        # Nur ein Wert → keine Abweichung möglich
        return float(vals[0]), 0.0              #std=0
    return float(stats.mean(vals)), float(stats.stdev(vals))  # Mittelwert & Standardabweichung zurückgeben


# ------------------------- SELBSTTEST (nur beim direkten Ausführen) -------------------------
if __name__ == "__main__":
    # Schneller Test, um zu prüfen, ob die Funktionen korrekt funktionieren

    # Beispiel 1: time_call misst die Laufzeit der sum()-Funktion
    dt, out = time_call(sum, [1, 2, 3])
    print(f"time={dt:.6f}s, out={out}")  # Gibt z. B. aus: time=0.000002s, out=6

    # Beispiel 2: mean_std berechnet Durchschnitt und Standardabweichung einer Liste
    m, s = mean_std([1, 2, 3, 4])
    print(f"mean={m}, std={s}")  # Erwartete Ausgabe: mean=2.5, std≈1.29
