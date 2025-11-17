from __future__ import annotations  # Ermöglicht Typannotationen, auch wenn Klassen später definiert werden
from typing import Tuple  # Wird verwendet, wenn Funktionen Tupel (z. B. (x, y)) zurückgeben

# Importiert wichtige Konstanten und Klassen aus dem state-Modul
# PuzzleState = beschreibt einen bestimmten Puzzle-Zustand
# GOAL = Zielzustand (z. B. (1,2,3,4,5,6,7,8,0))
# GOAL_POS = Dictionary, das jedem Stein seine Zielkoordinaten zuordnet
# INDEX_TO_RC = ordnet jedem Index (0–8) die passende (Zeile, Spalte)-Position zu
# N = Größe des Spielfelds (beim 8-Puzzle = 3)
from .state import PuzzleState, GOAL, GOAL_POS, INDEX_TO_RC, N


# ------------------------ HAMMING-HEURISTIK ------------------------
#Ziel:  helfen dem Suchalgorithmus abzuschätzen, wie weit ein aktueller Puzzle-Zustand noch vom Zielzustand entfernt ist.
#Idee: man zählt, wie viele steine nciht an ihrer richtigen Position liegen

def hamming(s: PuzzleState) -> int:

    tiles = s.tiles  # Zugriff auf die aktuelle Anordnung der Steine

    # Wir vergleichen jeden Stein mit seiner Position im Zielzustand
    # enumerate() liefert Index (Position im Array) und Wert (Steinnummer)
    # Wir zählen +1, wenn der Stein nicht an der richtigen Position ist UND nicht das leere Feld (0)
    return sum(1 for i, v in enumerate(tiles) if v != 0 and v != GOAL[i])


# ------------------------ MANHATTAN-HEURISTIK ------------------------

#Idee: misst, wie weit jeder Stein von seinem Platz entfernt ist.
def manhattan(s: PuzzleState) -> int:
   

    dist = 0  # Startwert der gesamten Distanz

    # Für jedes Feld im Puzzle: Index = Position, tile = Zahl des Steins
    for idx, tile in enumerate(s.tiles):
        if tile == 0:  # Leeres Feld überspringen
            continue

        # Aktuelle Zeilen-/Spaltenposition berechnen
        r, c = INDEX_TO_RC[idx]

        # Zielposition des Steins (gr = goal_row, gc = goal_col)
        gr, gc = GOAL_POS[tile]

        # Manhattan-Distanz = horizontale + vertikale Entfernung
        dist += abs(r - gr) + abs(c - gc)

    # Gesamt-Distanz zurückgeben
    return dist


# ------------------------ ZERO-HEURISTIK (KONTROLLWERT) ------------------------
def zero_heuristic(_: PuzzleState) -> int:
    """
    Gibt immer 0 zurück.
    Wird verwendet, um zu testen, wie der Algorithmus ohne Heuristik funktioniert
    (entspricht Uniform Cost Search).
    """
    return 0


# ------------------------ SELBSTTEST (nur beim direkten Ausführen) ------------------------
if __name__ == "__main__":

    # Zielzustand prüfen: Alle Steine richtig -> Hamming & Manhattan = 0
    g = PuzzleState(GOAL)
    assert hamming(g) == 0
    assert manhattan(g) == 0

    # Ein fast gelöstes Puzzle (nur eine Bewegung entfernt)
    # Der leere Platz (0) steht links von der '8'
    near = PuzzleState((1, 2, 3, 4, 5, 6, 7, 0, 8))

    # '8' ist falsch platziert -> Hamming = 1, Manhattan = 1
    assert hamming(near) == 1
    assert manhattan(near) == 1

    # Testzustand, leicht durchmischt
    s = PuzzleState((2, 8, 3, 1, 6, 4, 7, 0, 5))

    # Gibt die berechneten Heuristikwerte aus
    print("Hamming =", hamming(s))
    print("Manhattan =", manhattan(s))
