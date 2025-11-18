from __future__ import annotations

import time
import heapq
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .state import PuzzleState, GOAL


# ---------------------------- Result structure -------------------------------

@dataclass
class SearchResult:
    # Wurde eine Lösung gefunden? (True/False)
    solved: bool
    # Tiefe der Lösung (Anzahl der Züge vom Start bis zum Ziel)
    depth: int
    # Wie viele Zustände wurden im Suchprozess tatsächlich erweitert?
    expanded: int
    #  Laufzeit der Suche in Sekunden
    runtime_s: float
    # Name der verwendeten Heuristik (Hamming / Manhattan)
    heuristic: str
    # Startzustand, von dem aus die Suche gestartet wird (als tupel)
    start_state: Tuple[int, ...]
    # kompletter Pfad von Start bis Ziel (Liste von PuzzleStates)
    path: Optional[List[PuzzleState]] = field(default=None)


# ------------------------------- A* search -----------------------------------

def a_star(start: PuzzleState, h: Callable[[PuzzleState], int]) -> SearchResult:
    """
   Führt den A* Suchalgorithmus aus, um den kürzesten Weg zum
   Zielzustand des 8-Puzzles zu finden. Nutzt die übergebenen Heuristikfunktionen h.
   Gibt ein SearchResult mit allen relevanten Such-Informationen zurück.
    """
    # Startzeit für Laufzeitmessung
    t0 = time.perf_counter()
    # Name der verwendeten Heuristik extrahieren
    heuristic_name = h.__name__.capitalize()

    # Priority Queue (Open-List): speichert Zustände sortiert nach f = g + h
    open_heap: List[Tuple[int, int, int, PuzzleState]] = []  # (f, g, tie, state)
    # g_score: bisher bekannte beste Kosten vom Start zu einem Zustand
    g_score: Dict[PuzzleState, int] = {start: 0}
    # came_from: merkt sich, von welchem Zustand man gekommen ist für die Pfadrekontruktion
    came_from: Dict[PuzzleState, Optional[PuzzleState]] = {start: None}

    expanded_nodes = 0      # Zählt, wie viele Zustände tatsächlich erweitert wurden
    tie_counter = 0         # Tie-breaker für Heap, falls f-Werte gleich sind

    # Startzustand in Priority Queue einfügen
    f0 = h(start)   # f = g(=0) + h(start)
    heapq.heappush(open_heap, (f0, 0, tie_counter, start))
    # Closed-List: Zustände, die vollständig verarbeitet wurden
    closed: set[PuzzleState] = set()

    # Hauptschleife
    while open_heap:
        # Besten zustand (kleinstes f) entnehmen
        f, g, _, current = heapq.heappop(open_heap)

        # wenn Zustand bereits verarbeitet wurde --> überspringen
        if current in closed:
            continue

        # Zustand als abgeschlossen markieren
        closed.add(current)
        expanded_nodes += 1  # measure memory effort

        # Zieltest: Ist der aktuelle Zustand das Ziel?
        if current.is_goal():
            t1 = time.perf_counter()

            # Pfad vom Start zur Lösung rekonstruieren
            path = _reconstruct_path(came_from, current)

            # ergebnisobjekt zurückgeben
            return SearchResult(
                solved=True,
                depth=len(path) - 1,
                expanded=expanded_nodes,
                runtime_s=t1 - t0,
                heuristic=heuristic_name,
                start_state=start.tiles,
                path=path,
            )

        # Alle nachbarn (Folgezustände) des aktuellen Zustands durchgehen
        for neighbor, action, cost in current.neighbors():
            tentative_g = g + cost  # neue Kostenberechnung

            # Bereits abgeschlossene Zustände ignorieren
            if neighbor in closed:
                continue

            # wenn noch kein g-Wert existiert oder der neue Weg besser ist
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current   # Vorgänger speichern
                tie_counter += 1

                # f = neuer g-Wert + Heuristik
                f_val = tentative_g + h(neighbor)
                # Nachbar in Priority Queue einfügen
                heapq.heappush(open_heap, (f_val, tentative_g, tie_counter, neighbor))

    # Falls kein Ziel gefunden wurde: Ergebnis mit solved=False zurückgeben
    t1 = time.perf_counter()
    return SearchResult(
        solved=False,
        depth=0,
        expanded=expanded_nodes,
        runtime_s=t1 - t0,
        heuristic=heuristic_name,
        start_state=start.tiles,
        path=None,
    )


# ------------------------------ Path recovery --------------------------------

def _reconstruct_path(
    came_from: Dict[PuzzleState, Optional[PuzzleState]],
    goal_state: PuzzleState,
) -> List[PuzzleState]:
    """
    Rekonstruiert den vollständigen Lösungsweg vom Ziel zurück zum Start,
    indem die came_from-Verkettung rückwärts verfolgt wird.
    Gibt eine Liste von PuzzleStates von Start --> Ziel zurück
    """
    path: List[PuzzleState] = [goal_state]
    # Vom Ziel aus so lange zurückgehen, bis der Start erreicht ist
    while came_from[path[-1]] is not None:
        path.append(came_from[path[-1]])

    # Liste umdrehen, damit sie vom Start zum ziel läuft
    path.reverse()
    return path


# ------------------------------- Self-test -----------------------------------

if __name__ == "__main__":
    from .heuristics import manhattan, hamming

    start = PuzzleState((1, 2, 3, 4, 5, 6, 0, 7, 8))
    for fn in (hamming, manhattan):
        print(f"Running A* with {fn.__name__}")
        res = a_star(start, fn)
        print(f"Solved={res.solved}, depth={res.depth}, expanded={res.expanded}, time={res.runtime_s:.4f}s")