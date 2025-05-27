def compute_model_complexity(*net):

    """
    Calcola numero di posti, transizioni, archi e Indice Extend Cardoso.

    :param index: indice del modello di cui calcolare la complessità
    :return: numero di posti, numero di transizioni, numero di archi e metrica "Extended Cardoso" del modello
    """
    net = net[0]
    ext_card = 0
    for place in net.places:
        successor_places = set()
        for place_arc in place.out_arcs:
            successors = frozenset(transition_arc.target for transition_arc in place_arc.target.out_arcs)
            successor_places.add(successors)
        ext_card += len(successor_places)

    return len(net.places), len(net.transitions), len(net.arcs), ext_card
