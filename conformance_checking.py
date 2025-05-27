import pm4py

def check_precision(event_log, net, initial_marking, final_marking):
    """
    Calcola la precisione del modello di processo rispetto all'event log.

    :param event_log: DataFrame dell'event log
    :param net: modello di processo
    :param initial_marking: marcatura iniziale
    :param final_marking: marcatura finale
    :return: precisione del modello di processo rispetto all'event log
    """
    return pm4py.conformance.precision_alignments(event_log, net, initial_marking, final_marking, multi_processing=False)

def check_fitness(event_log, net, initial_marking, final_marking):
    """
    Calcola la fitness del modello di processo rispetto all'event log.
    
    :param event_log: DataFrame dell'event log
    :param net: modello di processo
    :param initial_marking: marcatura iniziale
    :param final_marking: marcatura finale
    :return: Dizionario con i vari valori di fitness del modello di processo rispetto all'event log
    """
    return pm4py.conformance.fitness_alignments(event_log, net, initial_marking, final_marking, multi_processing=False)

def f1_score(precision, fitness):
    """
    Calcola il valore F1 a partire dalla precisione e dalla fitness.

    :param precision: precisione del modello di processo rispetto all'event log
    :param fitness: fitness del modello di processo rispetto all'event log
    :return: valore F1 del modello di processo rispetto all'event log
    """
    return 2 * (precision * fitness) / (precision + fitness) if (precision + fitness) != 0 else 0
