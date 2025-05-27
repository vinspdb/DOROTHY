import pandas as pd
import pm4py
import re
from datetime import datetime
import sys
import time
import signal
def compute_model_complexity(net):
    """
    Calcola la complessità del modello indicizzato
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

def timeout_handler(signum, frame):
    raise TimeoutError("La funzione ha impiegato troppo tempo")

def conformance_iterative(miner, log_sampling, thr):
    f, cardoso = process_model(miner, log_sampling, thr)

    return f, cardoso

def process_model(model_type, log_sampling, thr):
    if model_type == 'ilp':
        net = pm4py.discover_petri_net_ilp(
            log_sampling,
            activity_key='concept:name',
            case_id_key='case:concept:name',
            timestamp_key='time:timestamp',
            alpha=thr
        )
    elif model_type == 'im':
        net = pm4py.discover_petri_net_inductive(
            log_sampling,
            activity_key='concept:name',
            case_id_key='case:concept:name',
            timestamp_key='time:timestamp',
            noise_threshold=thr
        )
    else:
        print(None)
    
    seconds = 60*120
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        try:
            fitness = pm4py.conformance.fitness_alignments(full_log, *net)['averageFitness']
            precision = pm4py.conformance.precision_alignments(full_log, *net)
            fmeasure = 2 * ((round((precision), 2) * round((fitness), 2)) / (
                                    round((precision), 2) + round((fitness), 2)))
        except:
             print('not sound net')
             fitness = 0
             precision = 0
             fmeasure = 0

    except TimeoutError as e:
                        print(f"Errore: {e}")
                        fitness = 0
                        precision = 0
                        fmeasure = 0
    print("FMEASURE %.2f" % fmeasure)
    print("Precision %.2f" % precision)
    print("Fitness %.2f" % fitness)
    print("Filter levele %.2f" % thr)

    places, transitions, arcs, ext_card = compute_model_complexity([*net])
    return fmeasure, ext_card


log_name = sys.argv[1]
miner = sys.argv[2]

file_res_conf = open(f'conformance/filtering_{log_name}_{miner}.csv','w')
file_res_conf.write('Encoder,Fmeasure,ExtCardoso,Time,FilterValue\n')

# Load the event log from an XES file
try:
    full_log = pm4py.read_xes(f'event_log/TKDE_Benchmark-2/{log_name}.xes')
except:
    full_log = pd.read_csv(f'event_log/TKDE_Benchmark-2/{log_name}.csv')
    full_log['time:timestamp'] = pd.to_datetime(full_log['time:timestamp'])
    full_log['case:concept:name'] = full_log['case:concept:name'].astype(str)
full_log = full_log.dropna(subset=['concept:name'])
#activity_names = sorted(full_log["concept:name"].unique())  # Sorting for consistency

check = False
i = 0.2
global_f = 0
start = time.time()
best_filter = 0
best_thr = 0.0
print('Event log--->', log_name)
while check ==False:
    if i>1.0:
        check = True
    else:
        f, cardoso = conformance_iterative(miner, full_log, i)
        #gviz = pn_visualizer.apply(net, initial_marking, final_marking)
        #pn_visualizer.view(gviz)
        #print("TOTAL Time")
        if f>global_f:
            global_f = f
            best_cardoso = cardoso
            best_thr = i
    i = i + 0.04
end_time = time.time()
comp_time = end_time - start
file_res_conf.write(f'filtering,{round((global_f), 2)},{cardoso},{comp_time},{best_thr}\n')

