import pandas as pd
import pm4py
import re
from datetime import datetime
import sys
import time

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

def conformance_iterative(miner, log_sampling):
    f, cardoso = process_model(miner, log_sampling)

    return f, cardoso

def process_model(model_type, log_sampling):
    if model_type == 'ilp':
        net = pm4py.discover_petri_net_ilp(
            log_sampling,
            activity_key='concept:name',
            case_id_key='case:concept:name',
            timestamp_key='time:timestamp'
        )
    elif model_type == 'im':
        net = pm4py.discover_petri_net_inductive(
            log_sampling,
            activity_key='concept:name',
            case_id_key='case:concept:name',
            timestamp_key='time:timestamp'
        )
    else:
        print(None)


    fitness = pm4py.conformance.fitness_alignments(full_log, *net)
    precision = pm4py.conformance.precision_alignments(full_log, *net)
    fmeasure = 2 * ((round((precision), 2) * round((fitness['averageFitness']), 2)) / (
                                round((precision), 2) + round((fitness['averageFitness']), 2)))
    print("FMEASURE %.2f" % fmeasure)
    print("Precision %.2f" % precision)
    print("Fitness %.2f" % fitness['averageFitness'])

    places, transitions, arcs, ext_card = compute_model_complexity([*net])
    return fmeasure, ext_card

def create_log(sequences):
    list_case = []
    list_act = []
    list_time = []
    i = 0
    for sequence in sequences:
        for s in sequence:
            current_datetime = datetime.now()
            # Convert to string
            #datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')
            list_case.append('case'+str(i))
            list_act.append(s)
            list_time.append(current_datetime)
        i = i + 1
    log = pd.DataFrame({'case:concept:name': list_case, 'concept:name': list_act, 'time:timestamp': list_time})
    return log


log_name = sys.argv[1]
miner = sys.argv[2]

# Load the event log from an XES file
try:
    full_log = pm4py.read_xes(f'event_log/TKDE_Benchmark-2/{log_name}.xes')
except:
    full_log = pd.read_csv(f'event_log/TKDE_Benchmark-2/{log_name}.csv')
    full_log['time:timestamp'] = pd.to_datetime(full_log['time:timestamp'])
    full_log['case:concept:name'] = full_log['case:concept:name'].astype(str)
full_log = full_log.dropna(subset=['concept:name'])

#activity_names = sorted(full_log["concept:name"].unique())  # Sorting for consistency

variants = pm4py.get_variants(
    full_log,
    activity_key='concept:name',
    case_id_key='case:concept:name',
    timestamp_key='time:timestamp'
)
sorted_variants = sorted(variants.items(), key=lambda x: x[1], reverse=True)

file_res_conf = open(f'conformance/sampling_{log_name}_{miner}.csv','w')
file_res_conf.write('Encoder,Fmeasure,ExtCardoso,Time,N.of.Variants,Variants\n')

list_variant = []
global_f = 0
start = time.time()
for line in sorted_variants:

        #pattern = r'(?<=Variant)(.*?)(?=is frequent)'
        # Find all matches
        #print(line)
        #matches = re.findall(pattern, line)
        #new_list = [inverse_mapping[element] for element in matches[0].strip().split()]
        #print(new_list)
        list_variant.append(list(line[0]))
        log_sampling = create_log(list_variant)
        f, cardoso = conformance_iterative(miner, log_sampling)
        if f>global_f:
            global_f = f
            best_cardoso = cardoso
            len_variant = len(list_variant)
            best_variant = ''
            for v in list_variant:
                best_variant = best_variant + ' $$ '  + ','.join(v)
        else:
            break

end_time = time.time()
comp_time = end_time - start

file_res_conf.write(f'sampling,{round((global_f), 2)},{cardoso},{comp_time},{len_variant},{best_variant}\n')