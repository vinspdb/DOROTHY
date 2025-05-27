import pm4py
from pm4py.visualization.petri_net import visualizer
from pm4py.objects.conversion.bpmn import converter as bpmn_converter
import SM_exe as SM
import subprocess
from pm4py import read_bpmn, read_pnml
from os import path
def discover_process_model(event_log, treshold = 0.0, alpha = 1.0, algorithm = "inductive"):
    """
    Scopre il modello di processo a partire da un event log.
    
    :param event_log: DataFrame dell'event log
    :param treshold: soglia per il miner induttivo
    :param alpha: soglia per il miner ilp
    :param algorithm: algoritmo da utilizzare per il mining del modello di processo
    :return: modello di processo
    """
    if algorithm == "im":
        net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(event_log, noise_threshold = treshold)
    elif algorithm == "ilp":
        net, initial_marking, final_marking = pm4py.discover_petri_net_ilp(event_log, alpha = alpha)
    elif algorithm == "sm":
        print('split miner')
        pm4py.write_xes(event_log, 'sm_temp.xes', case_id_key='case:concept:name')
        model_path = 'model.bpmn'
        script = path.join('scripts', 'run.sh')
        args = (script, "SPL", str(0.4), 'sm_temp.xes', "model", str(0.0), str(0.1))
        subprocess.call(args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        net, initial_marking, final_marking = bpmn_converter.apply(read_bpmn(model_path))
    return net, initial_marking, final_marking

def visualize_process_model(net, initial_marking, final_marking, img_name = "img", dir = "."):
    """
    Visualizza il modello di processo scoperto e lo salva in formato png.

    :param net: modello di processo
    :param initial_marking: marcatura iniziale
    :param final_marking: marcatura finale
    :param img_name: nome dell'immagine
    :param dir: directory in cui salvare l'immagine
    """
    pm4py.view_petri_net(net, initial_marking, final_marking)
    save_process_model(net, initial_marking, final_marking, img_name, dir)

def save_process_model(net, initial_marking, final_marking, img_name="img", dir = "."):
    """
    Salva il modello di processo in formato png.

    :param net: modello di processo
    :param initial_marking: marcatura iniziale
    :param final_marking: marcatura finale
    :param dir: directory in cui salvare il modello di processo
    """
    gviz = pm4py.visualization.petri_net.visualizer.apply(net, initial_marking, final_marking, parameters={"format": "png"})
    visualizer.save(gviz, f"{dir}/petri_net_{img_name}.png")