import cardoso
import conformance_checking
import time
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("model too complex!")

def init_metrics_dict():
    return {
        "Dataset": [], "Miner": [], "DataFrame": [], 
        "fitness": [], "precision": [], "f1": [], 
        "Posti": [], "Transizioni":[], "Archi": [], 
        "ExtCardoso":[], "Tempo mining":[], "Tempo conformance":[]
    }

def init_time_dict():
    return {
        "Nome misurazione": [], "Tempo (in secondi)": []
    }

def compute_metrics(metrics, eventlog, net, im, fm, discover_time, dataset = "set", dataframe_type = "filtered", miner="inductive"):
    """
    Calcola le metriche del modello di processo rispetto all'event log.

    :param eventlog: DataFrame dell'event log
    :param net: modello di processo
    :param im: marcatura iniziale
    :param fm: marcatura finale
    :param discover_time: tempo di mining del modello di processo
    :param dataframe_type: tipo di DataFrame (default: "filtered")
    :param miner: miner utilizzato (default: "inductive")
    :return: dizionario con le metriche calcolate
    """
    
    metrics["Tempo mining"].append(discover_time)
    metrics["Dataset"].append(dataset)
    metrics["Miner"].append(miner)
    metrics["DataFrame"].append(dataframe_type)
    
    start = time.time()
    seconds = 60*60
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
                        fitness_met = conformance_checking.check_fitness(eventlog, net, im, fm).get("averageFitness")
                        precision_met = conformance_checking.check_precision(eventlog, net, im, fm)
                        f1_met = conformance_checking.f1_score(precision_met, fitness_met)
    except TimeoutError as e:
                        print(f"Errore: {e}")
                        precision_met = 0
                        fitness_met = 0
                        f1_met = 0
    metrics["precision"].append(precision_met)
    metrics["fitness"].append(fitness_met)
    metrics["f1"].append(f1_met)
  

    cardoso_tuple = cardoso.compute_model_complexity(net)
    
    metrics["Posti"].append(cardoso_tuple[0])
    metrics["Transizioni"].append(cardoso_tuple[1])
    metrics["Archi"].append(cardoso_tuple[2])
    metrics["ExtCardoso"].append(cardoso_tuple[3])
    end = time.time()
    metrics["Tempo conformance"].append(end - start)
    #print(metrics)
    
    return metrics

def append_metrics(metrics, new_metrics, index = -1):
    """
    Aggiunge le metriche presenti nel dizionario new_metrics, in corrispondenza alla riga i,
      a metrics.

    :param metrics: dizionario con le metriche calcolate
    :param new_metrics: dizionario con le nuove metriche da aggiungere
    :param index: indice della riga in cui aggiungere le nuove metriche (default: -1)
    :return: dizionario con le metriche aggiornate  
    """

    metrics["Dataset"].append(new_metrics["Dataset"][index])
    metrics["Miner"].append(new_metrics["Miner"][index])
    metrics["DataFrame"].append(new_metrics["DataFrame"][index])
    
    metrics["fitness"].append(new_metrics["fitness"][index])
    metrics["precision"].append(new_metrics["precision"][index])
    metrics["f1"].append(new_metrics["f1"][index])

    metrics["Posti"].append(new_metrics["Posti"][index])
    metrics["Transizioni"].append(new_metrics["Transizioni"][index])
    metrics["Archi"].append(new_metrics["Archi"][index])
    metrics["ExtCardoso"].append(new_metrics["ExtCardoso"][index])
    metrics["Tempo mining"].append(new_metrics["Tempo mining"][index])
    metrics["Tempo conformance"].append(new_metrics["Tempo conformance"][index])
    
    return metrics

def add_time(time_metrics, label, value):
    """
    Aggiunge il tempo di esecuzione per una specifica misurazione.

    :param time_metrics: dizionario con le metriche temporali
    :param label: etichetta della misurazione
    :param value: valore del tempo in secondi
    :return: dizionario con le metriche temporali aggiornate
    """
    
    time_metrics["Nome misurazione"].append(label)
    time_metrics["Tempo (in secondi)"].append(value)
    
    return time_metrics

def save_metrics(metrics, csv_path, filename="metrics.xlsx"):
    """
    Salva le metriche in un file CSV.

    :param metrics: dizionario con le metriche calcolate
    :param csv_path: percorso del file CSV
    """
    import pandas as pd
    print(metrics)
    df = pd.DataFrame(metrics)
    df.to_excel(f"{csv_path}/{filename}", index=False)