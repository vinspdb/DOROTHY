from datetime import datetime, timedelta
import pandas as pd

def reduce_consecutive_duplicates(lst, timelst):
    """
    Riduce le occorrenze consecutive di un elemento in una lista a massimo 2.
    :param lst: Lista di elementi da filtrare
    :param timelst: Lista di timestamp corrispondenti agli elementi
    :return: Lista filtrata e lista di timestamp corrispondenti
    """
    result = []
    timestamps = []
    count = 0
    
    for i in range(len(lst)):
        # If the current element is the same as the last one, increment the count
        if result and lst[i] == result[-1]:
            count += 1
        else:
            count = 1  # Reset the count if the element changes
        
        # Add the element only if the count is 2 or less
        if count <= 2:
            result.append(lst[i])
            timestamps.append(timelst[i])

    return result, timestamps

def filterlog(event_log, activity_column):
    """
    Filtra l'event log per ridurre le occorrenze consecutive di una stessa attività nel log a massimo 2.
    :param event_log: DataFrame dell'event log  
    :param activity_column: Nome della colonna contenente le attività
    :return: DataFrame filtrato
    """
    filtered_log = {"case:concept:name":[],"concept:name":[],"time:timestamp":[]}

    for name, group in event_log:
        new_val, timestamps = reduce_consecutive_duplicates(group[activity_column].to_list(), group["time:timestamp"].to_list())
        filtered_log["case:concept:name"].extend([name]*len(new_val))
        filtered_log["concept:name"].extend(new_val)
        filtered_log['time:timestamp'].extend(timestamps)
    return pd.DataFrame(filtered_log)

def get_traces(event_log):
    """
    Estrae le tracce dall'eventlog producendo un dataframe con 2 colonne, una contenente l'id del trace e una contenente il trace.
    :param event_log: DataFrame dell'event log
    :return: DataFrame contenente le tracce
    """
    activity_traces = {"case:concept:name": [], "traces": []}
    
    for name, group in event_log:
        trace = " -> ".join(group["concept:name"].tolist())
        activity_traces["case:concept:name"].append(name)
        activity_traces["traces"].append(trace)

    return pd.DataFrame(activity_traces)

def get_variant_traces(event_log):
    """
    Estrae le tracce dall'eventlog producendo un dataframe con 3 colonne, una contenente l'id del trace, una contenente il trace e una contenente la frequenza del trace nel dataframe.
    :param event_log: DataFrame dell'event log
    :return: DataFrame contenente le tracce e la loro frequenza
    """
    activity_traces = {"case:concept:name": [], "traces": [], "frequency": []}
    
    for name, group in event_log:
        activity_traces["traces"].append(name)
        activity_traces["frequency"].append(len(group))
        activity_traces["case:concept:name"].append(group["case:concept:name"].iloc[0])

    return pd.DataFrame(activity_traces)

def get_df_redundant(event_log, traces_log):
    """
    Crea un DataFrame ridondante per rappresentare la frequenza delle varianti a partire da un event log e da un DataFrame di tracce delle varianti.
    :param event_log: DataFrame dell'event log
    :param traces_log: DataFrame delle tracce
    :return: DataFrame ridondante
    """
    event_log_redundant = {"case:concept:name": [], "concept:name": [], "time:timestamp": []}

    for _, trace_row in traces_log.iterrows():
        name = trace_row["case:concept:name"]
        frequency = trace_row["frequency"]
        group = event_log.get_group(name)
        for i in range(frequency):
            event_log_redundant["case:concept:name"].extend([name + "_" + str(i)] * len(group))
            event_log_redundant["concept:name"].extend(group["concept:name"].tolist())
            event_log_redundant["time:timestamp"].extend(group["time:timestamp"].tolist())

    return pd.DataFrame(event_log_redundant)

def get_df_non_redundant(event_log, traces_log):
    """
    Crea un DataFrame non ridondante per rappresentare le varianti a partire da un event log e da un DataFrame di tracce delle varianti.
    :param event_log: DataFrame dell'event log
    :param traces_log: DataFrame delle tracce
    :return: DataFrame non ridondante
    """
    event_non_log_redundant = {"case:concept:name": [], "concept:name": [], "time:timestamp": []}

    for _, trace_row in traces_log.iterrows():
        name = trace_row["case:concept:name"]
        group = event_log.get_group(name)
        event_non_log_redundant["case:concept:name"].extend([name] * len(group))
        event_non_log_redundant["concept:name"].extend(group["concept:name"].tolist())
        event_non_log_redundant["time:timestamp"].extend(group["time:timestamp"].tolist())

    return pd.DataFrame(event_non_log_redundant)