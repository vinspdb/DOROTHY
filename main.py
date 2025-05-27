import os
import time
import numpy as np
import pandas as pd
import gc

import pm4py
import discover
import embedding
import df_manipulation
import metrics_util
import sys
import argparse

dataset = sys.argv[1]
algorithm = sys.argv[2]


parser = argparse.ArgumentParser(description='Process Discovery using DOROTHY')

parser.add_argument('-event_log', type=str, help="Event log name")
parser.add_argument('-miner', type=str, help="Process Discovery Algorithm (ilp, im, sm)")

args = parser.parse_args()

dataset = args.event_log
n_layer = args.miner

this_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"..")
res_dir = os.path.join(this_dir, "results")
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

img_path = os.path.join(res_dir, f"img_{dataset}")
csv_path = os.path.join('results', f"csv_{dataset}")

if not os.path.exists(img_path):
    os.makedirs(img_path)
if not os.path.exists(csv_path):
    os.makedirs(csv_path)

time_metrics = metrics_util.init_time_dict()
start_pre = time.time()
# Lettura dell'event log e conversione in DataFrame
try:
    eventlog_df = pm4py.read_xes("event_log/"+ dataset + ".xes")
except:
    eventlog_df = pd.read_csv("event_log/"+ dataset + ".csv", sep=',')
    eventlog_df['case:concept:name'] = 'case'+ eventlog_df['case:concept:name'].astype(str)
    eventlog_df['time:timestamp'] = pd.to_datetime(eventlog_df['time:timestamp'])    
eventlog_df = eventlog_df.dropna(subset=['concept:name'])
unique_values = eventlog_df['concept:name'].unique()

# Step 2: Create mapping and reverse mapping
mapping = {value: idx for idx, value in enumerate(unique_values)}
reverse_mapping = {idx: value for value, idx in mapping.items()}

# Step 3: Apply mapping
eventlog_df = eventlog_df[["case:concept:name", "concept:name", "time:timestamp"]]

df_group = eventlog_df.groupby("case:concept:name", sort=False)
filtered_df = df_manipulation.filterlog(df_group, "concept:name")

df_group = filtered_df.groupby("case:concept:name", sort=False)
traces_df = df_manipulation.get_traces(df_group)

df_group = traces_df.groupby("traces", sort=False)
variant_traces_df = df_manipulation.get_variant_traces(traces_df.groupby("traces", sort=False))

time_embedding = time.time()
variant_traces_embeddings = embedding.get_variants_embeddings(variant_traces_df["traces"].to_list())

metrics_util.add_time(time_metrics, "Tempo embedding", time.time() - time_embedding)
start_kmeans = time.time()
best_kmeans = embedding.run_kmeans_elbow(variant_traces_embeddings)
variant_traces_df["cluster"] = best_kmeans.labels_
metrics_util.add_time(time_metrics, "Tempo kmeans", time.time() - start_kmeans)

medoids = embedding.get_medoid_df(variant_traces_df, variant_traces_embeddings, best_kmeans)

medoids.to_csv(f'{csv_path}/_{dataset}_filtered_eventlog_variant_traces_cluster.csv', index=False)

metrics_util.add_time(time_metrics, "Tempo preprocessing totale", time.time() - start_pre)

del best_kmeans, variant_traces_embeddings, traces_df, variant_traces_df
gc.collect()

df_group = filtered_df.groupby("case:concept:name", sort=False)

# creazione dei DataFrame dei medoidi
medoid_dfs = []
for i in range(len(medoids)):
    medoid_dfs.append(df_manipulation.get_df_redundant(df_group, medoids.iloc[[i]]))

# inizializzazione delle variabili per il ciclo iterativo
best_metrics = metrics_util.init_metrics_dict()
best_f1_pos = []
best_f1 = 0
improving = True
local_best_f1 = 0

while improving:
    iterative_metrics = metrics_util.init_metrics_dict()

    for i in range(len(medoid_dfs)):
        if i not in best_f1_pos:
            temp_df = pd.concat([medoid_dfs[pos] for pos in best_f1_pos] + [medoid_dfs[i]])
            start_time_d = time.time()
            net, im, fm = discover.discover_process_model(temp_df, algorithm=algorithm)
            end_time_d = time.time()
            
            del temp_df
            gc.collect()

            iterative_metrics = metrics_util.compute_metrics(iterative_metrics, eventlog_df, net, im, fm, end_time_d - start_time_d, dataset, dataframe_type=f"subset:{'_'.join(map(str, best_f1_pos))}_{i}", miner=f"{algorithm}_00")
            discover.save_process_model(net, im, fm, img_name=f"{dataset}_{algorithm}_subset_{'_'.join(map(str, best_f1_pos))}_{i}_layer{len(best_f1_pos)}_F1_{iterative_metrics['f1'][-1]:.2f}", dir=img_path)
            print(f"F1: {iterative_metrics['f1'][-1]}")
            if iterative_metrics["f1"][-1] > local_best_f1:
                local_best_f1 = iterative_metrics["f1"][-1]
                local_best_net = net
                local_best_im = im
                local_best_fm = fm
                best_rel_pos = i
            del net, im, fm
            gc.collect()

    metrics_util.save_metrics(iterative_metrics, csv_path, filename=f"_{dataset}_{algorithm}_iterative_metrics_layer_{len(best_f1_pos)}.xlsx")
    print(local_best_f1)
    print(best_f1)
    improving = local_best_f1 > best_f1
    if improving:
        best_f1_pos.append(best_rel_pos)
        best_f1 = local_best_f1
        best_net = local_best_net
        best_im = local_best_im
        best_fm = local_best_fm
        best_metrics = metrics_util.append_metrics(best_metrics, iterative_metrics, np.argmax(iterative_metrics["f1"]))
    else:
        improving=False
        print(f'{dataset} --> {best_f1:.2f}')
        print('exit')
        break
    metrics_util.save_metrics(best_metrics, csv_path, filename=f"_{dataset}_{algorithm}_best_metrics_layer_{len(best_f1_pos) - 1}.xlsx")

    del iterative_metrics, local_best_net, local_best_im, local_best_fm
    gc.collect()

discover.visualize_process_model(best_net, best_im, best_fm, img_name=f"{dataset}_{algorithm}_best_subset_00", dir=img_path)

# salvataggio del sottoinsieme di medoid con il miglior f1
medoid_subset = medoids.iloc[best_f1_pos]
medoid_subset.to_csv(f'{csv_path}/_{dataset}_variant_traces_cluster_best_f1.csv', index=False)

metrics = metrics_util.init_metrics_dict()

# aggiungo le metriche del miglior sottoinsieme
metrics = metrics_util.append_metrics(metrics, best_metrics)
