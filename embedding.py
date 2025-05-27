from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from kneed import KneeLocator

sbert_model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
from sklearn.feature_extraction.text import CountVectorizer

def get_feature_ranges(embeddings):
    """
    Calcola i range delle features degli embeddings.

    :param embeddings: matrice degli embeddings
    :return: min, max e range delle features
    """
    feature_min = np.min(embeddings, axis=0)
    feature_max = np.max(embeddings, axis=0)
    feature_range = feature_max - feature_min
    return feature_min, feature_max, feature_range


def get_variants_embeddings_agg(variants):
    """
    Calcola gli embeddings delle varianti utilizzando il modello SBERT.

    :param variants: lista di varianti
    :return: matrice degli embeddings
    """
    new_variants = []
    #print(variants)
    for v in variants:
        v_new = v.split(' -> ')
        l = " ".join([l.replace(' ', '') for l in v_new])
        new_variants.append(l)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(new_variants)
    return X.toarray()#sbert_model.encode(variants)

def get_variants_embeddings(variants):
    """
    Calcola gli embeddings delle varianti utilizzando il modello SBERT.

    :param variants: lista di varianti
    :return: matrice degli embeddings
    """
    return sbert_model.encode(variants)

def run_kmeans_elbow(embeddings, k_min=2, k_max=15, random_state=42):
    """
    Esegue il metodo dell'elbow per determinare il numero ottimale di cluster e applica il kmeans ottimale.

    :param embeddings: matrice degli embeddings
    :param k_min: numero minimo di cluster
    :param k_max: numero massimo di cluster
    :param random_state: seed per la riproducibilità
    :return: modello kmeans ottimale
    """
    inertia_values = []
    k_values = list(range(k_min, k_max + 1))

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(embeddings)
        inertia_values.append(kmeans.inertia_)
    
    # Uso della libreria kneed per trovare il punto di "elbow"
    knee = KneeLocator(k_values, inertia_values, curve="convex", direction="decreasing")  
    print(f"Elbow found at k = {knee.knee}")

    best_kmeans = KMeans(n_clusters=knee.knee, random_state=random_state)
    best_kmeans.fit(embeddings)
    
    return best_kmeans

def compute_medoid(cluster_embeddings):
    """
    Calcola il medoid di un cluster dato un insieme di embeddings.

    :param cluster_embeddings: matrice degli embeddings del cluster
    :return: indice del medoid
    """
    distancematrix = pairwise_distances(cluster_embeddings, metric='cosine')
    total_distances = np.sum(distancematrix, axis=1)
  
    return np.argmin(total_distances)

def get_medoid_df(df, variants_embeddings, kmeans):
    """
    Calcola i medoid globali per ogni cluster e crea un nuovo dataframe con le frequenze totali.
    
    :param df: dataframe delle varianti
    :param variants_embeddings: matrice degli embeddings delle varianti
    :param kmeans: modello kmeans applicato
    :return: dataframe con i medoid globali e le frequenze totali
    """
    medoid_indexes = []
    frequcies = []

    for cluster in range(kmeans.n_clusters):
        indexes = np.where(kmeans.labels_ == cluster)[0] # np.where restitusce una tupla il cui elemento [0] è l'array di indici
        cluster_embeddings = variants_embeddings[indexes] # recupera embeddings del cluster alle posizoni degli indici
        medoid_local_index = compute_medoid(cluster_embeddings)  # calcola il medoid locale del cluster
        medoid_global_index = indexes[medoid_local_index]  # recupera l'indice globale del medoid locale
        medoid_indexes.append(medoid_global_index) # aggiungi l'indice globale alla lista dei medoid
        frequcies.append(np.sum(df["frequency"].iloc[indexes])) # calcola la frequenza totale del cluster e aggiungila alla lista delle frequenze
    

    df_medoid = df.copy()
    df_medoid = df_medoid.iloc[medoid_indexes] # crea un nuovo dataframe con i medoid globali
    df_medoid["frequency"] = frequcies # aggiungi la frequenza totale al dataframe dei medoid

    return df_medoid