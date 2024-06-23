import streamlit as st
import numpy as np
import pandas as pd
st.title('News Summary App')
uploaded_file = st.file_uploader("Pilih file untuk diunggah (csv)")
if uploaded_file is not None:
    # Membaca file sebagai dataframe
    data_berita = pd.read_csv(uploaded_file)


# import modul
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
import string
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

# create stemmer dan stopword removal
factory = StemmerFactory() #inisialisasi objek dari kelas
stemmer = factory.create_stemmer()
stop_factory = StopWordRemoverFactory()

#mendefinisikan variabel tanda baca dan stopword
arrTdBaca = string.punctuation
arrSwindo = stop_factory.get_stop_words()

#mendefinisikan fungsi bukannum untuk mengecek apakah string adalah angka
def bukannum(string):
    pattern = r"[^\d]+" # regex untuk mencocokkan karakter bukan angka
    return re.match(pattern, string) is not None

#mendefinisikan fungsi split_text
def split_text(text):
    # Mengganti karakter \n dengan string kosong dalam teks
    text = text.replace("\n", "")
    # Menggunakan re.split() dengan regular expression untuk memisahkan kalimat
    pre1 = re.split(r"[.!?]\s", text)
    # Membersihkan setiap kalimat dari whitespace yang tidak diperlukan
    pre1 = [s.strip() for s in pre1 if s != ""]
    # Mengembalikan list kalimat
    return pre1

#fungsi preproses digunakan untuk meminimalkan jumlah kata\kalimat
def preproses(pre1):
  hasil=[]
  for i in range(0,len(pre1)):
    kalimat = pre1[i]
    pre2 = []
    # proses stemming
    kalimat = stemmer.stem(kalimat)
    # proses tokenisasi
    tokens = word_tokenize(kalimat)
    for kata in tokens:
    #proses filtering , stopword dan number removal
      if (kata not in arrTdBaca) and (kata not in arrSwindo) and bukannum(kata):
        pre2.append(kata)
    hasil.append(' '.join(pre2))

  return hasil

# Import library yang diperlukan
import networkx as nx # untuk membuat model graf dari input array kalimat
import sklearn
from sklearn.feature_extraction.text import CountVectorizer # class dari scikit-learn untuk mengubah teks menjadi vektor numerik
from sklearn.metrics.pairwise import cosine_similarity # class dari scikit-learn untuk menghitung cosine similarity antara dua vektor
import matplotlib.pyplot as plt # untuk membuat plot / gambar graf

def create_graph(sentences, preprocessed_sentences): #(bentuk input dari 2 parameter adalah list dari string)
    # Membuat objek graph dari class nx.Graph
    graph = nx.Graph()

    # Menambah setiap kalimat sebagai node baru
    for i, sentence in enumerate(sentences):
        graph.add_node(i, kalimat=sentence)

    # Mengubah setiap kalimat menjadi vektor numerik menggunakan metode BOW dengan bantuan library
    vectorizer = CountVectorizer()
    sentence_vectors = vectorizer.fit_transform(preprocessed_sentences)

    # Menambahkan edge di antara setiap pasang kalimat
    for i, node1 in enumerate(graph.nodes()):
        for j, node2 in enumerate(graph.nodes()):
            if i == j:
                continue

            # Menghitung cosine similarity antara dua kalimat
            similarity = cosine_similarity(sentence_vectors[i], sentence_vectors[j])[0][0]
            # Menambahkan edge dengan bobot yang setara dengan cosine similarity, jika cosine similarity dari 0
            if similarity > 0:
                graph.add_edge(i, j, weight=similarity)

    return graph

import random
import math

toleransi = 1/10000  # batas toleransi untuk error
debug = {'textrank': False, 'textrank2': False}  # dictionary yang berisi boolean value untuk menyalakan/mematikan debug

# Fungsi perhitungan textrank dan d sebagai faktor damping yang diset 0.85
def textrank(graph, d=0.85):
    nsimpul = []  # list yang menyimpan semua simpul dan nilai TextRank dari setiap simpul
    s = [random.randint(1, 10) for x in range(len(graph.nodes))]  # list yang menyimpan nilai TextRank awal dari setiap simpul diset random
    iterasi = 0  # menyimpan jumlah iterasi yang dilakukan saat perhitungan TextRank.
    ilanjut = True  # boolean yang menandakan apakah iterasi dilanjutkan atau tidak
    print(s)
    while ilanjut:
        if debug['textrank2']:
            print('iterasi', iterasi)
        nsimpul = []
        for i in graph.nodes():
            wij = 0  # bobot antara simpul j dan simpul i
            wjk = 0  # total bobot dari semua simpul yang terhubung dengan simpul j.
            sigma = 0
            for j in graph.neighbors(i):
                if debug['textrank']:
                    print("simpul", i , "dan simpul", j)
                wij = graph[j][i]['weight']
                if debug['textrank']:
                    print("wij", wij)
                wjk = sum(graph[i][j]['weight'] for i in graph.neighbors(j))
                if debug['textrank']:
                    print("wjk", wjk)
                    print("s[", j, "] = ", s[j]) # s[j] adalah nilai TextRank dari simpul j
                sigma += (wij * s[j]) / wjk
                if debug['textrank']:
                    print("sigma", sigma)
            # sigma
            if debug['textrank']:
                print("wij", wij, "wjk", wjk)
                print("sigma", sigma)
            if wjk > 0:
                txtrank = (1 - d) + d * sigma
                if debug['textrank']:
                    print("s[i] = s[", i, "] = ", s[i])
                    print('txt', txtrank)

            # hitung error
            error = math.fabs(txtrank - s[i])
            if error > toleransi:
                s[i] = txtrank
            elif i == (len(graph.nodes) - 1):
                ilanjut = False
                graph.nodes[i]['nilai'] = txtrank
            nsimpul.append([i, graph.nodes[i]])
            graph.nodes[i]['nilai'] = txtrank

        iterasi += 1
        if iterasi == 100:
            break
    return nsimpul

# baca input teks =======================================================================================================
def open_files(file_paths):
    data = []

data = data_berita
#print(data)
list_pretext = []
list_preproses = []
# Cetak konten file secara terpisah dengan baris baru setelah setiap konten
for i, content in enumerate(data):
    print("Konten file ke-", i+1, ":\n")
    st.title("Teks Berita Asli:")
    st.write(content)
    pre_text = split_text(content)
    list_pretext.append(pre_text)
    hasil_preproses = preproses(pre_text)
    list_preproses.append(hasil_preproses)
    print("hasil preproses ", ":\n")
    print(hasil_preproses)
    print("\n" + "=" * 50)  # Garis pemisah antara setiap konten file

# memasukkan hasil preproses ke ekstraksi fitur
graphs = []
for i in range(len(list_pretext)):
    graph = create_graph(list_pretext[i], list_preproses[i])
    graphs.append(graph)
    print(f"Graph {i} edges: {graph.edges}")

#+=========================
results = []
for i, graph in enumerate(graphs):
    result = textrank(graph)
    results.append(result)

#+==========================
hasil_listSimpul = []

for i, result in enumerate(results):
    listSimpul = []
    for node in result:
        listSimpul.append(node)
    hasil_listSimpul.append(listSimpul)

    cetak_listSimpul = '\n'.join(map(str, listSimpul))
    print(f"Informasi simpul graf {i}:")
    print(cetak_listSimpul)
    print()

print(hasil_listSimpul)
print(len(hasil_listSimpul))
type(hasil_listSimpul)

#mengurutkan hasil perankingan
def descending_sort(node):
    for t in range(0, len(node)):
        temp = t
        for i in range(1 + t, len(node)):
            if node[temp][1]['nilai'] < node[i][1]['nilai']:
                temp = i
        node[t], node[temp] = node[temp], node[t]
    return node

for i, graf in enumerate(hasil_listSimpul):
    print(f"Graf {i}:")
    print(descending_sort(graf))
    print()

#======================
def get_top_ranked_graphs(graf_list):
    top_ranked_graphs = []
    for graf in graf_list:
        top_ranked_nodes = descending_sort(graf)[:len(graf)//2]
        top_ranked_graphs.append(top_ranked_nodes)
    return top_ranked_graphs

#====
hasil_perankingan = get_top_ranked_graphs(hasil_listSimpul)

#=============
def get_sentences(graf):
    sentences = []
    for node in graf:
        kalimat = node[1]['kalimat']
        sentences.append(kalimat)
    return ' '.join(sentences)


graf_0 = hasil_perankingan[0]
hasil = get_sentences(graf_0)
st.title("Hasil Summarization:")
st.write(hasil)