from flask import Flask, render_template, json,request, Response, send_file
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import csv, nltk, time, re, string, pickle
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import io, random
import ast
import mysql.connector
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle


app = Flask(__name__)

addStopWord=[]
normalDict=[]

def get_files():
    global addStopWord, normalDict
    sql = "SELECT * FROM file_path"
    mycursor.execute(sql)
    myres = mycursor.fetchall()
    for i in myres:
        if i[1] == 'stopword':
            with open(i[2]) as f:
                addStopWord = f.read().splitlines()
        elif i[1] == 'normalisasi':
            with open(i[2], "r") as f:
                contents = f.read()
                normalDict = ast.literal_eval(contents)


def normalization(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"\d+", "", sentence)
    sentence = sentence.translate(str.maketrans("","", string.punctuation))
    sentence = sentence.strip()
    sentence = " ".join([normalDict.get(word, word) for word in sentence.split()])
    return sentence
    
def stopwordRemoval(sentence, addword):
    stop_factory = StopWordRemoverFactory().get_stop_words() #load default stopword
    more_stopword = addword #menambahkan stopword
    data = stop_factory + more_stopword #menggabungkan stopword
    dictionary = ArrayDictionary(data)
    str = StopWordRemover(dictionary)
    sentence = str.remove(sentence)
    return sentence

def stemming(sentence):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    sentence = stemmer.stem(sentence)
    return sentence

def extract_features(sentences):
    words = set(sentences)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in words)
    return features

def preprocessing(sentence):
    sentence = normalization(sentence)
    sentence = stemming(sentence)
    sentence = stopwordRemoval(sentence, addStopWord)
    sentence = nltk.tokenize.word_tokenize(sentence)
    return sentence

# define an empty list
featureList = []
# open file and read the content in a list
with open('BagofWord.txt', 'r') as filehandle:
    for line in filehandle:
        currentFeature = line[:-1]
        featureList.append(currentFeature)

# load the model from disk
load_NB_model = pickle.load(open('NB_model.sav', 'rb'))
load_Topic_model = pickle.load(open('Topic_model.sav', 'rb'))

user_name = ''
userrole = 5
tgl_option = ['All']

import mysql.connector
# Fetch Data
db = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="",
    database="sa_pjj"
)

mycursor = db.cursor()
get_files()

@app.route("/")
def main():
    print(user_name)
    return render_template('index.html', username=user_name, role=userrole)

def getTglAmbil():
    global tgl_option
    tgl_option = []
    df_temp = pd.read_sql("SELECT DISTINCT id_pengambilan, nama FROM pengambilan_kuesioner", con = db)
    for index, x in df_temp.iterrows():
        tmp = {
            'id':x['id_pengambilan'], 'nama':x['nama']
        }
        tgl_option.append(tmp)

def getSentiment(labels=[], label = '', sentiment = '', year ='', topic = "all"):
    if topic != "all":
        sql = "SELECT "+label+", COUNT(*) FROM review WHERE topic = '"+topic+"' AND sentiment = '"+sentiment+"' AND id_pengambilan = "+year+" GROUP BY "+label+" ORDER BY "+label
    else:
        sql = "SELECT "+label+", COUNT(*) FROM review WHERE sentiment = '"+sentiment+"' AND id_pengambilan = "+year+" GROUP BY "+label+" ORDER BY "+label
    print(sql)
    mycursor.execute(sql)
    myres = mycursor.fetchall()
    res = []
    for i in labels:
        c = 0
        for j in myres:
            if i == j[0]:
                res.append(j[1])
                c = 1
                break
        if c == 0:
            res.append(0)
    return res
    
label_report = 'angkatan'
id_ambil_report = 19
id_banding_report = 21

@app.route('/report', methods=['GET'])
def showRep():
    getTglAmbil()
    return render_template('report.html', username=user_name, role=userrole, cek=0, options=tgl_option)

@app.route('/searchReport', methods=['POST'])
def showGetReport():
    global label_report, id_ambil_report, id_banding_report
    print(request.method)

    negatif = []; positif = []; materinegatif = []; materipositif = []; dosennegatif = []; dosenpositif = []
    medianegatif = []; mediapositif = []; fasilnegatif = []; fasilpositif = []

    cek = 1
    label_report = str.lower(request.form['kategori'])
    id_ambil_report = request.form['tanggalAmbil']
    id_banding_report = request.form['tanggalBanding']
    
    label = label_report
    print(label, id_ambil_report)

    labels= []
    if label != 'perbandingan':
        if label == 'fakultas':
            label = "LEFT(fakultas,LOCATE('-',fakultas) - 1)"
        print(label)
        sql = "SELECT DISTINCT "+label+" FROM review WHERE id_pengambilan ="+id_ambil_report+" ORDER BY " + label
        print(sql)
        mycursor.execute(sql)
        myres_label = mycursor.fetchall()
        for i in myres_label:
            labels.append(i[0])

        negatif = getSentiment(labels, label, "Negatif", str(id_ambil_report))
        positif = getSentiment(labels, label, "Positif", str(id_ambil_report))
        materinegatif = getSentiment(labels, label, "Negatif", str(id_ambil_report), "Materi")
        materipositif = getSentiment(labels, label, "Positif", str(id_ambil_report), "Materi")
        dosennegatif = getSentiment(labels, label, "Negatif", str(id_ambil_report), "Dosen")
        dosenpositif = getSentiment(labels, label, "Positif", str(id_ambil_report), "Dosen")
        medianegatif = getSentiment(labels, label, "Negatif", str(id_ambil_report), "Media Pembelajaran")
        mediapositif = getSentiment(labels, label, "Positif", str(id_ambil_report), "Media Pembelajaran")
        fasilnegatif = getSentiment(labels, label, "Negatif", str(id_ambil_report), "Fasilitas Pendukung")
        fasilpositif = getSentiment(labels, label, "Positif", str(id_ambil_report), "Fasilitas Pendukung")
    
    else:
        sql = "SELECT sentiment, p.nama, COUNT(*) FROM review r RIGHT OUTER JOIN pengambilan_kuesioner p ON r.id_pengambilan = p.id_pengambilan WHERE p.id_pengambilan = "+str(id_ambil_report)+" OR p.id_pengambilan = "+str(id_banding_report) +" GROUP BY sentiment, p.id_pengambilan"
        data_df = pd.read_sql(sql, con=db)
        for index, x in data_df.iterrows():
            if index <= 1:
                labels.append(x[1])
            if x['sentiment'] == 'Positif':
                positif.append(x[2])
            else:
                negatif.append(x[2])
        sql = "SELECT sentiment, topic, p.nama, COUNT(*) FROM review r RIGHT OUTER JOIN pengambilan_kuesioner p ON r.id_pengambilan = p.id_pengambilan WHERE p.id_pengambilan = "+str(id_ambil_report)+" OR p.id_pengambilan = "+str(id_banding_report) +" GROUP BY sentiment, topic, p.id_pengambilan"
        data_df = pd.read_sql(sql, con=db)
        for index, x in data_df.iterrows():
            if x['sentiment'] == 'Positif':
                if x['topic'] == 'Materi':
                    materipositif.append(x[3])
                elif x['topic'] == 'Dosen':
                    dosenpositif.append(x[3])
                elif x['topic'] == 'Media Pembelajaran':
                    mediapositif.append(x[3])
                elif x['topic'] == 'Fasilitas Pendukung':
                    fasilpositif.append(x[3])
            elif x['sentiment'] == 'Negatif':
                if x['topic'] == 'Materi':
                    materinegatif.append(x[3])
                elif x['topic'] == 'Dosen':
                    dosennegatif.append(x[3])
                elif x['topic'] == 'Media Pembelajaran':
                    medianegatif.append(x[3])
                elif x['topic'] == 'Fasilitas Pendukung':
                    fasilnegatif.append(x[3])

    legend="Sentiment Data"

    print(labels)

    return render_template('report.html', username=user_name, role=userrole, cek=cek, options=tgl_option, valuesneg=negatif,  
    valuespos=positif, labels=labels, legend=legend, materineg=materinegatif, materipos=materipositif, 
    dosenneg=dosennegatif, dosenpos=dosenpositif, mediapos=mediapositif, medianeg=medianegatif, fasilpos=fasilpositif, fasilneg=fasilnegatif)

@app.route('/entry', methods=['GET'])
def showEntry():
    return render_template('new_entry.html', username=user_name, role=userrole)

@app.route('/addEntry', methods = ['POST'])
def uploadEntry():
    print("ENTRY")
    sentences = []
    positif = []
    negatif = []
    labels = []
    legend = 'Sentiment Analysis'
    
    sql = "SELECT COUNT(*) FROM pengambilan_kuesioner WHERE nama = '"+request.form['namaAjaran']+"' OR tgl_awal = '"+request.form['inputAwal']+"' AND tgl_akhir = '"+request.form['inputAwal']+"'"
    print(sql)
    mycursor.execute(sql)
    myres = mycursor.fetchone()
    cek = 0
    if myres[0] == 0:
        sql = "INSERT INTO pengambilan_kuesioner VALUES (DEFAULT, %s, %s, %s, %s, %s)"
        val = (request.form['namaAjaran'], request.form['tahunAjaran'], request.form['semester'], request.form['inputAwal'], request.form['inputAkhir'])
        print(request.form['namaAjaran'], request.form['tahunAjaran'], request.form['semester'], request.form['inputAwal'], request.form['inputAkhir'])
        mycursor.execute(sql, val)
        db.commit()
        
        sql = "SELECT id_pengambilan FROM pengambilan_kuesioner WHERE nama = '"+request.form['namaAjaran']+"'"
        mycursor.execute(sql)
        id_ambil = mycursor.fetchone()
        print(id_ambil[0])

        f = request.files['file']
        f.save(secure_filename(f.filename))
        test_data = pd.read_excel(f)
        for index, row in test_data.iterrows():
            if str(row['Komentar']).lower() == '' or str(row['Komentar']).lower() == 'tidak ada' or str(row['Komentar']).lower() == 'belum ada':
                continue
            processedSentence = preprocessing(str(row['Komentar']))
            res_sentiment = load_NB_model.classify(extract_features(processedSentence))
            res_topic = load_Topic_model.classify(extract_features(processedSentence))
            sentence = {
                "text": row[3], "sentiment": res_sentiment, "topic": res_topic
            }
            
            sql = "INSERT INTO review VALUES (DEFAULT, %s, %s, %s, %s, %s, %s, %s)"
            val = (row['Komentar'], res_sentiment, res_topic, row['Fakultas'], row['Angkatan'], row['Gender'], id_ambil[0])
            mycursor.execute(sql, val)
            db.commit()
            sentences.append(sentence)

        labels = ['All', 'Materi', 'Dosen', 'Media Pembelajaran', 'Fasilitas Pendukung']
            
        for l in labels:
            co_pos = 0
            co_neg = 0
            for i in sentences:
                if l == 'All':
                    if i['sentiment'] =='Positif':
                            co_pos += 1
                    else:
                            co_neg+=1

                elif i['topic'] == l:
                    if i['sentiment'] =='Positif':
                        co_pos += 1
                    else:
                        co_neg+=1
            positif.append(co_pos)
            negatif.append(co_neg)
        
        print(positif)
        print(negatif)
    else:
        cek = 1
        print("DATA SUDAH ADA")

    getTglAmbil()
    return render_template('new_entry.html', cek=cek, username=user_name, role=userrole, sentences=sentences, valuesneg=negatif, valuespos=positif, labels=labels, legend=legend)

@app.route("/admin")
def showAdmin():
    print(user_name)
    return render_template('admin.html', username=user_name, role=userrole)

@app.route('/wordList', methods = ['GET','POST'])
def showWordList():
    if request.method=='POST' :
        f = request.files['file']
        print(f.filename)
    sentence=''
    for i in normalDict:
        sentence=sentence+"<li>"+i+" : "+normalDict[i]+" </li>"
    return sentence

@app.route('/addStopWord', methods=['POST'])
def newStopWord():
    print("POST")
    f = request.files['file']
    print(f.filename)
    f.save(secure_filename(f.filename))
    sql = "UPDATE file_path SET nama = '"+f.filename+"' WHERE fungsi='stopword'"
    mycursor.execute(sql)
    db.commit()
    get_files()
    return render_template('admin.html', username=user_name, role=userrole, success="You've successfully add New Stopword")

@app.route('/addNormalWord', methods=['POST'])
def newNormalWord():
    print("POST")
    f = request.files['file']
    print(f.filename)
    f.save(secure_filename(f.filename))
    sql = "UPDATE file_path SET nama = '"+f.filename+"' WHERE fungsi='normalisasi'"
    mycursor.execute(sql)
    db.commit()
    get_files()
    return render_template('admin.html', username=user_name, role=userrole, success="You've successfully add New Normalize Dictionary")

@app.route('/download/<filename>')
def download_file(filename):
    print(filename)
    if filename == 'normalisasi':
	    path = "normalisasi_sample.txt"
    elif filename =='train':
        path = "train_sample.xlsx"
    elif filename =='stopword':
        path = "stopword_sample.txt"
    elif filename =='entry':
        path = "entry_sample.xlsx"

    return send_file(path, as_attachment=True)

@app.route('/stopWord')
def showStopWord():
    stop_factory = StopWordRemoverFactory().get_stop_words()
    print(len(stop_factory))
    sentence='<b> Sastrawi StopWord </b> <br/> <table>'
    cek=0
    for i in stop_factory:
        if cek == 0:
            sentence=sentence+"<tr>"
            sentence=sentence+"<td><li>"+i+"</li></td>"
        else:
            sentence=sentence+"<td><li>"+i+"</li></td>"
        cek+=1
        if cek >= 5 or stop_factory.index(i) == len(stop_factory)-1:
            cek = 0
            sentence=sentence+"</tr>"
            
    sentence=sentence+'</table><br/><br/> <b> Added StopWord </b> <br/>'
    for i in addStopWord:
        if i != '':
            sentence=sentence+"<li>"+i+"</li>"
        
    return sentence

NBClassifier = ''
TopicClassifier = ''
TrainTemp = []

@app.route("/training", methods=['POST'])
def trainData():
    global NBClassifier, TopicClassifier, featureList
    featureList = []
    f = request.files['file']
    pjj_data = pd.read_excel(f)
    data_df = pd.read_sql('SELECT sentence, sentiment, topic FROM training_set', con=db)
    data_df = pd.concat([data_df, pjj_data], ignore_index=True)

    train, test = train_test_split(data_df, test_size=0.2, stratify=data_df[['sentiment', 'topic']])

    topic_sentences = []
    sentiment_sentences = []

    for index, x in train.iterrows():
        topic = x['topic']; sentiment = x['sentiment']; sentence = x['sentence']
        tmp =[]; tmp.append(sentence); tmp.append(sentiment); tmp.append(topic)
        processedSentence = preprocessing(str(sentence))
        featureList.extend(processedSentence)
        topic_sentences.append((processedSentence, topic))
        sentiment_sentences.append((processedSentence, sentiment))
        TrainTemp.append(tmp)

    featureList = list(set(featureList))
    sentiment_set = nltk.classify.util.apply_features(extract_features, sentiment_sentences)
    topic_set = nltk.classify.util.apply_features(extract_features, topic_sentences)
    NBClassifier = nltk.NaiveBayesClassifier.train(sentiment_set)
    TopicNBClassifier = nltk.NaiveBayesClassifier.train(topic_set)

    sentiment_pred = []
    sentiment_true = []
    topic_pred = []
    topic_true = []

    #Preprocessing Test Data
    for index, row in test.iterrows():
        sentiment_true.append(row['sentiment']); topic_true.append(row['topic']); sentence = row['sentence']
        processedSentence = preprocessing(str(sentence))
        res_sentiment = NBClassifier.classify(extract_features(processedSentence))
        res_topic = TopicNBClassifier.classify(extract_features(processedSentence))
        sentiment_pred.append(res_sentiment)
        topic_pred.append(res_topic)
    
    Saccuracy = accuracy_score(sentiment_true, sentiment_pred)
    Taccuracy = accuracy_score(topic_true, topic_pred)
    
    return render_template('admin.html', username=user_name, role=userrole, Saccuracy=Saccuracy, Taccuracy=Taccuracy)

@app.route("/saveTRAIN", methods=['POST'])
def saveData():
    # save the model to disk
    filename = 'NB_model.sav'
    pickle.dump(NBClassifier, open(filename, 'wb'))
    filename = 'Topic_model.sav'
    pickle.dump(TopicClassifier, open(filename, 'wb'))
    f=open('BagofWord.txt','w')
    for x in featureList:
        f.write(x+'\n')
    f.close()

    for row in TrainTemp:
        sql = "INSERT INTO training_set VALUES (DEFAULT, %s, %s, %s)"
        val = (row[0], row[1], row[2])
        mycursor.execute(sql, val)
        db.commit()

    return "You've successfully saved your NEW Model"

@app.route("/login", methods=['POST'])
def cekLogin():
    global user_name
    global userrole
    sql = "SELECT COUNT(*), name, role FROM user WHERE username = '"+request.form['username']+"' AND password = '"+request.form['password']+"'"
    mycursor.execute(sql)
    myres = mycursor.fetchone()

    if myres[0] == 1:
        user_name = myres[1]
        userrole = int(myres[2])
        print(myres)
        print(user_name)
        return ("Welcome, "+myres[1])
    else:
        print("no")
        return ("User or Password is WRONG")

@app.route("/logout", methods=['POST'])
def cekLogout():
    global user_name
    global userrole
    user_name = ""
    userrole = 5
    return "success"


if __name__ == "__main__":
    app.run()