# Import library
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import yellowbrick
from streamlit_yellowbrick import st_yellowbrick
from wordcloud import WordCloud
from streamlit_option_menu import option_menu
from sklearn.utils import resample
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from yellowbrick.classifier import ROCAUC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import pickle
from sklearn.model_selection import train_test_split
from PIL import Image

################
# Imprt Pakage
from underthesea import word_tokenize, pos_tag, sent_tokenize
import regex
import demoji
from pyvi import ViPosTagger, ViTokenizer
import string

##LOAD EMOJICON
file = open('files/emojicon.txt', 'r', encoding="utf8")
emoji_lst = file.read().split('\n')
emoji_dict = {}
for line in emoji_lst:
    key, value = line.split('\t')
    emoji_dict[key] = str(value)
file.close()
#################
#LOAD TEENCODE
file = open('files/teencode.txt', 'r', encoding="utf8")
teen_lst = file.read().split('\n')
teen_dict = {}
for line in teen_lst:
    key, value = line.split('\t')
    teen_dict[key] = str(value)
file.close()
###############
#LOAD TRANSLATE ENGLISH -> VNMESE
file = open('files/english-vnmese.txt', 'r', encoding="utf8")
english_lst = file.read().split('\n')
english_dict = {}
for line in english_lst:
    key, value = line.split('\t')
    english_dict[key] = str(value)
file.close()
################
#LOAD wrong words
file = open('files/wrong-word.txt', 'r', encoding="utf8")
wrong_lst = file.read().split('\n')
file.close()
#################
#LOAD STOPWORDS
file = open('files/vietnamese-stopwords.txt', 'r', encoding="utf8")
stopwords_lst = file.read().split('\n')
file.close()
def process_text(text, emoji_dict, teen_dict, wrong_lst):
    document = text.lower()
    document = document.replace("’",'')
    document = regex.sub(r'\.+', ".", document)
    new_sentence =''
    for sentence in sent_tokenize(document):
        # if not(sentence.isascii()):
        ###### CONVERT EMOJICON
        sentence = ''.join(emoji_dict[word]+' ' if word in emoji_dict else word for word in list(sentence))
        ###### CONVERT TEENCODE
        sentence = ' '.join(teen_dict[word] if word in teen_dict else word for word in sentence.split())
        ###### DEL Punctuation & Numbers
        pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
        sentence = ' '.join(regex.findall(pattern,sentence))
        ###### DEL wrong words   
        sentence = ' '.join('' if word in wrong_lst else word for word in sentence.split())
        new_sentence = new_sentence+ sentence + '. '                    
    document = new_sentence  
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document
# Chuẩn hóa unicode tiếng việt
def loaddicchar():
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
        '|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
        '|')
    for i in range(len(char1252)):
        dic[char1252[i]] = charutf8[i]
    return dic

# Đưa toàn bộ dữ liệu qua hàm này để chuẩn hóa lại
def convert_unicode(text):
    dicchar = loaddicchar()
    return regex.sub(
        r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
        lambda x: dicchar[x.group()], text)
    
    # có thể bổ sung thêm các từ: chẳng, chả...
def process_special_word(text):
    new_text = ''
    text_lst = text.split()
    i= 0
    if 'không' in text_lst or 'chẳng' in text_lst or 'k' in text_lst:
        while i <= len(text_lst) - 1:
            word = text_lst[i]
            #print(word)
            #print(i)
            if  word == 'không':
                next_idx = i+1
                if next_idx <= len(text_lst) -1:
                    word = word +'_'+ text_lst[next_idx]
                i= next_idx + 1
            else:
                i = i+1
            new_text = new_text + word + ' '
    else:
        new_text = text
    return new_text.strip()

def process_postag_thesea(text):
    new_document = ''
    for sentence in sent_tokenize(text):
        sentence = sentence.replace('.','')
        ###### POS tag
        lst_word_type = ['N','Np','A','AB','V','VB','VY','R']
        sentence = ' '.join( word[0] if word[1].upper() in lst_word_type else '' for word in pos_tag(process_special_word(word_tokenize(sentence, format="text"))))
        new_document = new_document + sentence + ' '
    ###### DEL excess blank space
    new_document = regex.sub(r'\s+', ' ', new_document).strip()
    return new_document

def remove_stopword(text, stopwords):
    ###### REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    ###### DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

def preprocessing_comment(text, emoji_dict = emoji_dict, teen_dict = teen_dict, wrong_lst = wrong_lst, stopwords = stopwords_lst):
    comments = process_text(text, emoji_dict, teen_dict, wrong_lst)
    comments = convert_unicode(comments)
    comments = process_special_word(comments)
    comments = process_postag_thesea(comments)
    comments = remove_stopword(comments, stopwords)
    return comments

################



st.title("Sentiment Analysis - Shopee Comments")

# Read data
data = pd.read_pickle('model/data_new_selected_clean.pkl')

# Split Data
X = data[["comment_clean","length"]]
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state= 42)

# TFIDF
tfidf = pickle.load(open("model/tfidf_model.pkl", "rb"))
X_train_vec = tfidf.fit_transform(X_train['comment_clean'])
X_test_vec = tfidf.transform(X_test['comment_clean'])

# Evaluate model
model = pickle.load(open('model/LDS6_Model_LG.pkl', 'rb'))
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
score_train = model.score(X_train_vec, y_train)
score_test = model.score(X_test_vec, y_test)
precision = precision_score(y_test, y_pred, average = 'macro')
recall = recall_score(y_test, y_pred, average = 'macro')
f1 = f1_score(y_test, y_pred, average = 'macro')

# GUI
menu = ["Overview", "Evaluation", "Prediction"]
choice = option_menu(
    menu_title = None,
    options= menu,
    menu_icon= 'cast',
    orientation= 'horizontal'
)

if choice == "Overview":
    # Sentiment Analysis
    st.write("### 1. Sentiment Analysis")
    sentiment_img = Image.open("image/what-is-sentiment-analysis.jpg")
    st.image(sentiment_img, width = 700)
    st.write("##### Sentiment analysis is about predicting the sentiment of a piece of text and then using this information to understand users’ (such as customers) opinions. . The principal objective of sentiment analysis is to classify the polarity of textual data, whether it is positive, negative, or neutral. Whether the end-user sentiment is positive or negative or neutral can be used to answer many different business questions. The text whose sentiment needs to be processed can be extracted from many different sources, but in the current scenario, sentiment analysis is mostly about tweets and reviews.")
 
    # Shopee User Comments Analysis
    st.write("### 2. Shopee Analysis - User Comments")
    comment1 = Image.open('image/comment1.png')
    comment2 = Image.open('image/comment2.png')
    st.image([comment1, comment2], width = 700)

    # Information of data
    st.write("### 3. EDA")
    
    st.write("#### Some Data")
    st.dataframe(data.head(), width=700)

    # Barchart of each class
    bar_img = Image.open("image/target_bar.png")
    st.image(bar_img, width = 700)
    
    # Piechart of each class
    pie_img = Image.open("image/target_pie.png")
    st.image(pie_img, width = 1000)
    
    # WordCloud of each class
    st.write("#### WordCloud Of Each Class")
    
    # Positive wordcloud
    st.write("#### Biểu đồ WordCloud cho label Positive")
    wc_pos_img = Image.open("image/WordCloud_positive.png")
    st.image(wc_pos_img, width = 700)
    
    # Neutral wordcloud
    st.write("#### Biểu đồ WordCloud cho label Neutral")
    wc_neu_img = Image.open("image/WordCloud_neutral.png")
    st.image(wc_neu_img, width = 700)
    
    # Negative wordcloud
    st.write("#### Biểu đồ WordCloud cho label Negative")
    wc_neg_img = Image.open("image/WordCloud_negative.png")
    st.image(wc_neg_img, width = 700)
      
elif choice == 'Evaluation':
    st.write("### 1.Accuray, Precision, Recall, F1-Score")
    st.code("Accuracy: " + str(round(accuracy,2)))
    st.code("Score train: " + str(round(score_train,2)) + " vs Score test: " +str(round(score_test,2)))
    st.code("Precision: " + str(round(precision,2)))
    st.code("Recall: " + str(round(recall,2)))
    st.code("F1-Score: " + str(round(f1,2)))
    
    st.write("#### 2. ROC curve")
    visualizer = ROCAUC(model, )
    visualizer.fit(X_train_vec, y_train)
    visualizer.score(X_test_vec, y_test)
    st_yellowbrick(visualizer)
    
    st.write("### 3.Confusion Matrix")
    st.write("#### Classification Report")
    st.code(classification_report(y_test, y_pred))
    # st.write("#### Heat Map")
    # fig, ax= plt.subplots()
    # ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot= True,fmt ='d', cmap = plt.cm.Blues)
    # plt.show()
    # st.pyplot(fig)

    data_prediction = X_test.copy()
    data_prediction['prediction'] = y_pred

    st.write("### WordCloud")
    
    st.write("#### Biểu đồ WordCloud cho label Positive")
    text = " ".join(i for i in data_prediction[data_prediction['prediction'] == 1].dropna().comment_clean.values)
    wordcloud = WordCloud( max_words=150, height= 640, width = 700,  background_color="black", colormap= 'viridis').generate(text)
    fig= plt.figure(figsize = (15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    # plt.title('Biểu đồ WordCloud cho label Positive', color = "black", fontsize=20, fontweight='bold')
    plt.axis("off")
    plt.show()
    st.pyplot(fig)
    
    st.write("#### Biểu đồ WordCloud cho label Neutral")
    text = " ".join(i for i in data_prediction[data_prediction['prediction'] == 2].dropna().comment_clean.values)
    wordcloud = WordCloud( max_words=150, height= 640, width = 700,  background_color="black", colormap= 'viridis').generate(text)
    fig= plt.figure(figsize = (15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    # plt.title('Biểu đồ WordCloud cho label Neutral', color = "black", fontsize=20, fontweight='bold')
    plt.axis("off")
    plt.show()
    st.pyplot(fig)
    
    st.write("#### Biểu đồ WordCloud cho label Negative")
    text = " ".join(i for i in data_prediction[data_prediction['prediction'] == 3].dropna().comment_clean.values)
    wordcloud = WordCloud( max_words=150, height= 640, width = 700,  background_color="black", colormap= 'viridis').generate(text)
    fig= plt.figure(figsize = (15, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    # plt.title('Biểu đồ WordCloud cho label Negative', fontsize = 16)
    plt.axis("off")
    plt.show()
    st.pyplot(fig)

elif choice == "Prediction":
    st.subheader("What Do You Want?")
    lines = None
    option = st.selectbox("",options = ("Input A Comment", "Upload A File"))
    if option == "Input A Comment":
        comment = st.text_area("Type Your Comment: ")
        if comment != "":
            comment_pre = preprocessing_comment(comment)
            lines = np.array([comment_pre])
            tfidf_comment = tfidf.transform(lines)
            y_pred_new = model.predict(tfidf_comment)
            if y_pred_new == 1:
                positive = Image.open("image/positive.png")
                positive = positive.resize((600,600))
                st.image(positive, width = 200)
            elif y_pred_new == 2:
                neutral = Image.open("image/neutral.png")
                neutral = neutral.resize((600,600))
                st.image(neutral, width = 200)
            else:
                negative = Image.open("image/negative.png")
                negative = negative.resize((600,600))
                st.image(negative, width = 200)
    if option == "Upload A File":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            df = pd.read_csv(uploaded_file_1, header = None)
            df = df.iloc[1:,:]
            st.write("Your DataFrame:")
            st.dataframe(df[0])
            lines = df[0].apply(lambda x: preprocessing_comment(x))
            x_new = tfidf.transform(lines)        
            y_pred_new = model.predict(x_new)
            df['prediction'] = y_pred_new
            st.write("### Prediction:")
            st.write("#### 1: Positive, 2: Neutral, 3: Negative")
            st.dataframe(df[[0,'prediction']])

        


