from django.shortcuts import render, redirect
from .forms import TextForm, LoginForm, SignUpForm
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from .models import ModelFile
# モデル構造をインポート
import numpy as np
import pandas as pd
import torch
import pickle
import sqlite3
import sys, os
sys.path.append("/Users/yukiiimori/Desktop/MyApp/MyApp_project/model/sixty_thou")
from model.net_structure import Net_section_classifier, Net_subclass_classifier, config

# テキストの前処理
import MeCab
import mojimoji
import re

#----- modelのload -----
base_dir_vec = f'model/vectorizer/'
base_dir_cate = f'model/categories/'
base_dir_net = f'model/net/'
base_dir_IPC = f'model/IPC_A-Hsection/IPC_Ver2021-A-Hsection.db'

# (1) text -> sections
# vectorizer
path = base_dir_vec + 'vectorizer_for_all_sections.pkl'
with open(path, mode='rb') as f:
  vectorizer_section = pickle.load(f)

# categories
path = base_dir_cate + 'categories_for_all_sections.pkl'
with open(path, mode='rb') as f:
  categories_section = pickle.load(f)

# NN
n_feats = len(vectorizer_section.vocabulary_)
n_classes = len(categories_section)
path = base_dir_net + 'Net_classify_all_sections_20210920.pth'
model_section = Net_section_classifier(n_feats, n_classes)
model_section.load_state_dict(torch.load(path))

# (2) text -> subclasses
# vectorizer
vectorizer_paths = [
  'vectorizer_for_A_section.pth',
  'vectorizer_for_B_section.pth',
  'vectorizer_for_C_section.pth',
  'vectorizer_for_D_section.pth',
  'vectorizer_for_E_section.pth',
  'vectorizer_for_F_section.pth',
  'vectorizer_for_G_section.pth',
  'vectorizer_for_H_section.pth',
]
vectorizer_subclass = []
for path in vectorizer_paths:
  path = base_dir_vec + path
  with open(path, mode='rb') as f:
    vectorizer_subclass.append(pickle.load(f))

# categories
categories_paths = [
  'categories_dict_for_A_section.pth',
  'categories_dict_for_B_section.pth',
  'categories_dict_for_C_section.pth',
  'categories_dict_for_D_section.pth',
  'categories_dict_for_E_section.pth',
  'categories_dict_for_F_section.pth',
  'categories_dict_for_G_section.pth',
  'categories_dict_for_H_section.pth',
]
categories_dict_subclass = []
for path in categories_paths:
  path = base_dir_cate + path
  with open(path, mode='rb') as f:
    categories_dict_subclass.append(pickle.load(f))

# NN
model_paths = [
  'Net_classify_A_section_subclass.pth',
  'Net_classify_B_section_subclass.pth',
  'Net_classify_C_section_subclass.pth',
  'Net_classify_D_section_subclass.pth',
  'Net_classify_E_section_subclass.pth',
  'Net_classify_F_section_subclass.pth',
  'Net_classify_G_section_subclass.pth',
  'Net_classify_H_section_subclass.pth',
]
model_subclass = []
for i, path in enumerate(model_paths):
  path = base_dir_net + path
  vectorizer = vectorizer_subclass[i]
  categories = categories_dict_subclass[i]
  n_feats = len(vectorizer.vocabulary_)
  n_classes = len(categories)
  model = Net_subclass_classifier(n_feats, n_classes, config)
  # deviceをcpuに指定
  model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
  # model.load_state_dict(torch.load(path))
  model_subclass.append(model)

# ストップワード
file_path = 'model/stopword/Japanese.txt'
with open(file_path, 'r') as f:
    stopword = f.read().splitlines()

# デバイス
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#-----------------------

# ログインページ
class Login(LoginView):
    form_class = LoginForm
    template_name = 'MyApp/login.html'

# ログアウトページ
class Logout(LogoutView):
    template_name = 'MyApp/base.html'

# サインアップページ
def signup(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            new_user = authenticate(username=username, password=password)
            if new_user is not None:
                login(request, new_user)
                return redirect('textinput')

        # フォーム入力内容が適切でなかった場合の処理（自分で追加）
        else:
            form = SignUpForm()
            return render(request, 'MyApp/signup.html', {'form': form})

    # POST で送信がなかった場合の処理
    else:
        form = SignUpForm()
        return render(request, 'MyApp/signup.html', {'form': form})

# テキストの前処理を行う関数
chasen = MeCab.Tagger('-Ochasen')

def preprocessing(text):
  text = mojimoji.zen_to_han(text) # 全角→半角
  text = re.sub('\d+', '0', text) # 数字→0
  text = re.sub('前記|該|当該|[（）\(\)]', '', text)
  text = chasen.parse(text)
  words = text.split('\n')

  words_list = []

  for word in words[: -2]:
    # 基本形を取得
    word = word.split('\t')[2]
    # ストップワードにないなら
    if word not in stopword:
      words_list.append(word)
    
  words = ' '.join(words_list)
  return words

# 推論を行う関数（text->section, text->subclassの各段階で使用する関数）
def predict(text, vectorizer, model, device, below_threshold=False):
  """
  input: below_threshold: probaの要素が全て閾値以下となり、
  predがゼロ配列になる場合に、probaが最大の要素を1、それ以外を0とする配列を出力するか
  output: pred: 特許分類（文字列）を格納したリスト
  output: proba: 各特許運類の信頼度（整数値）を格納したリスト
  """
  # 前処理
  text = preprocessing(text)
  # vectorizer
  text = vectorizer.transform([text]).toarray()
  # tensorに変換
  inputs = torch.tensor(text, dtype=torch.float32)
  # modelで予測
  model = model.to(device)
  # モデルをCPUに送る。AWSの容量の関係で、torchはcpu版とし、torch cpu版で推論させるため。
  model.eval().cpu()
  # model.eval()
  model.freeze()
  # _, proba = model(inputs.unsqueeze(dim=0).to(device))
  _, proba = model(inputs.to(device))
  proba = proba.detach().cpu().squeeze()
  # 閾値以上の値を1、それ以外を0にする
  THRESHOLD = 0.5
  proba = proba.numpy()
  upper, lower = 1, 0
  pred = np.where(proba > THRESHOLD, upper, lower)

  # もし全ての要素が0なら信頼度が最も高い要素を1、それ以外を0にする
  if below_threshold == True:
    if np.all(pred == 0):
      max = np.max(proba)
      pred = np.where(proba == max, upper, lower)

  return pred, proba

# 推論を行う関数（textを入力すると、特許分類と信頼度を出力）
def inference(text):
  """
  input: text: 発明内容を表すテキスト
  output: pred: 特許分類（文字列）を格納したリスト
  output: proba: 各特許運類の信頼度（整数値）を格納したリスト
  """
  # (1) text -> section
  prediction, _ = predict(text, vectorizer_section, model_section, device, below_threshold=True)

  # (2) text -> subclass
  # predictionから1のフラグが立っているindex（section）を取得
  section_list = list(np.where(prediction==1)[0])

  # IPCと信頼度を格納するリスト
  pred = []
  proba = []

  # 各index（section）でloop
  for i in section_list:
    # 予測
    model = model_subclass[i]
    vectorizer = vectorizer_subclass[i]
    prediction, probability = predict(text, vectorizer, model, device, below_threshold=True)

    # predで1のフラグが立っているindex（subclass）を取得
    subclass_list = list(np.where(prediction==1)[0])

    # 各index（subclass）でloop
    for j in subclass_list:
      # 予測値をラベルに変換
      categories_dict = categories_dict_subclass[i]
      pred.append(categories_dict[j])
      proba.append(probability[j])
  
  # 信頼度を100倍し、整数に丸める
  proba = [round(i * 100) for i in proba]

  return pred, proba
  
# テキストの入力ページ
@login_required
def text_input(request):
  if request.method == 'POST': # Formの入力があった時、
    form = TextForm(request.POST) # 入力データを取得する。
    if form.is_valid():
      form.save()
      text = request.POST['text']
      # 推論
      pred, proba = inference(text)
      # 辞書に格納し、信頼度が大きい順にソート
      pred_dict = {k: v for k, v in zip(pred, proba)}
      pred_list = sorted(pred_dict.items(), key=lambda x:x[1], reverse=True)
      # pred_listの要素にIPCのタイトルを追加。(IPC, 信頼度, タイトル)。
      conn = sqlite3.connect(base_dir_IPC)
      for i, item in enumerate(pred_list):
        # IPCに対応するタイトルを取得
        sql = f"select * from IPC where  記号 = '{item[0]}'"
        title = pd.read_sql_query(sql, conn).iloc[0, -1]
        pred_list[i] = item + (title, )
      conn.close()
      # 保存
      data = ModelFile.objects.order_by('id').reverse()[0]
      data.text = text
      data.proba = proba
      data.pred = pred
      data.save()
    return render(request, 'MyApp/result.html', {'text':text, 'pred_list':pred_list})

  else:
    form = TextForm()
    return render(request, 'MyApp/input.html', {'form':form})
