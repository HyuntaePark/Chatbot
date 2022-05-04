from flask import Flask, render_template, request, jsonify, make_response
import chatbot
import tensorflow as tf
import tensorflow_datasets as tfds
import re
import pandas as pd


app = Flask(__name__)

tf.keras.backend.clear_session()

# Hyper-parameters
# VOCAB_SIZE = 8180
NUM_LAYERS = 2
D_MODEL = 256
NUM_HEADS = 8
DFF = 512
DROPOUT = 0.1

train_data = pd.read_csv('chatbotdata.csv')
train_data.dropna(inplace=True)

questions = []
for sentence in train_data['질문']:
    # 구두점에 대해서 띄어쓰기
    # ex) 12시 땡! -> 12시 땡 !
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    questions.append(sentence)

answers = []
for sentence in train_data['답변']:
    # 구두점에 대해서 띄어쓰기
    # ex) 12시 땡! -> 12시 땡 !
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = sentence.strip()
    answers.append(sentence)

# 서브워드텍스트인코더를 사용하여 질문과 답변을 모두 포함한 단어 집합(Vocabulary) 생성
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=2**13)

# 시작 토큰과 종료 토큰에 대한 정수 부여.
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# 시작 토큰과 종료 토큰을 고려하여 단어 집합의 크기를 + 2
VOCAB_SIZE = tokenizer.vocab_size + 2

model = chatbot.transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)

MAX_LENGTH = 40

learning_rate =chatbot.CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(
    learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def accuracy(y_true, y_pred):
  # ensure labels have shape (batch_size, MAX_LENGTH - 1)
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

model.compile(optimizer=optimizer, loss=chatbot.loss_function, metrics=[accuracy])

model.load_weights("model_weight/chatbot")

def evaluate(sentence):
  sentence = preprocess_sentence(sentence)

  sentence = tf.expand_dims(
      START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

  output = tf.expand_dims(START_TOKEN, 0)

  # 디코더의 예측 시작
  for i in range(MAX_LENGTH):
    predictions = model(inputs=[sentence, output], training=False)

    # 현재(마지막) 시점의 예측 단어를 받아온다.
    predictions = predictions[:, -1:, :]
    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

    # 만약 마지막 시점의 예측 단어가 종료 토큰이라면 예측을 중단
    if tf.equal(predicted_id, END_TOKEN[0]):
      break

    # 마지막 시점의 예측 단어를 출력에 연결한다.
    # 이는 for문을 통해서 디코더의 입력으로 사용될 예정이다.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0)

def predict(sentence):
  prediction = evaluate(sentence)

  predicted_sentence = tokenizer.decode(
      [i for i in prediction if i < tokenizer.vocab_size])

  print('Input: {}'.format(sentence))
  print('Output: {}'.format(predicted_sentence))

  return predicted_sentence


def preprocess_sentence(sentence):
  sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
  sentence = sentence.strip()
  return sentence


@app.route("/")
def index():
    #print(chatbot.sample())
    output = predict("고민이 있어")
    print(output)
    return render_template("index.html")


@app.route("/ai_bot", methods = ["POST"])
def ai_bot():

    #데이터 불러오기
    df = pd.read_csv('최종기사데이터.csv')
    # 예제 URL
    url = 'https://news.v.daum.net/v/EOleJemQOS'
    # 언론사 목록 리스트로 저장
    df_new = df.drop_duplicates(subset="source")
    source = sorted(df_new['source'].astype(str))
    l_source = []
    for word in source:
        new_word = re.sub('[^A-Za-z0-9가-힣”’]', '', word)
        if new_word != '':
            l_source.append(new_word)
    l_source = list(set(l_source))

    # 딕셔너리 index
    idx = 1
    # 딕셔너리
    text = {}
    # 해당 url인 기사 정보 데이터프레임으로 저장장
    df= df[df['url'] == url]
    # 인덱스 초기화
    df.reset_index(drop=True, inplace=True)

    #전처리 시작
    for article in df['article']:
        highlight_words = []
        word_list = []
        # 1. 기사본문 보여주기
        text[str(idx)] = str(article)
        idx += 1
        # 2. 영어본문제거
        if len(re.findall('[가-힣]', str(article))) == 0:
            article = None
        elif (len(re.findall('[가-힣]', str(article))) / len(str(article)) * 100) <= 10:
            article = None
        else:
            article = str(article)

        text[str(idx)] = article
        idx += 1
        # 3. 이메일 제거
        word_list = list(set(re.findall('[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', article)))
        cnt = 0
        for word in word_list:
            highlight_words = word.replace(word, '<font class="delete-word">' + word + '</font>')
            if cnt == 0:
                highlighted = re.sub(word, highlight_words, article)
            else:
                highlighted = re.sub(word, highlight_words, highlighted)
            cnt += 1
        for word in word_list:
            article = re.sub(word, '', article)

        text[str(idx)] = highlighted
        idx += 1
        word_list = []
        highlight_words = []
        cnt = 0

        # 4. 기자 이름 제거V1
        word_list = list(set(re.findall('[가-힣 ]+ ?기자', article) + re.findall('[가-힣 ]+ ?특파원', article)))
        cnt = 0
        for word in word_list:
            highlight_words = word.replace(word, '<font class="delete-word">' + word + '</font>')
            if cnt == 0:
                highlighted = re.sub(word, highlight_words, article)
            else:
                highlighted = re.sub(word, highlight_words, highlighted)
            cnt += 1
        for word in word_list:
            article = re.sub(word, '', article)

        text[str(idx)] = highlighted
        idx += 1
        word_list = []
        highlight_words = []
        cnt = 0

        # 5. 기자 이름 제거V2
        word_list = list(set(re.split(r'[^A-Za-z0-9가-힣]', str(df['name'][0]))))
        cnt = 0
        for word in word_list:
            highlight_words = word.replace(word, '<font class="delete-word">' + word + '</font>')
            if cnt == 0:
                highlighted = re.sub(word, highlight_words, article)
            else:
                highlighted = re.sub(word, highlight_words, highlighted)
            cnt += 1
        for word in word_list:
            article = re.sub(word, '', article)

        text[str(idx)] = highlighted
        idx += 1
        word_list = []
        highlight_words = []
        cnt = 0

        # 6. (지역=언론사) 양식
        word_list = list(set(re.findall('[A-Za-z가-힣 ]+=[가-힣0-9 ]+', article)))
        cnt = 0
        for word in word_list:
            highlight_words = word.replace(word, '<font class="delete-word">' + word + '</font>')
            if cnt == 0:
                highlighted = re.sub(word, highlight_words, article)
            else:
                highlighted = re.sub(word, highlight_words, highlighted)
            cnt += 1
        for word in word_list:
            article = re.sub(word, '', article)

        text[str(idx)] = highlighted
        idx += 1
        word_list = []
        highlight_words = []
        cnt = 0

        # 7. 홑 따옴표 안에 문자열 제거
        word_list = list(set(re.findall('‘.+?’', article)))
        cnt = 0
        for word in word_list:
            highlight_words = word.replace(word, '<font class="delete-word">' + word + '</font>')
            if cnt == 0:
                highlighted = re.sub(word, highlight_words, article)
            else:
                highlighted = re.sub(word, highlight_words, highlighted)
            cnt += 1
        for word in word_list:
            article = re.sub(word, '', article)

        text[str(idx)] = highlighted
        idx += 1
        word_list = []
        highlight_words = []
        cnt = 0

        # 8. 특수문자 제거(띄어쓰기)
        article = re.sub('[^A-Za-z0-9가-힣-]', ' ', article)
        article = article.strip()
        article = ' '.join(article.split())

        text[str(idx)] = article
        idx += 1

        # 9. 언론사 제거
        cnt = 0
        for word in l_source:
            highlight_words = word.replace(word, '<font class="delete-word">' + word + '</font>')
            if cnt == 0:
                highlighted = re.sub(word, highlight_words, article)
            else:
                highlighted = re.sub(word, highlight_words, highlighted)
            cnt += 1
        for word in l_source:
            article = re.sub(word, '', article)

        text[str(idx)] = highlighted
        idx += 1
        word_list = []
        highlight_words = []
        cnt = 0

        # 10. 한음절 글자 제거
        for i in range(5):
            cnt = 0
            word_list = list(set(re.findall(' . ', article)))
            if len(word_list) > 0:
                for word in word_list:
                    highlight_words = word.replace(word, '<font class="delete-word">' + word + '</font>')
                    if cnt == 0:
                        highlighted = re.sub(word, highlight_words, article)
                    else:
                        highlighted = re.sub(word, highlight_words, highlighted)
                    cnt += 1
                for word in word_list:
                    article = re.sub(word, ' ', article)
                    article = article.strip()
                    article = ' '.join(article.split())
                text[str(idx)] = highlighted
                idx += 1
            else:
                break
        word_list = []
        highlight_words = []
        cnt = 0

        # 11. 숫자만 있는 글자 제거
        word_list = list(set(re.findall(' [-]?[0-9]+ ', article)))
        cnt = 0
        for word in word_list:
            highlight_words = word.replace(word, '<font class="delete-word">' + word + '</font>')
            if cnt == 0:
                highlighted = re.sub(word, highlight_words, article)
            else:
                highlighted = re.sub(word, highlight_words, highlighted)
            cnt += 1
        for word in word_list:
            article = re.sub(word, '', article)
            article = article.strip()
            article = ' '.join(article.split())

        text[str(idx)] = highlighted
        idx += 1
        word_list = []
        highlight_words = []
        cnt = 0

        # 12. 숫자+글자(공백전까지) 제거(+,-)
        for i in range(5):
            cnt = 0
            word_list = list(set(re.findall(' [-]?[0-9]+[A-Za-z가-힣]+', article)))
            if len(word_list) > 0:
                for word in word_list:
                    highlight_words = word.replace(word, '<font class="delete-word">' + word + '</font>')
                    if cnt == 0:
                        highlighted = re.sub(word, highlight_words, article)
                    else:
                        highlighted = re.sub(word, highlight_words, highlighted)
                    cnt += 1
                for word in word_list:
                    article = re.sub(word, ' ', article)
                    article = article.strip()
                    article = ' '.join(article.split())
                text[str(idx)] = highlighted
                idx += 1
            else:
                break
        word_list = []
        highlight_words = []
        cnt = 0

        # 13. (-)+글자 제거
        for i in range(5):
            cnt = 0
            word_list = list(set(re.findall(' [-][A-Za-z가-힣]+ ', article)))
            if len(word_list) > 0:
                for word in word_list:
                    highlight_words = word.replace(word, '<font class="delete-word">' + word + '</font>')
                    if cnt == 0:
                        highlighted = re.sub(word, highlight_words, article)
                    else:
                        highlighted = re.sub(word, highlight_words, highlighted)
                    cnt += 1
                for word in word_list:
                    article = re.sub(word, ' ', article)
                    article = article.strip()
                    article = ' '.join(article.split())
                text[str(idx)] = highlighted
                idx += 1
            else:
                break
        word_list = []
        highlight_words = []
        cnt = 0

        # 14. 전처리 완료
        text[str(idx)] = article

    url = request.form.get("url")
    answer = text
    response = make_response(jsonify(answer))
    response.headers.add("Access-Control-Allow-Origin", "*")

    return response


if __name__ == '__main__':
    app.debug = True
    app.run()


