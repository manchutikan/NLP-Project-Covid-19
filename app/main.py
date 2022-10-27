
from flask import Flask,request,abort
import requests
from app.Config import *
import json

from pythainlp.tokenize import sent_tokenize, word_tokenize
from pythainlp import Tokenizer
from pythainlp.util import dict_trie
from pythainlp.corpus.common import thai_words
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import tensorflow as tf


def load_model() :
    loaded_model = tf.keras.models.load_model('my_model')
    loaded_model.summary()
    return loaded_model

Model = load_model()

def load_w2vec_model():
    word2vec_model = KeyedVectors.load_word2vec_format('src/LTW2V_v0.1.bin', binary=True, unicode_errors='ignore')
    return word2vec_model

Word2vec_model = load_w2vec_model()

def map_word_index(word_seq):
    indices = [] 
    for word in word_seq:
        if word in Word2vec_model.vocab:
            indices.append(Word2vec_model.vocab[word].index + 1)
        else:
            indices.append(1)
    return indices


app=Flask(__name__)

@app.route('/webhook',methods=['POST','GET'])

def webhook():
    if request.method=='POST':
        payload =request.json
        Reply_token=payload['events'][0]['replyToken']
        message=payload['events'][0]['message']['text']
        Reply_text= prediction(message)
        print(Reply_text,flush=True)
        ReplyMessage(Reply_token,Reply_text,Channel_access_token)
        return request.json,200
    elif request.method=='GET':
        return "this is method GET!!!",200
    else:
        abort(400)


def prediction(message):
    words = ["covid", "covid19", "โควิด", "โควิด19", "sinovac", "sinopharm", "moderna", "astraZeneca", "Pfizer", "โรงบาล", "ไฮ", "โยว่", "Hi", "bot", "บอต", "อัลฟ่า",  "เบต้า", "เดลต้า", "โอไมครอน", "โอมิครอน", "omicron"]
    custom_words_list = set(thai_words())
    custom_words_list.update(words)
    trie = dict_trie(dict_source=custom_words_list)


    word_seq = word_tokenize(message, engine="newmm", custom_dict=trie)
    word_indices = map_word_index(word_seq)

    
    max_leng = Model.layers[0].output_shape[0][1]
    padded_wordindices = pad_sequences([word_indices], maxlen=max_leng, value=0)

    logit = Model.predict(padded_wordindices, batch_size=32)
    index = [ logit[0][pred] for pred in np.argmax(logit, axis=1) ][0]

    index_to_label = sorted(['อาการ', 'การป้องกัน', 'การดูแลตัวเองช่วงติดโควิด', 'vaccine', 'สถานการณ์ปัจจุบันในไทย', 'ข้อมูลสายพันธุ์โควิด', 'โรงพยาบาลใกล้ฉัน', 'greeting'])
    predict = [index_to_label[pred] for pred in np.argmax(logit, axis=1) ][0]

    if index > 0.65:
        if predict == "greeting":
            Reply_text ="สวัสดีค่ะ เราคือ chatbot ที่ให้ข้อมูลเกี่ยวกับโควิด 19 คุณสามารถถามเรื่อง \n-อาการของโควิด \n-การป้องกัน \n-การดูแลตัวเองเวลาติดโควิด \n-ข้อมูลของวัคซีนต่าง เช่น Sinovac, Sinopharm, Moderna, AstraZeneca, Pfizer \n-สถานการณ์ปัจจุบันในไทย \n-ข้อมูลสายพันธุ์โควิด, \n-โรงพยาบาลใกล้ที่คุณอยู่"    
        elif predict == "อาการ":
            Reply_text= "อาการทั่วไปมีดังนี้ \n- มีไข้ \n- ไอ\n- อ่อนเพลีย \n- สูญเสียความสามารถในการดมกลิ่นและรับรส \n\nอาการที่พบไม่บ่อยนักมีดังนี้ \n- เจ็บคอ \n- ปวดศีรษะ \n- ปวดเมื่อยเนื้อตัว \n- ท้องเสีย \n- มีผื่นบนผิวหนัง หรือนิ้วมือนิ้วเท้าเปลี่ยนสี \n- ตาแดงหรือระคายเคืองตา \n\nอาการรุนแรงมีดังนี้ \n- หายใจลำบากหรือหายใจถี่\n- สูญเสียความสามารถในการพูดหรือเคลื่อนไหว หรือมึนงง \n- เจ็บหน้าอก"
        elif predict == "การป้องกัน":
            Reply_text="วิธีป้องกันการแพร่ระบาดของโควิด-19 \n- รักษาระยะห่างที่ปลอดภัยจากผู้อื่น (อย่างน้อย 1 เมตร) แม้ว่าผู้นั้นจะไม่ได้ป่วยก็ตาม\n- สวมหน้ากากอนามัยในที่สาธารณะ โดยเฉพาะเมื่ออยู่ในพื้นที่ปิดหรือเว้นระยะห่างไม่ได้ \n- หลีกเลี่ยงพื้นที่ปิด พยายามอยู่ในพื้นที่เปิดโล่งและอากาศถ่ายเทสะดวก เปิดหน้าต่างเมื่ออยู่ในพื้นที่ปิด \n- ล้างมือบ่อยๆ โดยใช้สบู่และน้ำ หรือเจลล้างมือที่มีส่วนผสมหลักเป็นแอลกอฮอล์ \n- รับวัคซีนเมื่อได้รับสิทธิ์ ปฏิบัติตามหลักเกณฑ์ในพื้นที่เกี่ยวกับการฉีดวัคซีน \n- ปิดจมูกและปากด้วยข้อพับด้านในข้อศอกหรือกระดาษชำระเมื่อไอหรือจาม \n- เก็บตัวอยู่บ้านเมื่อรู้สึกไม่สบาย\n\nหากมีไข้ ไอ และหายใจลำบาก โปรดไปพบแพทย์ โดยติดต่อล่วงหน้าเพื่อที่ผู้ให้บริการด้านสุขภาพจะได้แนะนำให้คุณไปยังสถานพยาบาลที่ถูกต้อง ซึ่งจะช่วยปกป้องคุณ รวมถึงป้องกันการแพร่กระจายของไวรัสและการติดเชื้ออื่นๆ"
        elif predict == "การดูแลตัวเองช่วงติดโควิด":
            Reply_text="การดูแลตนเอง\n หลังจากสัมผัสกับผู้ที่ติดเชื้อโควิด-19 โปรดทำตามขั้นตอนต่อไปนี้\n - โทรหาผู้ให้บริการด้านการดูแลสุขภาพหรือสายด่วนโควิด-19 เพื่อหาสถานที่และเวลาเพื่อรับการตรวจ \n- ให้ความร่วมมือตามขั้นตอนการติดตามผู้สัมผัสเพื่อหยุดการแพร่กระจายของไวรัส \n- หากยังไม่ทราบผลตรวจ ให้อยู่บ้านและอยู่ห่างจากผู้อื่นเป็นเวลา 14 วัน \n- ขณะที่กักตัว อย่าออกไปที่ทำงาน โรงเรียน หรือสถานที่สาธารณะ ขอให้ผู้อื่นนำของอุปโภคบริโภคมาให้ \n-  รักษาระยะห่างจากผู้อื่นอย่างน้อย 1 เมตร แม้จะเป็นสมาชิกในครอบครัวก็ตาม \n - สวมหน้ากากอนามัยเพื่อป้องกันการแพร่เชื้อสู่ผู้อื่น รวมถึงในกรณีที่คุณต้องเข้ารับการรักษา \n- ล้างมือบ่อยๆ \n- กักตัวเองในห้องแยกจากสมาชิกครอบครัวคนอื่นๆ หากทำไม่ได้ ให้สวมหน้ากากอนามัย จัดให้ห้องมีอากาศถ่ายเทสะดวก \n- หากใช้ห้องร่วมกับผู้อื่น ให้จัดเตียงห่างกันอย่างน้อย 1 เมตร \n- สังเกตอาการตนเองเป็นเวลา 14 วัน \n- โทรหาผู้ให้บริการด้านการดูแลสุขภาพทันทีหากพบสัญญาณอันตรายต่อไปนี้ ได้แก่ หายใจลำบาก สูญเสียความสามารถในการพูดและเคลื่อนไหว แน่นหน้าอกหรือมีภาวะสับสน \n- ติดต่อกับคนที่คุณรักด้วยโทรศัพท์หรือทางออนไลน์ รวมถึงออกกำลังกายที่บ้าน เพื่อให้คุณมีสภาพจิตใจที่ดีอยู่เสมอ"
        elif predict == "vaccine":
            Reply_text = "คุณสามารถดูข้อมูลของวัคซีน Sinovac ได้ที่เว็บนี้เลยค่ะ \nhttps://hdmall.co.th/c/covid-vaccine-sinovac \n\nคุณสามารถดูข้อมูลของวัคซีน Sinopharm ได้ที่เว็บนี้ค่ะ \nhttps://hdmall.co.th/c/covid-vaccine-sinopharm \n\nคุณสามารถดูข้อมูลของวัคซีน Moderna ได้ที่เว็บนี้ค่ะ \nhttps://hdmall.co.th/c/covid-vaccine-moderna \n\nคุณสามารถดูข้อมูลของวัคซีน AstraZeneca ได้ที่เว็บนี้ค่ะ \nhttps://hdmall.co.th/c/covid-vaccine-astrazeneca \n\nคุณสามารถดูข้อมูลของวัคซีน Pfizer ได้ที่เว็บนี้ค่ะ \nhttps://hdmall.co.th/c/covid-vaccine-pfizer"
        elif predict == 'สถานการณ์ปัจจุบันในไทย':
            Reply_text="ดูสถาณการณ์ปัจจุบันในไทยได้ในเว็บใต้นี้ได้เลยค่ะ \n\n https://news.google.com/covid19/map?hl=th&mid=%2Fm%2F07f1x&gl=TH&ceid=TH%3Ath"
        elif predict == 'ข้อมูลสายพันธุ์โควิด':
            Reply_text="คุณสามารถเข้าไปดูข้อมูลสายพันธุ์โควิดได้ตามลิ้งนี้เลยค่ะ \n\nhttps://www.medicallinelab.co.th/%E0%B8%9A%E0%B8%97%E0%B8%84%E0%B8%A7%E0%B8%B2%E0%B8%A1/%E0%B8%AD%E0%B8%B1%E0%B8%9B%E0%B9%80%E0%B8%94%E0%B8%95-%E0%B8%AA%E0%B8%B2%E0%B8%A2%E0%B8%9E%E0%B8%B1%E0%B8%99%E0%B8%98%E0%B8%B8%E0%B9%8C%E0%B9%82%E0%B8%84%E0%B8%A7%E0%B8%B4%E0%B8%94-19-%E0%B9%83/"
        elif predict == 'โรงพยาบาลใกล้ฉัน':
            Reply_text="คุณสามารถกดลิ้งด้านล่างนี้ได้เลย มันสามารถพาคุณได้เจอกับโรงพยาบาลที่ใกล้คุณค่ะ \n\nhttps://www.google.co.th/maps/search/%E0%B9%82%E0%B8%A3%E0%B8%87%E0%B8%9E%E0%B8%A2%E0%B8%B2%E0%B8%9A%E0%B8%B2%E0%B8%A5%E0%B9%83%E0%B8%81%E0%B8%A5%E0%B9%89%E0%B8%89%E0%B8%B1%E0%B8%99"
    else: Reply_text="ไม่เข้าใจคำถามค่ะ"
    return Reply_text
        


def ReplyMessage(Reply_token,TextMessage,Line_Acees_Token):
    LINE_API='https://api.line.me/v2/bot/message/reply/'
    
    Authorization='Bearer {}'.format(Line_Acees_Token)
    print(Authorization)
    headers={
        'Content-Type':'application/json; char=UTF-8',
        'Authorization':Authorization
    }

    data={
        "replyToken":Reply_token,
        "messages":[{
            "type":"text",
            "text":TextMessage
        }
        ]
    }
    data=json.dumps(data) # ทำเป็น json
    r=requests.post(LINE_API,headers=headers,data=data)
    return 200