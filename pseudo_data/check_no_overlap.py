
import jieba
import json
from tqdm import tqdm

def read_file(file, test=True):
    with open(file,'r',encoding='utf8')as fr:
        lines = fr.readlines()
    if test:
        return [list(jieba.cut(line.strip()))for line in lines]
    else:
        return [list(jieba.cut(line.strip().split('\t')[1])) for line in lines]
    
def load_json(file):
    with open(file,'r',encoding='utf8')as fr:
        data = json.load(fr)
    return data

def overlap_detect(cons_sent, sent_list):
    cons_sents_chat = set(cons_sent)
    insert_sentence = []
    for sent in sent_list:
        insert_words = cons_sents_chat&set(sent)
        if len(cons_sents_chat)-len(insert_words)<=2:
            insert_sentence.append(sent)
    return insert_sentence

def check(construct_file,dev_file,test_file,overlap_file):
    
    test_dev_Sentences =read_file(test_file)+read_file(dev_file,test=False)
    
    data  = load_json(construct_file)
    with open(overlap_file,'w',encoding='utf8')as fw:
        for da in tqdm(data):
            input = da['input']
            input_words = list(jieba.cut(input))
            over_lap_sentences = overlap_detect(input_words,test_dev_Sentences)
            if over_lap_sentences:
                for sent in over_lap_sentences:
                    fw.write(input+'\t'+''.join(sent)+'\n')
            
        
check(construct_file='instruction.json',
      dev_file='nacgec.dev.ref.para',
      test_file='nacgec.test.input',
      overlap_file='overlap.txt')

