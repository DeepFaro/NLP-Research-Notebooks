import string
import unicodedata

import MeCab

jpn_stop_words = ["あそこ","あっ","あの","あのかた","あの人","あり","あります","ある","あれ","い","いう","います","いる","う","うち","え","お","および","おり","おります","か","かつて","から","が","き","ここ","こちら","こと","この","これ","これら","さ","さらに","し","しかし","する","ず","せ","せる","そこ","そして","その","その他","その後","それ","それぞれ","それで","た","ただし","たち","ため","たり","だ","だっ","だれ","つ","て","で","でき","できる","です","では","でも","と","という","といった","とき","ところ","として","とともに","とも","と共に","どこ","どの","な","ない","なお","なかっ","ながら","なく","なっ","など","なに","なら","なり","なる","なん","に","において","における","について","にて","によって","により","による","に対して","に対する","に関する","の","ので","のみ","は","ば","へ","ほか","ほとんど","ほど","ます","また","または","まで","も","もの","ものの","や","よう","より","ら","られ","られる","れ","れる","を","ん","何","及び","彼","彼女","我々","特に","私","私達","貴方","貴方方"]

class Rake:
    def __init__(self):
        self.tagger = MeCab.Tagger("-Owakati")
    
    def remove_punctuation(self,text):
        text = unicodedata.normalize("NFKC", text)  # 全角記号をざっくり半角へ置換（でも不完全）
        # 記号を消し去るための魔法のテーブル作成
        table = str.maketrans("", "", string.punctuation  + "「」、。・※" + string.digits)
        text = text.translate(table)

        return text
        
    def get_word_score(self, word_list):
        freq = {}
        deg = {}

        for word in word_list:
            freq[word] = (freq.get(word) or 0) + 1
            deg[word] = (deg.get(word) or 0) + len(word) - 1 # word length must be > 1 to be considered as a Japanese 'word'
      
        scores = {}
        for word in word_list:
            scores[word] = deg[word]/freq[word]
        
        scores = {k:v for k, v in  sorted(scores.items(), key=lambda item: item[1], reverse=True)}
      
        return scores
    
    def get_keywords(self, text, limit=0):
        parsed_text = self.tagger.parse(text)
        raw_word_list = self.remove_punctuation(parsed_text).split()
        word_list = [word for word in raw_word_list if word not in jpn_stop_words ]
        
        score_list = self.get_word_score(word_list)
   
        return score_list
