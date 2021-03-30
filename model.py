import itertools
from utils import clean_text


class ModelHandler:
    def __init__(self):
        #모델이 받았을때 긍정이냐 부정이냐 해줌 
        self.id2label = {0: 'negative', 1: 'positive'}

    def _clean_text(self, text):
        #데이터 전처리부분
        model_input = []
        if isinstance(text, str):
            cleaned_text = clean_text(text)
            model_input.append(cleaned_text)
        elif isinstance(text, (list, tuple)) and len(text) > 0 and (all(isinstance(t, str) for t in text)):
            cleaned_text = itertools.chain((clean_text(t) for t in text))
            model_input.extend(cleaned_text)
        else:
            model_input.append('')
        return model_input


class MLModelHandler(ModelHandler):
    def __init__(self):
        super().__init__()
        self.initialize()

    def initialize(self, ):
        # De-serializing model and loading vectorizer
        # 초기화 한것을 configuration 불러옴
        # 사전 학습한 모델의 파라미터를 불러옴
        # 모델을 전역변수로 불러와야함
        import joblib
        self.model = joblib.load('model/ml_model.pkl')
        self.vectorizer = joblib.load('model/ml_vectorizer.pkl')
        pass

    def preprocess(self, text):
        # cleansing raw text
        model_input = self._clean_text(text)

        # vectorizing cleaned text
        model_input = self.vectorizer.transform(model_input)
        return model_input

    def inference(self, model_input):
        # get predictions from model as probabilities
        model_output = self.model.predict_proba(model_input)

        return model_output

    def postprocess(self, model_output):
        # process predictions to predicted label and output format
        # 모델 성능 보정
        predicted_probabilities = model_output.max(axis = 1)
        predicted_ids = model_output.argmax(axis = 1)
        predicted_labels = [self.id2label[id_] for id_ in predicted_ids]
        return predicted_labels, predicted_probabilities 

    def handle(self, data):
        #데이터 전처리
        model_input = self.preprocess(data)
        #추론
        model_output = self.inference(model_input)
        #결과 보내기
        return self.postprocess(model_output)



class DLModelHandler(ModelHandler):
    def __init__(self):
        super().__init__()
        self.initialize()

    def initialize(self, ):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.model_name_or_path = 'sackoh/bert-base-multilingual-cased-nsmc'
        self.tokenzier = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path)
        self.model.to('cpu')

    def preprocess(self, data):
        model_input =self._clean_text(text)
        model_input = self.tokenzier(text, return_tensors='pt', padding = True)
        return data 


    def inference(self, model_input):
        with torch.no_grad():
            model_output = self.model(**model_input)[0].cpu()
            model_output = 1.0/(1.0 + torch.exp(-model_output))
            model_output = model.output.numpy().astype('flaot')
        return model_output

    def postprocess(self, model_output):
        # process predictions to predicted label and output format
        # 모델 성능 보정
        predicted_probabilities = model_output.max(axis = 1)
        predicted_ids = model_output.argmax(axis = 1)
        predicted_labels = [self.id2label[id_] for id_ in predicted_ids]
        return predicted_labels, predicted_probabilities 

    def handle(self, data):
        #데이터 전처리
        model_input = self.preprocess(data)
        #추론
        model_output = self.inference(model_input)
        #결과 보내기
        return self.postprocess(model_output)
