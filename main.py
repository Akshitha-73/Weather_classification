import os
import pickle
from img2vec_pytorch import Img2Vec
import PIL 
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

img2vec=Img2Vec()

data='feature_extraction/dataset'
train=os.path.join(data,'train')
val=os.path.join(data,'val')

data={}
for j,dir_ in enumerate([train,val]):
    features=[]
    labels=[]
    for cat in os.listdir(dir_):
        for img_path in os.listdir(os.path.join(dir_,cat)):
            img_path_=os.path.join(dir_,cat,img_path)
            img=Image.open(img_path_).convert('RGB')

            img_features=img2vec.get_vec(img)
            features.append(img_features)
            labels.append(cat)

    data[['training_data','validation_data'][j]] = features 
    data[['training_labels','validation_labels'][j]]=labels

    

model = RandomForestClassifier(random_state=0)
model.fit(data['training_data'], data['training_labels'])


y_pred = model.predict(data['validation_data'])
score = accuracy_score(y_pred, data['validation_labels'])

print(score)

with open('./model.p', 'wb') as f:
    pickle.dump(model, f)
    f.close()

