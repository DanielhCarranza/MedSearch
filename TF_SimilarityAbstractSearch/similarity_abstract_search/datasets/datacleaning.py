import os
import json
import pandas as pd
import numpy as np
import h5py
from pathlib import Path 
from langid.langid import LanguageIdentifier, model
from similarity_abstract_search.utils import get_paper_set, get_raw_json_data, save_dict2json, load_json

Path.ls = lambda x: list(x.iterdir())

class TextDataCleaning:
  def __init__(self, data_path, data_files:list, paper_set_file=None):
    self.data_files, self.data_path = data_files, data_path
    if paper_set_file is not None:
      self.paper_set = get_paper_set(paper_set_file)
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    self.lang = lambda s : identifier.classify(str(s)) 

  def pruning(self, df)->pd.DataFrame:
    "Pruning: Reduce text without losing meaning"
    return df[df["totalCitation"].transform(lambda x: bool(set(x)&self.paper_set))]
  
  def remove_nonenglish(self, df, journalSet=set())->pd.DataFrame:
    jSet = journalSet|set(df['journalName'])
    vSet = journalSet|set(df['venue'])
    engJournals=[i for i in jSet if str(self.lang(i)).find('en') != -1 and self.lang(i)[1]>0.5]  
    engVenues = [i for i in vSet if str(self.lang(i)).find('en') != -1 and self.lang(i)[1]>0.5]  
    # use .loc to save the copy of the slicing 
    df = df[df['journalName'].isin(engJournals) | df['venue'].isin(engVenues)]
    df.drop(['journalName','venue'], axis=1, inplace=True)
    return df


  def embProcessing(self, df):
    df['citeEmbeddingsID']=df.totalCitation.map(lambda row: [self.paperID2EmbeddingID[i] 
                                      for i in row if self.paperID2EmbeddingID.get(i)])
    df['EmbeddingID'] = df['id'].map(self.paperID2EmbeddingID) 
    df = df.loc[:,['id','EmbeddingID', 'paperAbstract', 'title', 'citeEmbeddingsID']]
    return df
  

  @staticmethod
  def saveEmbedIDs(df, filename):
    embIDs = df.loc[:,['EmbeddingID', 'citeEmbeddingsID']]
    embIDs['citeEmbeddingsID'] = embIDs['citeEmbeddingsID'].apply(lambda x: np.array(x)) 
    
    h5f = h5py.File(f'{filename}.h5', 'w')
    dt = h5py.special_dtype(vlen=np.dtype('int32'))
    h5f.create_dataset('cites', data=embIDs.values[:,1], dtype=dt, compression='gzip', compression_opts=9, chunks=True, maxshape=(None,) )
    h5f.create_dataset('paper', data=embIDs.values[:,0].astype(np.int32), compression='gzip', compression_opts=9, chunks=True, maxshape=(None,) )
    h5f.close()
  

  def pruning_and_cleaning(self):
    # paper_set = set()
    for fn in self.data_files:
      df = get_raw_json_data(fn)
      print(fn)
      df = self.pruning(df)
      df = self.remove_nonenglish(df)
      # paper_set.update(set(df.id.values))
      df.to_json(self.data_path/f'pruned{fn.stem[-3:]}.json.gz', compression='gzip')
      os.remove(str(fn))

    #self.paperID2EmbeddingID = {id: idx for idx, id in enumerate(paper_set)}
    #save_dict2json(str(self.data_path.parent/'paperID2emb' ), self.paperID2EmbeddingID)

  def concat_files(self, paperID2EmbeddingID):
    df = pd.concat([pd.read_json(fn, compression='gzip', lines=True, chunksize=100) 
                    for fn in self.data_files])
    paper_set = load_json(paperID2EmbeddingID) 
    self.paperID2EmbeddingID = {id: idx for idx, id in enumerate(paper_set)}
    df  = self.embProcessing(df)
    df.to_json(self.data_path/'pruned_and_clean.json.gz', compression='gzip')

  
  def concat_filesV2(self, paperID2EmbeddingID):
    data = []
    self.paperID2EmbeddingID = load_json(paperID2EmbeddingID) 
    for fn in self.data_files:
        print(f'CONCAT {fn}')
        df = pd.read_json(fn, compression='gzip') 
        df  = self.embProcessing(df)
        data.append(df)
    df = pd.concat(data)
    df.to_json(self.data_path/'pruned_and_clean.json.gz', compression='gzip')

  

def main():
  INNER_PATH = Path(__file__).resolve().parents[3]/'Data/processed/SemanticScholarData'
  # PAPER_SET  = INNER_PATH.parent/'paper_set.txt'
  data_files = INNER_PATH.ls()
  # data_files = [fn for fn in data_files if fn.suffix=='.json']
  # import ipdb; ipdb.set_trace()
  data_cleaning = TextDataCleaning(INNER_PATH, data_files )
  # data_cleaning.pruning_and_cleaning()
  data_cleaning.concat_filesV2(str(INNER_PATH.parent/'paperID2emb.json'))

  
if __name__ == "__main__":
     main()
