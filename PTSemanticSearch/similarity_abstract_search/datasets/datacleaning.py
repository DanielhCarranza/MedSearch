import os
import json
import pandas as pd
import numpy as np
import h5py
from pathlib import Path 
from langid.langid import LanguageIdentifier, model
from similarity_abstract_search import utils
Path.ls = lambda x: list(x.iterdir())

class TextDataCleaning:
  def __init__(self, data_path, data_files:list, paper_set_file=None):
    self.data_files, self.data_path = data_files, data_path
    if paper_set_file is not None:
      self.paper_set = utils.get_paper_set(paper_set_file)
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
    embIDs = df.loc[:,['EmbeddingID', 'citeEmbeddingsID']]
    embIDs['citeEmbeddingsID'] = embIDs['citeEmbeddingsID'].apply(lambda x: np.array(x)) 
    return df, embIDs
  
  
  def cleanEmbeddings(self): 
    embed_list=[]
    for i, fn in enumerate(self.data_files):
        print(f'Embedding Cleaning {i} {fn}')
        df = pd.read_json(fn, compression='gzip') 
        df, embIDs= self.embProcessing(df) 
        df.to_json(self.data_path/f'pruned_and_clean{i:003}.json.gz', compression='gzip')
        embed_list.append(embIDs)
        os.remove(fn)
    embIDs = pd.concat(embed_list)
    utils.saveEmbedIDs(embIDs.values[:,1], embIDs.values[:,0].astype(np.int32), self.data_path.parent/'embedIDs') 
  
  def dataCleaning(self, save_paper_id=False):
    paper_set = set()
    for fn in self.data_files:
      df = utils.get_raw_json_data(fn)
      print(fn)
      df = self.pruning(df)
      df = self.remove_nonenglish(df)
      paper_set.update(set(df.id.values))
      df.to_json(self.data_path/f'pruned{fn.stem[-3:]}.json.gz', compression='gzip')
      os.remove(str(fn))
    self.paperID2EmbeddingID = {id: idx for idx, id in enumerate(paper_set)}
    if save_paper_id:
      utils.save_dict2json(self.data_path.parent/'paperID2emb', self.paperID2EmbeddingID)
    self.data_files = self.data_path.ls()
    self.cleanEmbeddings()

def main():
  INNER_PATH = Path(__file__).resolve().parents[3]/'Data/processed/SemanticScholarData'
  PAPER_SET  = INNER_PATH.parent/'paper_set.txt'
  data_files = INNER_PATH.ls()
  data_cleaning = TextDataCleaning(INNER_PATH, data_files, PAPER_SET)
  data_cleaning.dataCleaning()

  
if __name__ == "__main__":
     main()
