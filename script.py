from bs4 import BeautifulSoup
import re
from itertools import chain
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from groq import Groq
from pinecone import Pinecone as p , ServerlessSpec
from huggingface_hub import InferenceClient
import time
import json
from pypdf import PdfReader


class RAGmodule():
  def __init__(self, link: str, user_query, HG_TOKEN , P_TOKEN):
    '''
    link -  site link
    user_query -  texts from pdf doc
    HG_TOKEN-  Huggingface token
    P_TOKEN -  pinecone token
    '''
    def clean_text(text):
        soup = BeautifulSoup(text, "lxml")
        return re.sub(r"\n\n+", "  ", soup.text).strip()
    self.clean_text = clean_text
    self.links = [link]   ##pages to scrape information from
    self.user_query = user_query

    self.client = InferenceClient(model="thenlper/gte-small", token= HG_TOKEN)  #hugging face has daily limits.. upgrade might be needed after some time

    self.pc = p(api_key=P_TOKEN)

    self.index_name =  "semantic-search-fast"

    try:
      self.existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]
    except:
      raise Exception("Pinecone API key Error")
    self.spec = ServerlessSpec(cloud="aws", region="us-east-1")


    if self.index_name not in self.existing_indexes:
      self.pc.create_index(
          self.index_name,
          dimension=384,
          metric='dotproduct',
          spec=self.spec
      )
      while not self.pc.describe_index(self.index_name).status['ready']:
        time.sleep(1)

    self.index = self.pc.Index(self.index_name)


  def knowledge_base(self):
      loader = [RecursiveUrlLoader(url , extractor = self.clean_text).load() for url in self.links]
      docs = list(map(str , chain.from_iterable(loader)))

      text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=2)

      if len(docs) == 1:
         all_splits = text_splitter.split_text(docs[0])
      else:
          raise Exception("More than one page to scrape")
      try:
         embedding = [self.client.feature_extraction(chunk) for chunk in all_splits]
      except:
          raise Exception ("Hugging Face Daily Limit Reached or API key error")

      bill_rate_index =  [i for i in range(3,11)]
      assert len(embedding) == len(all_splits)

      for i in range(len(embedding)):
          '''
          if i in bill_rate_index:
              self.index.upsert(vectors = [{"id": f"id {i}", "values": embedding[i] , "metadata": {"class": "bill_rate"}}])
          else:
              self.index.upsert(vectors = [{"id": f"id {i}", "values": embedding[i] , "metadata": {"class": "other"}}])
          '''

          self.index.upsert(vectors = [{"id": f"id {i}", "values": embedding[i]}])
      return all_splits


  def take_user_query(self,  user_query : str):
      e = self.client.feature_extraction(user_query).tolist()
      return e

  def retriever(self):

      all_splits = self.knowledge_base()

      context = self.index.query(vector =  self.take_user_query(self.user_query), top_k=12, include_metadata=True)
      return [all_splits , context]


#########
def prompt(link: str, query, HG_TOKEN , P_TOKEN, G_KEY):

  #G_KEY -  groq api key
  instruction =  "You are muvar's AI assiatant, and you are to personalize you answers to the users and answer like youre their guider and help with utility bill questions and provide assistance to help save utility bills cost,you are to provide a short and consise overview and recommendations on ways to save money based on the user's bill while comparing it to the cheapest market rate provided through muvar"
  all_splits , context = RAGmodule(query).retriever()
  matchh = context["matches"]
  context_id = sorted ([int(matchh[i]["id"].split(" ")[-1]) for i in range(len(matchh))])
  contexts = [all_splits[i] for i in context_id]
  context = "\n\n".join(contexts)


  try:
    client = Groq(api_key = G_KEY)
  except:
    raise Exception("Groq API key Error")
  completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "you are an assistant built by Muvar "
        },
        {
            "role": "user",
            "content": f"process {context} to bring out energy rates comparisons from each company , dont give any opening statement or pleasantries just output the energy rate comparisons from each company "
        }
    ],
    temperature=1,
    max_completion_tokens=1024,
    top_p=1,
    stream=True,
    stop=None,
  )
  new_context= []
  for chunk in completion:
      new_context.append(chunk.choices[0].delta.content) if chunk.choices[0].delta.content is not None else None

  new_context2 = " ".join(new_context)

  prompt = f"observe the bill paid by me: [{query}],  then compare the administrative charges,natural gas,electricity paid.. do not include delivery charges in the comparison [Note* the electricity charge & natural gas paid should be the (rate * the how much i used (eg. rate of 9.59¢ / kWh * 150kWh used = 14.39$  )) or (total elctricity/natural gas charge) - delivery charges ] to the latest rates : [{new_context2}] which are the current market rates muvar offers.. when you compare, give a very short and consise overview of the bill and compare the bill paid to the rates offeredby muvar if there is any cheaper offer in the rates offered by muvar when compared to the bill paid.. recommend joining to joing muvar inc, because muvar has the best deals in the market, while pointing out the company which the cheaper rate.. else tell the user to join muvar for better future deals"
  return instruction , prompt

def main():
  def response(pdf_text , G_KEY):
    instruction ,  full_prompt = prompt(pdf_text)
  
    client = Groq(api_key = G_KEY)
    completion = client.chat.completions.create(
      model="llama-3.3-70b-versatile",
      messages=[
          {
              "role": "system",
              "content": instruction
          },
          {
              "role": "user",
              "content": full_prompt
          }
      ],
      temperature=1,
      max_completion_tokens=1024,
      top_p=1,
      stream=True,
      stop=None,
  )

    out = []
    for chunk in completion:
        out.append(chunk.choices[0].delta.content)
    if None in out:
      out.remove(None)

    response= " ".join(out)
    return response



if __name__ == "__main__":
    main()