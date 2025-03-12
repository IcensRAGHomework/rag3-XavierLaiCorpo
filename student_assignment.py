import datetime
import chromadb
import traceback
import pandas

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"
csv_path = "./COA_OpenData.csv"

def generate_hw01():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    if collection.count() == 0:
        df = pandas.read_csv(csv_path)

        for idx, row in df.iterrows():
            metadata = {
                "file_name": csv_path,
                "name": row["Name"],
                "type": row["Type"],
                "address": row["Address"],
                "tel": row["Tel"],
                "city": row["City"],
                "town": row["Town"],
                "date": int(datetime.datetime.strptime(row['CreateDate'], '%Y-%m-%d').timestamp())
            }
            print(f"{idx} - {metadata['name']}")
            
            collection.add(
                ids=[str(idx)],
                metadatas=[metadata],
                documents=[row["HostWords"]]
            )
    return collection
    
def generate_hw02(question, city, store_type, start_date, end_date):
    print(f"question={question}, city={city}, store_type={store_type}, start_date={start_date}, end_date={end_date}")

    collection = generate_hw01()
    results = collection.query(
        query_texts=question,
        n_results=10, 
        include=["metadatas", "distances"],
        where={
            "$and": [
                {"date": {"$gte": int(start_date.timestamp())}},
                {"date": {"$lte": int(end_date.timestamp())}},
                {"type": {"$in": store_type}},
                {"city": {"$in": city}}
            ]
        }
    )
    target_similarity = 0.8
    return filterQueryResult(results, target_similarity)
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    print(f"question={question}, store_name={store_name}, new_store_name={new_store_name}, city={city}, store_type={store_type}")

    collection = generate_hw01()
    target_store = collection.get(where={"name": store_name})
    metadatas = [{**meta, "new_store_name": new_store_name} for meta in target_store.get("metadatas", [])]
    collection.upsert(ids=target_store.get("ids", []), 
                      metadatas=metadatas,
                      documents=target_store.get("documents", []))
    
    results = collection.query(
        query_texts=[question],
        n_results=10,
        include=["metadatas", "distances"],
        where={
            "$and": [
                {"type": {"$in": store_type}},
                {"city": {"$in": city}}
            ]
        }
    )
    target_similarity = 0.8
    return filterQueryResult(results, target_similarity)

def filterQueryResult(results, target_similarity):
    filtered_similarity = []
    filtered_store_name = []
    for idx in range(len(results['ids'])):
        for distance, metadata in zip(results['distances'][idx], results['metadatas'][idx]):
            # the higher the better
            similarity = 1 - distance
            #print(f"{metadata['name']} - {similarity}")
            if similarity >= target_similarity:
                new_name = metadata.get("new_store_name", "")
                name = metadata["name"]
                filtered_store_name.append(new_name if new_name else name)
                filtered_similarity.append(similarity)

    filtered_results = sorted(zip(filtered_similarity, filtered_store_name), key=lambda x: x[0], reverse=True)
    sorted_store_names = [name for _, name in filtered_results]
    
    print(sorted_store_names)
    return sorted_store_names


if __name__ == "__main__":
    #generate_hw01()
    generate_hw02("我想要找有關茶餐點的店家",
                  ["宜蘭縣", "新北市"],
                  ["美食"],
                  datetime.datetime(2024, 4, 1), 
                  datetime.datetime(2024, 5, 1))
    generate_hw03("我想要找南投縣的田媽媽餐廳，招牌是蕎麥麵", 
                  "耄饕客棧", 
                  "田媽媽（耄饕客棧）", 
                  ["南投縣"], ["美食"])