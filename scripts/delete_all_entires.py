from pymongo import MongoClient, errors

def get_mongo_client(uri='mongodb://localhost:27017/', retries=5, wait_time=5):
    while retries > 0:
        try:
            client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            client.server_info()  # Trigger exception if cannot connect to MongoDB
            print("Connected to MongoDB")
            return client
        except errors.ServerSelectionTimeoutError as err:
            print(f"Error connecting to MongoDB: {err}")
            retries -= 1
            if retries > 0:
                print(f"Retrying in {wait_time} seconds... ({retries} retries left)")
                time.sleep(wait_time)
            else:
                print("Exceeded maximum retries. Exiting.")
                raise

def delete_all_entries(db_name, collection_name):
    try:
        client = get_mongo_client()
        db = client[db_name]
        collection = db[collection_name]
        print(f"Deleting all entries from database '{db_name}', collection '{collection_name}'")
        result = collection.delete_many({})
        print(f"Deleted {result.deleted_count} documents.")
        client.close()
    except errors.PyMongoError as e:
        print(f"Error deleting entries from MongoDB: {e}")

if __name__ == "__main__":
    db_name = 'detections_db'
    collection_name = 'detections'
    delete_all_entries(db_name, collection_name)
