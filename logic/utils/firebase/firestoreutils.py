from firebase_admin import firestore
import networkx as nx

# Initialize Firestore client
db = firestore.client()

# Create a new document in Firestore
def create_document(collection, data):
    doc_ref = db.collection(collection).document()
    doc_ref.set(data)
    return doc_ref.id

# Read a document from Firestore
def read_document(collection, document_id):
    doc_ref = db.collection(collection).document(document_id)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    else:
        return None

# Update a document in Firestore
def update_document(collection, document_id, data):
    doc_ref = db.collection(collection).document(document_id)
    return doc_ref.update(data)

# Delete a document from Firestore
def delete_document(collection, document_id):
    doc_ref = db.collection(collection).document(document_id)
    doc_ref.delete()

def write_graph_to_firestore(graph, collection, document_id):
    data = nx.node_link_data(graph)
    return update_document(collection, document_id, data)