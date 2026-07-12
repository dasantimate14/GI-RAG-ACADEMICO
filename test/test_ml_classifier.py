# Ejecutar con: python tests/test_ml_classifier.py

import sys
sys.path.append(".")

from app.vector_store import VectorStore
from app.rag_chain import RAGChain
from app.ml_classifier import MLClassifier

def test_compute_embedding():
    chunks = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    classifier = MLClassifier.__new__(MLClassifier)
    resultado  = classifier._compute_document_embedding(chunks)
    assert len(resultado) == 3
    assert abs(resultado[0] - 0.333) < 0.01
    print("✅ _compute_document_embedding OK")

def test_insufficient_data():
    vs         = VectorStore()
    rc         = RAGChain(vs)
    classifier = MLClassifier(vs, rc)
    resultado  = classifier.train({
        "doc1.pdf": [0.1, 0.2],
        "doc2.pdf": [0.3, 0.4]
    })
    assert resultado["status"] == "insufficient_data"
    assert classifier.is_trained == False
    print("✅ insufficient_data guard OK")

def test_train_and_predict():
    vs         = VectorStore()
    rc         = RAGChain(vs)
    classifier = MLClassifier(vs, rc)

    # Embeddings con 2 grupos obvios en 2D
    embeddings = {
        "doc1.pdf": [1.0, 0.0],
        "doc2.pdf": [0.9, 0.1],
        "doc3.pdf": [0.0, 1.0],
        "doc4.pdf": [0.1, 0.9]
    }

    resultado = classifier.train(embeddings)
    assert resultado["status"] == "trained"
    assert resultado["silhouette_score"] > 0.5
    assert classifier.is_trained == True

    # doc1 y doc2 deben estar en el mismo cluster
    asig = resultado["cluster_assignments"]
    assert asig["doc1.pdf"]["cluster_id"] == asig["doc2.pdf"]["cluster_id"]
    assert asig["doc3.pdf"]["cluster_id"] == asig["doc4.pdf"]["cluster_id"]
    assert asig["doc1.pdf"]["cluster_id"] != asig["doc3.pdf"]["cluster_id"]

    # Predict: punto cercano a doc1 debe ir al mismo cluster
    pred = classifier.predict([0.95, 0.05])
    assert pred["cluster_id"] == asig["doc1.pdf"]["cluster_id"]
    assert len(pred["cluster_label"]) > 0

    print(f"✅ train y predict OK — silhouette: {resultado['silhouette_score']}")

def test_save_and_load():
    import os
    vs         = VectorStore()
    rc         = RAGChain(vs)
    classifier = MLClassifier(vs, rc)
    classifier.train({
        "a.pdf": [1.0, 0.0],
        "b.pdf": [0.9, 0.1],
        "c.pdf": [0.0, 1.0],
        "d.pdf": [0.1, 0.9]
    })
    classifier.save()
    assert os.path.exists("./data/ml_model.pkl")

    nuevo = MLClassifier(vs, rc)
    assert nuevo.is_trained == True
    assert nuevo.cluster_labels == classifier.cluster_labels
    print("✅ save y load OK")

if __name__ == "__main__":
    test_compute_embedding()
    test_insufficient_data()
    test_train_and_predict()
    test_save_and_load()
    print("\n✅ Todos los tests pasaron")