import os
import gc
import json
import time
import numpy as np
import torch
import faiss
import requests as http_requests
import sqlite3
from typing import List, Dict, Any
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from transformers import pipeline as hf_pipeline
import warnings
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify
from queue import Queue
import threading

# Read environment variables
TOTAL_NODES = int(os.environ.get("TOTAL_NODES", 1))
NODE_NUMBER = int(os.environ.get("NODE_NUMBER", 0))
NODE_0_IP = os.environ.get("NODE_0_IP", "localhost:8000")
NODE_1_IP = os.environ.get("NODE_1_IP", "localhost:8000")
NODE_2_IP = os.environ.get("NODE_2_IP", "localhost:8000")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
DOCUMENTS_DIR = os.environ.get("DOCUMENTS_DIR", "documents/")

# Configuration
CONFIG = {
    "faiss_index_path": FAISS_INDEX_PATH,
    "documents_path": DOCUMENTS_DIR,
    "faiss_dim": 768,  # You must use this dimension
    "max_tokens": 128,  # You must use this max token limit
    "retrieval_k": 10,  # You must retrieve this many documents from the FAISS index
    "truncate_length": 512,  # You must use this truncate length
}

# Flask app
app = Flask(__name__)

# Request queue and results storage
request_queue = Queue()
results = {}
results_lock = threading.Lock()


@dataclass
class PipelineRequest:
    request_id: str
    query: str
    timestamp: float


@dataclass
class PipelineResponse:
    request_id: str
    generated_response: str
    sentiment: str
    is_toxic: str
    processing_time: float


class MonolithicPipeline:
    """
    Deliberately inefficient monolithic pipeline
    """

    def __init__(self):
        self.device = torch.device("cpu")
        print(f"Initializing pipeline on {self.device}")
        print(f"Node {NODE_NUMBER}/{TOTAL_NODES}")
        print(f"FAISS index path: {CONFIG['faiss_index_path']}")
        print(f"Documents path: {CONFIG['documents_path']}")

        # Model names
        self.embedding_model_name = "BAAI/bge-base-en-v1.5"
        self.reranker_model_name = "BAAI/bge-reranker-base"
        self.llm_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.safety_model_name = "unitary/toxic-bert"
        # Initialize all models to None
        self.embedding_model = None
        self.faiss_index = None
        self.reranker_tokenizer = None
        self.reranker_model = None
        self.llm_model = None
        self.llm_tokenizer = None
        self.sentiment_classifier = None
        self.safety_classifier = None

        # Node 0: Frontend + embedder + orchestration
        if NODE_NUMBER == 0:
            print("Loading embedding model for Node 0...")
            self.embedding_model = SentenceTransformer(self.embedding_model_name).to(
                self.device
            )
            self.embedding_model.eval()
            print("Embedding model loaded!")

        # Node 1: FAISS + document retrieval + reranking
        elif NODE_NUMBER == 1:
            print("Loading FAISS index and reranker for Node 1...")

            # Load FAISS index
            if os.path.exists(CONFIG["faiss_index_path"]):
                print("Loading FAISS index...")
                self.faiss_index = faiss.read_index(CONFIG["faiss_index_path"])
                print("FAISS index loaded!")
            else:
                raise FileNotFoundError(
                    f"FAISS index not found at {CONFIG['faiss_index_path']}"
                )

            # Load reranker
            print("Loading reranker model...")
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                self.reranker_model_name
            )
            self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                self.reranker_model_name
            ).to(self.device)
            self.reranker_model.eval()
            print("Reranker model loaded!")

        # Node 2: LLM + sentiment + sensitivity filters
        elif NODE_NUMBER == 2:
            print("Loading LLM, sentiment, and safety models for Node 2...")

            # Load LLM
            print("Loading LLM model...")
            try:
                # Try torch_dtype (newer transformers API)
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name,
                    dtype=torch.float16,
                ).to(self.device)
            except TypeError:
                # Fallback for older transformers versions
                print("Using fallback method for LLM loading...")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_name
                ).to(self.device)
                self.llm_model = self.llm_model.half()  # Convert to float16

            self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            print("LLM model loaded!")

            # Sentiment and safety models will be loaded lazily when needed
            # (They're initialized as None above)

        else:
            raise ValueError(f"Invalid NODE_NUMBER: {NODE_NUMBER}. Must be 0, 1, or 2.")

        print(f"Node {NODE_NUMBER} initialization complete!")

    def process_request(self, request: PipelineRequest) -> PipelineResponse:
        """
        Backwards-compatible single-request entry point that delegates
        to the batch processor with a batch size of 1.
        """
        responses = self.process_batch([request])
        return responses[0]

    def process_batch(self, requests: List[PipelineRequest]) -> List[PipelineResponse]:
        """
        Main pipeline execution for a batch of requests.
        """
        if not requests:
            return []

        batch_size = len(requests)
        start_times = [time.time() for _ in requests]
        queries = [req.query for req in requests]

        print("\n" + "=" * 60)
        print(f"Processing batch of {batch_size} requests")
        print("=" * 60)
        for request in requests:
            print(f"- {request.request_id}: {request.query[:50]}...")

        # Step 1: Generate embeddings
        print("\n[Step 1/7] Generating embeddings for batch...")
        query_embeddings = self._generate_embeddings_batch(queries)

        # Step 2: FAISS ANN search
        print("\n[Step 2/7] Performing FAISS ANN search for batch...")
        doc_id_batches = self._faiss_search_batch(query_embeddings)

        # Step 3: Fetch documents from disk
        print("\n[Step 3/7] Fetching documents for batch...")
        documents_batch = self._fetch_documents_batch(doc_id_batches)

        # Step 4: Rerank documents
        print("\n[Step 4/7] Reranking documents for batch...")
        reranked_docs_batch = self._rerank_documents_batch(queries, documents_batch)

        # Step 5: Generate LLM responses
        print("\n[Step 5/7] Generating LLM responses for batch...")
        responses_text = self._generate_responses_batch(queries, reranked_docs_batch)

        # Step 6: Sentiment analysis
        print("\n[Step 6/7] Analyzing sentiment for batch...")
        sentiments = self._analyze_sentiment_batch(responses_text)

        # Step 7: Safety filter on responses
        print("\n[Step 7/7] Applying safety filter to batch...")
        toxicity_flags = self._filter_response_safety_batch(responses_text)

        responses = []
        for idx, request in enumerate(requests):
            processing_time = time.time() - start_times[idx]
            print(
                f"\n✓ Request {request.request_id} processed in {processing_time:.2f} seconds"
            )
            sensitivity_result = "true" if toxicity_flags[idx] else "false"
            responses.append(
                PipelineResponse(
                    request_id=request.request_id,
                    generated_response=responses_text[idx],
                    sentiment=sentiments[idx],
                    is_toxic=sensitivity_result,
                    processing_time=processing_time,
                )
            )

        return responses

    def _generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        if self.embedding_model is None:
            raise RuntimeError("Embedding model not loaded on this node")
        with torch.no_grad():
            embeddings = self.embedding_model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True
            )
        return embeddings

    def _faiss_search_batch(self, query_embeddings: np.ndarray) -> List[List[int]]:
        """Step 3: Perform FAISS ANN search for a batch of embeddings"""
        if self.faiss_index is None:
            raise FileNotFoundError("FAISS index not loaded")
        query_embeddings = query_embeddings.astype("float32")
        _, indices = self.faiss_index.search(query_embeddings, CONFIG["retrieval_k"])
        return [row.tolist() for row in indices]

    def _fetch_documents_batch(
        self, doc_id_batches: List[List[int]]
    ) -> List[List[Dict]]:
        """Step 4: Fetch documents for each query in the batch using SQLite"""
        db_path = f"{CONFIG['documents_path']}/documents.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        documents_batch = []
        for doc_ids in doc_id_batches:
            documents = []
            for doc_id in doc_ids:
                cursor.execute(
                    "SELECT doc_id, title, content, category FROM documents WHERE doc_id = ?",
                    (doc_id,),
                )
                result = cursor.fetchone()
                if result:
                    documents.append(
                        {
                            "doc_id": result[0],
                            "title": result[1],
                            "content": result[2],
                            "category": result[3],
                        }
                    )
            documents_batch.append(documents)
        conn.close()
        return documents_batch

    def _rerank_documents_batch(
        self, queries: List[str], documents_batch: List[List[Dict]]
    ) -> List[List[Dict]]:
        """Step 5: Rerank retrieved documents for each query in the batch"""
        if self.reranker_model is None or self.reranker_tokenizer is None:
            raise RuntimeError("Reranker model not loaded on this node")
        reranked_batches = []
        for query, documents in zip(queries, documents_batch):
            if not documents:
                reranked_batches.append([])
                continue
            pairs = [[query, doc["content"]] for doc in documents]
            with torch.no_grad():
                inputs = self.reranker_tokenizer(
                    pairs, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)
                scores = (
                    self.reranker_model(**inputs, return_dict=True)
                    .logits.view(
                        -1,
                    )
                    .float()
                )
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_batches.append([doc for doc, _ in doc_scores])
        return reranked_batches

    def _generate_responses_batch(
        self, queries: List[str], documents_batch: List[List[Dict]]
    ) -> List[str]:
        """Step 6: Generate LLM responses for each query in the batch"""
        if self.llm_model is None or self.llm_tokenizer is None:
            raise RuntimeError("LLM model not loaded on this node")
        responses = []
        for query, documents in zip(queries, documents_batch):
            context = "\n".join(
                [f"- {doc['title']}: {doc['content'][:200]}" for doc in documents[:3]]
            )
            messages = [
                {
                    "role": "system",
                    "content": "When given Context and Question, reply as 'Answer: <final answer>' only.",
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
                },
            ]
            text = self.llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = self.llm_tokenizer([text], return_tensors="pt").to(
                self.device
            )
            generated_ids = self.llm_model.generate(
                **model_inputs,
                max_new_tokens=CONFIG["max_tokens"],
                temperature=0.01,
                pad_token_id=self.llm_tokenizer.eos_token_id,
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.llm_tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            responses.append(response)
        return responses

    def _analyze_sentiment_batch(self, texts: List[str]) -> List[str]:
        """Step 7: Analyze sentiment for each generated response"""
        if self.sentiment_classifier is None:
            self.sentiment_classifier = hf_pipeline(
                "sentiment-analysis",
                model=self.sentiment_model_name,
                device=self.device,
            )
        truncated_texts = [text[: CONFIG["truncate_length"]] for text in texts]
        raw_results = self.sentiment_classifier(truncated_texts)
        sentiment_map = {
            "1 star": "very negative",
            "2 stars": "negative",
            "3 stars": "neutral",
            "4 stars": "positive",
            "5 stars": "very positive",
        }
        sentiments = []
        for result in raw_results:
            sentiments.append(sentiment_map.get(result["label"], "neutral"))
        return sentiments

    def _filter_response_safety_batch(self, texts: List[str]) -> List[bool]:
        """Step 8: Filter responses for safety for each entry in the batch"""
        if self.safety_classifier is None:
            self.safety_classifier = hf_pipeline(
                "text-classification", model=self.safety_model_name, device=self.device
            )
        truncated_texts = [text[: CONFIG["truncate_length"]] for text in texts]
        raw_results = self.safety_classifier(truncated_texts)
        toxicity_flags = []
        for result in raw_results:
            toxicity_flags.append(result["score"] > 0.5)
        return toxicity_flags


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return (
        jsonify({"status": "healthy", "node": NODE_NUMBER, "total_nodes": TOTAL_NODES}),
        200,
    )


def run_gateway():
    # current queue/worker + /query routes
    print("Initializing pipeline...")
    pipeline = MonolithicPipeline()

    def process_requests_worker():
        """Worker thread that processes requests from the queue"""
        BATCH_SIZE = 4
        BATCH_TIMEOUT = 0.1
        while True:
            try:
                batch = []
                batch_start_time = time.time()

                # Get first request (blocking)
                first_request = request_queue.get()
                if first_request is None:  # Shutdown signal
                    break
                batch.append(first_request)

                # Try to collect more requests (non-blocking with timeout)
                while len(batch) < BATCH_SIZE:
                    try:
                        elapsed = time.time() - batch_start_time
                        remaining_timeout = max(0, BATCH_TIMEOUT - elapsed)

                        if remaining_timeout > 0:
                            request_data = request_queue.get(timeout=remaining_timeout)
                            if request_data is None:
                                break
                            batch.append(request_data)
                        else:
                            break  # Timeout reached, process what we have
                    except:
                        break  # Queue empty, process what we have

                if not batch:
                    continue

                print(f"[Node 0] Processing batch of {len(batch)} requests")

                # Extract batch data
                request_ids = [req["request_id"] for req in batch]
                queries = [req["query"] for req in batch]

                # Step 1: Generate embeddings for batch
                query_embeddings = pipeline._generate_embeddings_batch(queries)
                # Call Node 1
                node1_response = http_requests.post(
                    f"http://{NODE_1_IP}/retrieve",
                    json={
                        "request_ids": request_ids,
                        "queries": queries,
                        "embeddings": query_embeddings.tolist(),
                    },
                    timeout=300,
                )
                node1_data = node1_response.json()

                # Call Node 2
                node2_response = http_requests.post(
                    f"http://{NODE_2_IP}/generate",
                    json={
                        "request_ids": request_ids,
                        "queries": queries,
                        "documents": node1_data["documents"],  # List of document lists
                    },
                    timeout=300,
                )
                node2_data = node2_response.json()

                # Store result
                with results_lock:
                    # Handle both single and batch responses
                    if "results" in node2_data:
                        # Batch response
                        for result in node2_data["results"]:
                            results[result["request_id"]] = {
                                "request_id": result["request_id"],
                                "generated_response": result["generated_response"],
                                "sentiment": result["sentiment"],
                                "is_toxic": result["is_toxic"],
                            }
                    else:
                        # Single response (backward compatibility)
                        results[request_ids[0]] = {
                            "request_id": request_ids[0],
                            "generated_response": node2_data["generated_response"],
                            "sentiment": node2_data["sentiment"],
                            "is_toxic": node2_data["is_toxic"],
                        }

                # Mark all tasks as done
                for _ in batch:
                    request_queue.task_done()
            except Exception as e:
                print(f"Error processing request: {e}")
                request_queue.task_done()

    worker_thread = threading.Thread(target=process_requests_worker, daemon=True)
    worker_thread.start()
    print("Worker thread started!")
    app = Flask("gateway")

    @app.route("/query", methods=["POST"])
    def handle_query():
        """Handle incoming query requests"""
        try:
            data = request.json
            request_id = data.get("request_id")
            query = data.get("query")

            if not request_id or not query:
                return jsonify({"error": "Missing request_id or query"}), 400

            # Check if result already exists (request already processed)
            with results_lock:
                if request_id in results:
                    return jsonify(results[request_id]), 200

            print(f"queueing request {request_id}")
            # Add to queue
            request_queue.put({"request_id": request_id, "query": query})

            # Wait for processing (with timeout). Very inefficient - would suggest using a more efficient waiting and timeout mechanism.
            timeout = 300  # 5 minutes
            start_wait = time.time()
            while True:
                with results_lock:
                    if request_id in results:
                        result = results.pop(request_id)
                        return jsonify(result), 200

                if time.time() - start_wait > timeout:
                    return jsonify({"error": "Request timeout"}), 504

                time.sleep(0.1)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/health", methods=["GET"])
    def gateway_health():
        return jsonify({"status": "healthy", "role": "gateway"}), 200

    host = NODE_0_IP.split(":")[0]
    port = int(NODE_0_IP.split(":")[1]) if ":" in NODE_0_IP else 8000
    app.run(host=host, port=port, threaded=True)


def run_retriever():
    app = Flask("retriever")
    pipeline: MonolithicPipeline = MonolithicPipeline()

    @app.route("/retrieve", methods=["POST"])
    def retrieve():
        data = request.json

        # Handle both single request and batch
        if "request_ids" in data:
            # Batch mode
            request_ids = data.get("request_ids")
            queries = data.get("queries")
            embeddings_list = data.get(
                "embeddings"
            )  # List of embedding lists [[...], [...], ...]
        else:
            # Single request mode (backward compatibility)
            request_ids = [data.get("request_id")]
            queries = [data.get("query")]
            embeddings_list = [data.get("embeddings")]
        embeddings_array = np.array(embeddings_list, dtype=np.float32)

        doc_id_batches = pipeline._faiss_search_batch(embeddings_array)
        documents_batch = pipeline._fetch_documents_batch(doc_id_batches)
        reranked_docs_batch = pipeline._rerank_documents_batch(queries, documents_batch)
        return (
            jsonify(
                {
                    "documents": reranked_docs_batch  # List of document lists, one per request
                }
            ),
            200,
        )

    @app.route("/health", methods=["GET"])
    def retriever_health():
        return jsonify({"status": "healthy", "role": "retriever"}), 200

    app.run(host=NODE_1_IP.split(":")[0], port=int(NODE_1_IP.split(":")[1]))


def run_generator():
    app = Flask("generator")
    pipeline = MonolithicPipeline()

    @app.route("/generate", methods=["POST"])
    def generate():
        data = request.json
        if "request_ids" in data:
            # Batch mode
            request_ids = data.get("request_ids")
            queries = data.get("queries")
            documents_batch = data.get("documents")  # Already list of lists
        else:
            # Single request mode (backward compatibility)
            request_ids = [data.get("request_id")]
            queries = [data.get("query")]
            documents = data.get("documents")
            # Ensure it's a list of lists
            if documents and not isinstance(documents[0], list):
                documents_batch = [documents]
            else:
                documents_batch = [documents] if documents else [[]]
        responses_text = pipeline._generate_responses_batch(queries, documents_batch)
        sentiments = pipeline._analyze_sentiment_batch(responses_text)
        toxicity_flags = pipeline._filter_response_safety_batch(responses_text)
        is_toxic_list = ["true" if flag else "false" for flag in toxicity_flags]

        # Return batch response
        if len(request_ids) == 1:
            # Single request - return single object (backward compatibility)
            return (
                jsonify(
                    {
                        "generated_response": responses_text[0],
                        "sentiment": sentiments[0],
                        "is_toxic": is_toxic_list[0],
                    }
                ),
                200,
            )
        else:
            # Batch - return list of results
            results = []
            for i in range(len(request_ids)):
                results.append(
                    {
                        "request_id": request_ids[i],
                        "generated_response": responses_text[i],
                        "sentiment": sentiments[i],
                        "is_toxic": is_toxic_list[i],
                    }
                )
            return jsonify({"results": results}), 200

    @app.route("/health", methods=["GET"])
    def generator_health():
        return jsonify({"status": "healthy", "role": "generator"}), 200

    app.run(host=NODE_2_IP.split(":")[0], port=int(NODE_2_IP.split(":")[1]))


if __name__ == "__main__":
    if NODE_NUMBER == 0:
        run_gateway()
    elif NODE_NUMBER == 1:
        run_retriever()
    elif NODE_NUMBER == 2:
        run_generator()
    else:
        raise ValueError("unexpected NODE_NUMBER")
