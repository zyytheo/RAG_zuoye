# #  ==== 流式响应的服务端版本 ======
# #  知识库加载优化：改为增量加载，不用每次重新构建和加载，定义两个参数用来记录已经向量化的数据和文件。

# -----------------
import datetime
import os
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import aiohttp
import numpy as np
import sqlalchemy
import torch
import pickle
import functools
import requests
import json
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from FlagEmbedding import FlagReranker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
from requests import sessions, Request, Response
from starlette.responses import StreamingResponse
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.schema.runnable import RunnableSequence
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

from config.config import CHUNK_SIZE, CHUNK_OVERLAP, KNOWLEDGE_BASE_DIR, OLLAMA_MODEL_NAME, EMBEDDING_MODEL_NAME, \
    RERANKER_MODEL_NAME, VECTOR_INDEX_DIR, FILE_STATE_FILE
import logging
from concurrent.futures import ThreadPoolExecutor
import asyncio
from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, ForeignKey, text, event
from sqlalchemy.orm import sessionmaker, declarative_base, Session

from model.entities import Message, SessionModel, Base, SessionModelDB, MessageDB, KbFile, KbFileModel
# 开启新的会话
import uuid

from utils.company_parser import get_company_name_from_md, extract_company_name

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 数据库配置（切换为本地 SQLite）
SQLALCHEMY_DATABASE_URL = "sqlite:///./llmrag.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False}  # 允许多线程访问（FastAPI 中常见）
)

# 开启 SQLite 外键约束
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    except Exception as e:
        logger.warning(f"Failed to enable SQLite foreign_keys pragma: {e}")
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 创建表
Base.metadata.create_all(bind=engine)

# 初始化 FastAPI 应用
app = FastAPI()

# 配置 CORS 中间件
origins = [
    "http://localhost:8080",  # 根据实际情况修改
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def collect_file_states(directory: str) -> dict:
    """收集目录下所有文件的最后修改时间，用于判断是否需要重建向量索引。"""
    file_states = {}
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            try:
                file_states[path] = os.path.getmtime(path)
            except OSError as exc:
                logger.warning(f"读取文件时间失败 {path}: {exc}")
    return file_states


def nz(value):
    """Return a trimmed string or None if the value is empty, NULL-like or missing."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() == "NULL":
        return None
    return text


def build_sample_summary(record: dict) -> str:
    """Transform a JSON record into an English, human-readable summary."""
    sample = record.get("sampleDTO") or {}
    location = record.get("locationDTO") or {}
    material = record.get("materialDTO") or {}
    literature = record.get("LiteratureDTO") or []

    lines = []
    sample_parts = []
    sample_name = nz(sample.get("name"))
    if sample_name:
        sample_parts.append(sample_name)
    sample_id = nz(sample.get("id"))
    if sample_id:
        sample_parts.append(f"ID {sample_id}")
    sample_kind = nz(sample.get("sampleKindName"))
    if sample_kind:
        sample_parts.append(f"Type {sample_kind}")
    material_name = nz(sample.get("materialName"))
    material_kind = nz(material.get("materialKind"))
    material_label = "/".join([p for p in [material_name, material_kind] if p])
    if material_label:
        sample_parts.append(f"Material {material_label}")
    sample_method = nz(sample.get("sampleMethodName"))
    if sample_method:
        sample_parts.append(f"Method {sample_method}")
    if sample_parts:
        lines.append("Sample: " + "; ".join(sample_parts))

    location_parts = []
    lat = nz(location.get("lat"))
    lon = nz(location.get("lon"))
    if lat:
        location_parts.append(f"lat {lat}")
    if lon:
        location_parts.append(f"lon {lon}")
    precision = nz(location.get("latLonPrecision"))
    if precision:
        location_parts.append(f"precision ≈ {precision} m")
    elevation = nz(sample.get("referenceElevation"))
    if elevation:
        location_parts.append(f"reference elevation {elevation} m")
    data_package = nz(sample.get("dataPackageName"))
    if data_package:
        location_parts.append(f"data package {data_package}")
    archive = nz(sample.get("archiveName"))
    if archive:
        location_parts.append(f"archive {archive}")
    if location_parts:
        lines.append("Location: " + ", ".join(location_parts))

    time_parts = []
    collect_min = nz(sample.get("collectDateMin"))
    collect_max = nz(sample.get("collectDateMax"))
    if collect_min or collect_max:
        if collect_min and collect_max and collect_min != collect_max:
            time_parts.append(f"Collected {collect_min} to {collect_max}")
        else:
            time_parts.append(f"Collected {collect_min or collect_max}")
    created = nz(sample.get("createdTimestamp"))
    if created:
        time_parts.append(f"Created {created}")
    updated = nz(sample.get("lastEditedTimestamp"))
    if updated:
        time_parts.append(f"Updated {updated}")
    if time_parts:
        lines.append("Timeline: " + "; ".join(time_parts))

    igsn_parts = [nz(sample.get("igsn")), nz(sample.get("igsnHandleURL"))]
    igsn_parts = [part for part in igsn_parts if part]
    if igsn_parts:
        lines.append("IGSN: " + ", ".join(igsn_parts))

    description = nz(sample.get("description"))
    if description:
        lines.append("Sample Notes: " + description)
    material_desc = nz(material.get("description"))
    if material_desc:
        lines.append("Material Notes: " + material_desc)

    literature_titles = []
    for item in literature:
        detail = item.get("literatureDetail") or {}
        title = nz(detail.get("calcName")) or nz(detail.get("title"))
        if title:
            literature_titles.append(title)
    if literature_titles:
        lines.append("References: " + "; ".join(literature_titles))

    return "\n".join([line for line in lines if line])


def load_json_as_documents(file_path: str) -> List[Document]:
    """Convert JSON records into structured LangChain Documents for higher-quality retrieval."""
    try:
        payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error(f"Failed to load JSON file {file_path}: {exc}")
        return []

    if not isinstance(payload, list):
        payload = [payload]

    documents: List[Document] = []
    for idx, record in enumerate(payload):
        sample = record.get("sampleDTO") or {}
        location = record.get("locationDTO") or {}
        material = record.get("materialDTO") or {}

        summary = build_sample_summary(record)
        if not summary:
            continue

        literature_titles = []
        for item in record.get("LiteratureDTO") or []:
            detail = item.get("literatureDetail") or {}
            title = nz(detail.get("calcName")) or nz(detail.get("title"))
            if title:
                literature_titles.append(title)

        metadata = {
            "source": file_path,
            "record_index": idx,
            "sample_id": sample.get("id"),
            "sample_name": sample.get("name"),
            "sample_kind": sample.get("sampleKindName"),
            "material_id": sample.get("materialId"),
            "material_name": sample.get("materialName"),
            "material_kind": material.get("materialKind"),
            "lat": location.get("lat"),
            "lon": location.get("lon"),
            "data_package": sample.get("dataPackageName"),
            "igsn": sample.get("igsn"),
            "archive": sample.get("archiveName"),
            "literature_titles": literature_titles,
        }
        documents.append(Document(page_content=summary, metadata=metadata))
    return documents


# 向量与精排模型缓存
@functools.lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_kwargs = {"device": device}
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True}
    )


@functools.lru_cache(maxsize=1)
def get_reranker_model() -> FlagReranker:
    return FlagReranker(
        RERANKER_MODEL_NAME,
        use_fp16=torch.cuda.is_available()
    )


# 全局检索状态
vector_store: Optional[FAISS] = None
documents_store: List[Document] = []
doc_index_lookup: Dict[int, Document] = {}
bm25_index: Optional[BM25Okapi] = None
bm25_corpus: List[List[str]] = []
sample_id_to_chunks = defaultdict(list)

BM25_WEIGHT = 0.6
VECTOR_WEIGHT = 0.4
RRF_CONSTANT = 60.0
BM25_TOP_K = 200
VECTOR_TOP_K = 200
FUSION_TOP_K = 100
RERANK_TOP_K = 3
RERANK_BATCH_SIZE = 32


def tokenize_en(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


def extract_sample_id(query: str) -> Optional[str]:
    if not query:
        return None
    patterns = [
        r"\b(?:id|sample[_\s-]?id)\s*[:=]?\s*(\d{5,})\b",
        r"id是\s*(\d{5,})"
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def build_vector_index(directory: str) -> Tuple[FAISS, List[Document]]:
    embeddings = get_embeddings()
    raw_documents: List[Document] = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                raw_documents.extend(loader.load())
            elif file.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
                raw_documents.extend(loader.load())
            elif file.endswith('.md'):
                loader = TextLoader(file_path, encoding='utf-8')
                doc = loader.load()[0]
                company_name = get_company_name_from_md(file_path)
                doc.metadata = {"company_name": company_name}
                raw_documents.append(doc)
            elif file.endswith('.json'):
                raw_documents.extend(load_json_as_documents(file_path))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
    )
    split_documents = splitter.split_documents(raw_documents)
    for idx, doc in enumerate(split_documents):
        doc.metadata = dict(doc.metadata or {})
        doc.metadata["chunk_index"] = idx
        sample_id = doc.metadata.get("sample_id")
        if sample_id is not None:
            doc.metadata["sample_id"] = str(sample_id)

    vector_db = FAISS.from_documents(
        split_documents,
        embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT
    )
    return vector_db, split_documents


def initialize_retrieval_structures(documents: List[Document]) -> None:
    global documents_store, doc_index_lookup, bm25_index, bm25_corpus, sample_id_to_chunks

    documents_store = documents
    doc_index_lookup = {}
    sample_map: defaultdict = defaultdict(list)
    tokenized_corpus: List[List[str]] = []

    for idx, doc in enumerate(documents_store):
        doc.metadata = dict(doc.metadata or {})
        doc.metadata.setdefault("chunk_index", idx)
        doc_index_lookup[idx] = doc
        sample_id = doc.metadata.get("sample_id")
        if sample_id is not None:
            sample_map[str(sample_id)].append(doc)
        tokenized_corpus.append(tokenize_en(doc.page_content))

    bm25_corpus = tokenized_corpus
    bm25_index = BM25Okapi(bm25_corpus) if bm25_corpus else None
    sample_id_to_chunks = sample_map


def rrf_merge(query: str) -> List[Document]:
    if not documents_store:
        return []

    candidates: set[int] = set()
    bm25_ranks: Dict[int, int] = {}
    vec_ranks: Dict[int, int] = {}

    if bm25_index is not None:
        tokens = tokenize_en(query)
        scores = np.array(bm25_index.get_scores(tokens))
        topk = min(BM25_TOP_K, len(documents_store))
        top_indices = np.argsort(-scores)[:topk]
        bm25_ranks = {int(idx): rank for rank, idx in enumerate(top_indices, start=1)}
        candidates.update(bm25_ranks.keys())

    if vector_store is not None:
        top_vec = min(VECTOR_TOP_K, len(documents_store))
        results = vector_store.similarity_search_with_score(query, k=top_vec)
        for rank, (doc, _) in enumerate(results, start=1):
            chunk_idx = doc.metadata.get("chunk_index")
            if chunk_idx is None:
                continue
            chunk_idx = int(chunk_idx)
            vec_ranks[chunk_idx] = rank
            candidates.add(chunk_idx)

    if not candidates:
        return []

    def fused_score(idx: int) -> float:
        score = 0.0
        rank_bm25 = bm25_ranks.get(idx)
        rank_vec = vec_ranks.get(idx)
        if rank_bm25 is not None:
            score += BM25_WEIGHT / (RRF_CONSTANT + rank_bm25)
        if rank_vec is not None:
            score += VECTOR_WEIGHT / (RRF_CONSTANT + rank_vec)
        return score

    ranked = sorted(candidates, key=lambda i: fused_score(i), reverse=True)
    top_indices = ranked[:min(FUSION_TOP_K, len(ranked))]
    return [doc_index_lookup[i] for i in top_indices if i in doc_index_lookup]


def rerank_documents(query: str, candidates: List[Document], topn: int = RERANK_TOP_K) -> List[Document]:
    if not candidates:
        return []
    reranker = get_reranker_model()
    pairs = [(query, doc.page_content) for doc in candidates]
    batch = max(1, min(RERANK_BATCH_SIZE, len(pairs)))
    scores = reranker.compute_score(pairs, batch_size=batch)
    ranked_docs = [doc for _, doc in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
    return ranked_docs[:min(topn, len(ranked_docs))]


def retrieve_documents(query: str) -> List[Document]:
    sid = extract_sample_id(query)
    if sid and sid in sample_id_to_chunks:
        logger.info(f"Sample ID hit detected: {sid}")
        return rerank_documents(query, sample_id_to_chunks[sid])

    candidates = rrf_merge(query)
    if not candidates:
        return []
    return rerank_documents(query, candidates)


def load_vector_store_from_disk(embeddings: HuggingFaceEmbeddings) -> Optional[FAISS]:
    if not os.path.isdir(VECTOR_INDEX_DIR):
        return None
    try:
        return FAISS.load_local(
            VECTOR_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as exc:
        logger.error(f"Failed to load FAISS index from disk: {exc}")
        return None


async def load_or_build_vector_index(directory: str):
    global vector_store

    embeddings = get_embeddings()
    current_state = collect_file_states(directory)

    cached_state = {}
    if os.path.exists(FILE_STATE_FILE):
        try:
            with open(FILE_STATE_FILE, 'rb') as f:
                cached_state = pickle.load(f) or {}
        except Exception as exc:
            logger.warning(f"Failed to read cached file states: {exc}")
            cached_state = {}

    if isinstance(cached_state, list):
        logger.info("Detected legacy file state format; forcing index rebuild.")
        cached_state = {}

    needs_rebuild = (not os.path.isdir(VECTOR_INDEX_DIR)) or (cached_state != current_state)

    if needs_rebuild:
        logger.info("Building FAISS index from knowledge base...")
        vector_store, documents = build_vector_index(directory)
        if os.path.isdir(VECTOR_INDEX_DIR):
            shutil.rmtree(VECTOR_INDEX_DIR)
        os.makedirs(VECTOR_INDEX_DIR, exist_ok=True)
        vector_store.save_local(VECTOR_INDEX_DIR)
        with open(FILE_STATE_FILE, 'wb') as f:
            pickle.dump(current_state, f)
        initialize_retrieval_structures(documents)
    else:
        logger.info("Loading FAISS index from cache...")
        vector_store = load_vector_store_from_disk(embeddings)
        if vector_store is None:
            logger.warning("Cached index unavailable; rebuilding.")
            vector_store, documents = build_vector_index(directory)
            if os.path.isdir(VECTOR_INDEX_DIR):
                shutil.rmtree(VECTOR_INDEX_DIR)
            os.makedirs(VECTOR_INDEX_DIR, exist_ok=True)
            vector_store.save_local(VECTOR_INDEX_DIR)
            with open(FILE_STATE_FILE, 'wb') as f:
                pickle.dump(current_state, f)
            initialize_retrieval_structures(documents)
        else:
            docs = list(vector_store.docstore._dict.values())
            initialize_retrieval_structures(docs)

    return vector_store

# 异步处理 llm.stream 结果的函数
async def process_ollama_stream(chunk):
    if 'response' in chunk:
        return chunk['response'].encode('utf-8')
    else:
        logger.error(f"Unexpected chunk format: {chunk}")
        return None

# 定义新的提示模板（包含历史对话）
prompt_template = """历史对话：
{history}

使用以下文档中的信息来回答问题：
{context}

问题: {question}"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["history", "context", "question"]
)

# 异步生成回答
# 异步生成回答（修改后的函数）
async def generate_answer(question, session_id, db: Session):
    global vector_store
    if vector_store is None or not documents_store:
        vector_store = await load_or_build_vector_index(KNOWLEDGE_BASE_DIR)
    try:
        logger.info(f"Received question: {question}")

        # 获取历史对话（排除当前尚未保存的问题）
        history_messages = db.query(MessageDB).filter(
            MessageDB.session_id == session_id,
            MessageDB.type.in_(['question', 'answer'])
        ).order_by(MessageDB.created_date).all()

        # 构造历史对话字符串
        history_str = ""
        for msg in history_messages:
            if msg.type == 'question':
                history_str += f"用户: {msg.content}\n"
            elif msg.type == 'answer':
                history_str += f"助手: {msg.content}\n"

        company_name = extract_company_name(question)
        logger.info(f"Company name extracted from query: {company_name}")

        final_results = retrieve_documents(question)

        if final_results:
            logger.info(f"Retrieved {len(final_results)} documents for response.")
            context = "\n\n".join(doc.page_content for doc in final_results)
        else:
            logger.info("No relevant documents found in the knowledge base.")
            context = ""

        # 构造完整提示（包含历史对话和上下文）
        full_prompt = prompt.format(
            history=history_str.strip(),
            context=context,
            question=question
        )

        # 流式请求Ollama（保持原有逻辑）
        url = 'http://localhost:11434/api/generate'
        headers = {'Content-Type': 'application/json'}
        data = {
            "model": OLLAMA_MODEL_NAME,
            "prompt": full_prompt,
            "parameters": {
                "max_tokens": 100,
                "temperature": 0.2
            },
            "stream": True
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as res:
                answer = ""
                async for line in res.content:
                    if line:
                        try:
                            chunk = json.loads(line)
                            result = await process_ollama_stream(chunk)
                            if result:
                                answer_chunk = result.decode('utf-8')
                                answer += answer_chunk
                                # 流式返回数据
                                yield answer_chunk.encode('utf-8')
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decoding error: {e}, line: {line}")

        # 保存问题和回答到数据库
        question_message = MessageDB(content=question, type='question', session_id=session_id)
        answer_message = MessageDB(content=answer, type='answer', session_id=session_id)
        db.add(question_message)
        db.add(answer_message)
        db.commit()

    except Exception as general_exception:
        error_message = f"General error occurred: {str(general_exception)}"
        logger.error(error_message)
        error_message_obj = MessageDB(
            content=error_message,
            type='error',
            session_id=session_id
        )
        db.add(error_message_obj)
        db.commit()
        yield error_message.encode('utf-8')
        return  # 确保生成器停止


# 获取数据库会话
def get_db():
    max_retries = 3
    retries = 0
    db = None
    while retries < max_retries:
        try:
            db = SessionLocal()
            yield db
            break
        except sqlalchemy.exc.OperationalError as e:
            retries += 1
            logger.error(f"Database connection error (attempt {retries}/{max_retries}): {e}")
            if retries < max_retries:
                time.sleep(2)  # 等待 2 秒后重试
            else:
                raise
        finally:
            if db:
                db.close()


def parse_think_string(input_string):
    try:
        # print("input_string=" +input_string)
        # 使用正则表达式查找 <think> 标签内的内容，re.DOTALL 让 . 匹配包括换行符的任意字符
        think_match = re.search(r'<think>(.*?)</think>', input_string, re.DOTALL)
        # 如果找到匹配项，提取标签内的内容并去除首尾空格，否则赋值为空字符串
        x = think_match.group(1).strip() if think_match else ""
        # print(f"匹配到的 <think> 标签内的内容: {x}")  # 调试输出

        # 移除 <think> 标签及其内容
        pattern = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
        y = pattern.sub('', input_string).strip()
        # print(f"替换后的内容: {y}")  # 调试输出

        return x, y
    except AttributeError:
        print("未找到 <think> 标签或在解析过程中出现错误。")
        return "", ""


# 定义中间件函数
@app.middleware("http")
async def set_utf8_encoding(request, call_next):
    # 调用下一个中间件或路由处理函数
    response = await call_next(request)
    # 设置响应头
    response.headers['Content-Type'] = 'application/json; charset=utf-8'
    return response


# 对话接口
@app.get("/ask")
async def ask_question(question: str, session_id: int = None, db: Session = Depends(get_db)):
    global vector_store
    if vector_store is None or not documents_store:
        vector_store = await load_or_build_vector_index(KNOWLEDGE_BASE_DIR)
    try:
        if session_id is None:
            new_session = SessionModelDB(session_id=str(uuid.uuid4()))  # 这里可以使用更合适的会话 ID 生成方式
            db.add(new_session)
            db.commit()
            db.refresh(new_session)
            session_id = new_session.id

        headers = {
            'Content-Type': 'text/plain; charset=utf-8',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Transfer-Encoding': 'chunked'
        }
        return StreamingResponse(generate_answer(question, session_id, db), media_type="text/plain", headers=headers)
    except Exception as e:
        logger.error(f"Error in ask_question: {e}")
        return {"error": "An error occurred while processing the request."}
    finally:
        db.close()


@app.post("/new_session")
async def start_new_session(db: Session = Depends(get_db)):
    new_session_id = str(uuid.uuid4())
    new_session = SessionModelDB(session_id=new_session_id)
    db.add(new_session)
    db.commit()
    db.refresh(new_session)
    return {"id": new_session.id}


# 删除会话接口
@app.delete("/delete_session/{session_id}")
async def delete_session(session_id: int, db: Session = Depends(get_db)):
    # 查询要删除的会话
    session_to_delete = db.query(SessionModelDB).filter(SessionModelDB.id == session_id).first()

    # 如果会话不存在，返回 404 错误
    if not session_to_delete:
        raise HTTPException(status_code=404, detail="会话不存在")

    # 删除会话
    db.delete(session_to_delete)
    db.commit()

    return {"message": "会话删除成功"}


# 获取会话历史接口
@app.get("/sessions", response_model=List[SessionModel])
async def get_sessions(db: Session = Depends(get_db)):
    # 查询数据库中的会话历史
    sessions = db.query(SessionModelDB).all()
    # 按创建时间降序排序
    sorted_sessions = sorted(sessions, key=lambda x: x.created_date, reverse=True)
    return sorted_sessions


@app.get("/session/{session_id}/messages", response_model=List[Message])
async def get_session_messages(session_id: int, db: Session = Depends(get_db)):
    messages_db = db.query(MessageDB).filter(MessageDB.session_id == session_id).all()
    messages = []
    for message_db in messages_db:
        content = message_db.content
        # print(f"原始内容: {content}")  # 打印原始内容用于调试

        think, final_content = parse_think_string(content)

        # print(f"解析后的 think: {think}")
        # print(f"解析后的 final_content: {final_content}")
        message = Message(
            id=message_db.id,
            think=think,
            final_content=final_content,
            type=message_db.type,
            created_date=message_db.created_date,
            session_id=message_db.session_id
        )
        messages.append(message)
    return messages


# 新增接口：列出所有知识库文档
@app.get("/list_documents")
async def list_documents():
    documents = []
    for root, dirs, files in os.walk(KNOWLEDGE_BASE_DIR):
        for file in files:
            if file.endswith('.md'):  # 只获取 md 文件
                file_path = os.path.join(root, file)
                company_name  = get_company_name_from_md(file_path)
                if company_name  is not None:
                    documents.append(company_name )
    return {"documents": documents}


# 新增接口：获取文档内容
@app.get("/get_document_content")
async def get_document_content(file_path: str = Query(..., description="要获取内容的文档名称"),
                               db: Session = Depends(get_db)):
    print("预览的文件===" + file_path)
    # 根据文件名从数据库中查询对应的路径
    kb_file = db.query(KbFile).filter(KbFile.name == file_path).first()
    if not kb_file:
        raise HTTPException(status_code=404, detail="文档不存在")

    full_path = kb_file.path
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="文档不存在")
    try:
        with open(full_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取文档内容出错: {str(e)}")


# 新增接口：删除指定的知识库文档
@app.delete("/delete_document")
async def delete_document(file_path: str = Query(..., description="要删除的文档路径"),
                          db: Session = Depends(get_db)):
    kb_file = db.query(KbFile).filter(KbFile.name == file_path).first()
    if not kb_file:
        raise HTTPException(status_code=404, detail="文档不存在")

    full_path = kb_file.path

    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="文档不存在")
    os.remove(full_path)

    # 重新构建向量索引
    logger.info("Document deleted. Rebuilding vector index...")
    global vector_store
    vector_store = await load_or_build_vector_index(KNOWLEDGE_BASE_DIR)

    return {"message": f"文档 {file_path} 已成功删除"}


# 初始化知识库文件
# 修改 kb_init 函数，确保在更新数据库后重建向量索引
@app.get("/kb_init")
async def kb_init():
    db = SessionLocal()
    inserted_kb_files = []
    try:
        # 清空数据库里的所有 KbFile 记录 - 使用更明确的删除方式
        try:
            count = db.query(KbFile).delete(synchronize_session='fetch')
            db.commit()
            logger.info(f"已删除 {count} 条知识库文件记录")

            # 验证删除是否成功
            remaining = db.query(KbFile).count()
            logger.info(f"剩余知识库文件记录数: {remaining}")

            if remaining > 0:
                logger.warning("数据库记录未完全清空，尝试使用原生SQL")
                db.execute(text("DELETE FROM kb_file"))
                db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"删除知识库文件记录失败: {e}")
            return {"message": f"知识库文件初始化失败: {e}"}

        # 扫描并添加知识库文件到数据库
        for root, dirs, files in os.walk(KNOWLEDGE_BASE_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()

                if file_ext == '.md':
                    name = get_company_name_from_md(file_path)
                    if not name:
                        name = os.path.splitext(file)[0]  # 如果获取第一行失败，使用文件名
                else:
                    name = os.path.splitext(file)[0]

                new_kb_file = KbFile(name=name, path=file_path)
                db.add(new_kb_file)
                inserted_kb_files.append(new_kb_file)

        db.commit()

        # 重要：强制重建向量索引
        # 删除现有的向量索引和文件列表，以便它们被重新创建
        if os.path.isdir(VECTOR_INDEX_DIR):
            shutil.rmtree(VECTOR_INDEX_DIR)
        if os.path.exists(FILE_STATE_FILE):
            os.remove(FILE_STATE_FILE)

        # 重新构建向量索引
        logger.info("知识库文件更新，正在重建向量索引...")
        global vector_store
        vector_store = await load_or_build_vector_index(KNOWLEDGE_BASE_DIR)
        logger.info("向量索引重建完成")

        # 将 SQLAlchemy 对象转换为 Pydantic 模型对象
        pydantic_kb_files = [KbFileModel.model_validate(kb_file) for kb_file in inserted_kb_files]

        return {
            "message": "知识库文件初始化成功",
            "inserted_kb_files": pydantic_kb_files
        }
    except Exception as e:
        db.rollback()
        logger.error(f"知识库文件初始化失败: {e}")
        return {"message": f"知识库文件初始化失败: {e}"}
    finally:
        db.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
