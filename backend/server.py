from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime, timezone
import uuid
import os
import logging
import base64
import json
import requests
import re
from urllib.parse import urlparse

# Load env
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# JWT & Auth
from jose import jwt, JWTError
from passlib.context import CryptContext

JWT_SECRET = os.environ.get('JWT_SECRET', 'dev-secret-change-me')
JWT_ALGO = 'HS256'
pwd_context = CryptContext(schemes=['pbkdf2_sha256'], deprecated='auto')
security = HTTPBearer(auto_error=False)

# External LLM config (Emergent LLM key works across providers; using OpenAI REST)
EMERGENT_LLM_KEY = os.environ.get('EMERGENT_LLM_KEY')
OPENAI_API_BASE = 'https://api.openai.com/v1'

# App setup
app = FastAPI(title='FakeFinder API')
api_router = APIRouter(prefix='/api')

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=['*'],
    allow_headers=['*'],
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# -------------------- Models --------------------
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = 'bearer'

class MeResponse(BaseModel):
    id: str
    email: EmailStr

class AnalyzeRequest(BaseModel):
    headline: Optional[str] = None
    url: Optional[str] = None
    image_base64: Optional[str] = None  # data:image/..;base64,....

class RewriteOutput(BaseModel):
    left: str
    right: str
    neutral: str

class AnalyzeResponse(BaseModel):
    verdict: str
    p_fake: float
    bias_label: str
    rewrites: RewriteOutput
    evidence: List[str] = []
    citations: Optional[List[Dict[str, Any]]] = None
    model_used: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

# -------------------- Helpers --------------------
def hash_password(p: str) -> str:
    return pwd_context.hash(p)

def verify_password(p: str, hp: str) -> bool:
    return pwd_context.verify(p, hp)

def create_access_token(user_id: str, email: str) -> str:
    payload = {
        'sub': user_id,
        'email': email,
        'iat': int(datetime.now(timezone.utc).timestamp()),
        'exp': int(datetime.now(timezone.utc).timestamp()) + 60 * 60 * 24
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, Any]:
    if not credentials:
        raise HTTPException(status_code=401, detail='Not authenticated')
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        uid = payload.get('sub')
        if not uid:
            raise HTTPException(status_code=401, detail='Invalid token')
        user = await db.users.find_one({'id': uid})
        if not user:
            raise HTTPException(status_code=401, detail='User not found')
        return {'id': user['id'], 'email': user['email']}
    except JWTError:
        raise HTTPException(status_code=401, detail='Invalid token')

def strip_html_simple(html: str) -> str:
    # naive stripping: remove tags and decode entities
    from html import unescape
    text = re.sub(r'&lt;|&gt;|&amp;', lambda m: unescape(m.group(0)), html)
    text = re.sub(r'<script[\s\S]*?</script>', ' ', text, flags=re.I)
    text = re.sub(r'<style[\s\S]*?</style>', ' ', text, flags=re.I)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Lightweight article extraction ---

def extract_article(url: str, max_chars: int = 12000) -> Tuple[str, List[str]]:
    """Fetch URL and extract title + paragraphs list.
    Avoids heavy dependencies. Uses bs4 if available, otherwise simple regex fallback.
    Returns: (title, paragraphs)
    """
    headers = {'User-Agent': 'FakeFinderBot/1.0 (+https://example.com)'}
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        logger.warning(f'URL fetch failed: {e}')
        return '', []

    title = ''
    paragraphs: List[str] = []

    # Try BeautifulSoup if present
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(html, 'html.parser')
        # Title and headline
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        h1 = soup.find('h1')
        if h1 and h1.get_text(strip=True):
            title = h1.get_text(strip=True)
        # Remove nav/script/style
        for tag in soup(['script', 'style', 'noscript', 'header', 'footer', 'aside']):
            tag.decompose()
        # Collect paragraphs
        for p in soup.find_all('p'):
            t = p.get_text(" ", strip=True)
            if t and len(t) > 40:
                paragraphs.append(t)
        if not paragraphs:
            # fallback to text body
            body_text = soup.get_text(" ", strip=True)
            paragraphs = [body_text[i:i+600] for i in range(0, min(len(body_text), max_chars), 600)]
    except Exception:
        # Regex fallback
        m = re.search(r'<title>(.*?)</title>', html, flags=re.I|re.S)
        if m:
            title = strip_html_simple(m.group(1))
        # get paragraphs
        ps = re.findall(r'<p[^>]*>(.*?)</p>', html, flags=re.I|re.S)
        for raw in ps:
            t = strip_html_simple(raw)
            if len(t) > 40:
                paragraphs.append(t)
        if not paragraphs:
            text = strip_html_simple(html)
            paragraphs = [text[i:i+600] for i in range(0, min(len(text), max_chars), 600)]

    # Trim & limit
    paragraphs = [p.strip()[:600] for p in paragraphs][:60]
    title = (title or '').strip()[:280]
    return title, paragraphs


def score_paragraphs(query: str, paragraphs: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    """Very simple relevance score: token overlap + position bias."""
    if not paragraphs:
        return []
    q_tokens = set(re.findall(r'[a-zA-Z0-9]+', (query or '').lower()))
    scored: List[Tuple[str, float]] = []
    for idx, p in enumerate(paragraphs):
        p_tokens = re.findall(r'[a-zA-Z0-9]+', p.lower())
        if not p_tokens:
            continue
        overlap = sum(1 for t in p_tokens if t in q_tokens)
        dens = overlap / max(6, len(p_tokens))
        pos_bonus = 0.2 if idx < 5 else 0.0
        score = dens + pos_bonus
        scored.append((p, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def fetch_url_snippet(url: str, max_chars: int = 1200) -> str:
    title, paragraphs = extract_article(url, max_chars=max_chars)
    joined = ' '.join(paragraphs)
    return (title + ' ' + joined)[:max_chars]


def validate_image_data_url(image_data_url: str) -> bool:
    try:
        if not image_data_url.startswith('data:image/'):
            return False
        header, b64 = image_data_url.split(',', 1)
        base64.b64decode(b64)
        return True
    except Exception:
        return False

async def call_gpt4o(messages: List[Dict[str, Any]], use_json: bool = True, model: str = 'gpt-4o') -> Dict[str, Any]:
    if not EMERGENT_LLM_KEY:
        raise HTTPException(status_code=500, detail='LLM key not configured')
    headers = {
        'Authorization': f'Bearer {EMERGENT_LLM_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': model,
        'messages': messages,
        'temperature': 0.2,
        'max_tokens': 1200,
    }
    if use_json:
        payload['response_format'] = {'type': 'json_object'}
    try:
        resp = requests.post(f'{OPENAI_API_BASE}/chat/completions', headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error(f'OpenAI error: {e}')
        raise HTTPException(status_code=503, detail='LLM service temporarily unavailable')

# -------------------- Routes --------------------
@api_router.get('/')
async def root():
    return {'message': 'Hello World'}

@api_router.post('/status', response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_obj = StatusCheck(client_name=input.client_name)
    await db.status_checks.insert_one({
        'id': status_obj.id,
        'client_name': status_obj.client_name,
        'timestamp': status_obj.timestamp.isoformat()
    })
    return status_obj

@api_router.get('/status', response_model=List[StatusCheck])
async def get_status_checks():
    items = await db.status_checks.find().to_list(1000)
    out: List[StatusCheck] = []
    for it in items:
        ts = it.get('timestamp')
        if isinstance(ts, str):
            try:
                it['timestamp'] = datetime.fromisoformat(ts)
            except Exception:
                it['timestamp'] = datetime.now(timezone.utc)
        out.append(StatusCheck(**it))
    return out

# -------- Auth --------
@api_router.post('/auth/signup', response_model=TokenResponse)
async def signup(payload: UserCreate):
    existing = await db.users.find_one({'email': payload.email})
    if existing:
        raise HTTPException(status_code=400, detail='Email already registered')
    uid = str(uuid.uuid4())
    user_doc = {
        'id': uid,
        'email': payload.email,
        'password_hash': hash_password(payload.password),
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    await db.users.insert_one(user_doc)
    token = create_access_token(uid, payload.email)
    return TokenResponse(access_token=token)

@api_router.post('/auth/login', response_model=TokenResponse)
async def login(payload: UserLogin):
    user = await db.users.find_one({'email': payload.email})
    if not user or not verify_password(payload.password, user.get('password_hash', '')):
        raise HTTPException(status_code=401, detail='Invalid email or password')
    token = create_access_token(user['id'], user['email'])
    return TokenResponse(access_token=token)

@api_router.get('/auth/me', response_model=MeResponse)
async def me(current=Depends(get_current_user)):
    return MeResponse(id=current['id'], email=current['email'])

# -------- LLM Analyze with scraping-based evidence --------
@api_router.post('/llm/analyze', response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, current=Depends(get_current_user)):
    if not req.headline and not req.url and not req.image_base64:
        raise HTTPException(status_code=400, detail='Provide at least one of headline, url, image_base64')

    # Build evidence from URL if present
    evidence_snippets: List[str] = []
    citations: List[Dict[str, Any]] = []
    if req.url:
        art_title, paragraphs = extract_article(req.url)
        query = req.headline or art_title or ''
        top = score_paragraphs(query, paragraphs, top_k=3)
        domain = urlparse(req.url).netloc if req.url else ''
        for i, (snip, sc) in enumerate(top, start=1):
            evidence_snippets.append(f"[S{i}] ({domain}) {snip}")
            citations.append({'id': f'S{i}', 'url': req.url, 'domain': domain, 'snippet': snip})

    # URL snippet for extra context (short)
    url_snippet = ''
    if req.url:
        url_snippet = fetch_url_snippet(req.url)

    if req.image_base64 and not validate_image_data_url(req.image_base64):
        raise HTTPException(status_code=400, detail='Invalid image_base64 data URL format')

    schema = {
        'type': 'object',
        'properties': {
            'verdict': {'type': 'string', 'enum': ['verified', 'fake', 'unclear']},
            'p_fake': {'type': 'number'},
            'bias_label': {'type': 'string', 'enum': ['left', 'right', 'neutral', 'unknown']},
            'rewrites': {
                'type': 'object',
                'properties': {
                    'left': {'type': 'string'},
                    'right': {'type': 'string'},
                    'neutral': {'type': 'string'}
                },
                'required': ['left', 'right', 'neutral']
            },
            'evidence': {'type': 'array', 'items': {'type': 'string'}}
        },
        'required': ['verdict', 'p_fake', 'bias_label', 'rewrites', 'evidence']
    }

    system = (
        'You are FakeFinder, an AI that verifies news headlines, detects bias, and rewrites headlines from alternate perspectives. '
        'Use any provided URL text snippet and image (OCR the image content if present). '
        'You MUST cite supporting snippets using [S1], [S2], [S3] where applicable, matching the given context snippets. '
        'Return STRICT JSON adhering to the provided schema. Keep p_fake in [0,1]. '
        'Ensure rewrites are concise and faithful.'
    )

    user_content: List[Dict[str, Any]] = []
    prompt_parts = []
    if req.headline:
        prompt_parts.append(f'Headline: {req.headline}')
    if req.url:
        prompt_parts.append(f'URL: {req.url}')
    if url_snippet:
        prompt_parts.append(f'URL_Snippet: {url_snippet[:800]}')
    if evidence_snippets:
        prompt_parts.append('Context snippets: ' + ' \n'.join(evidence_snippets))

    if prompt_parts:
        user_content.append({'type': 'text', 'text': '\n'.join(prompt_parts)})

    if req.image_base64:
        user_content.append({'type': 'image_url', 'image_url': {'url': req.image_base64}})

    messages = [
        {'role': 'system', 'content': f"{system} JSON Schema: {json.dumps(schema)}"},
        {'role': 'user', 'content': user_content}
    ]

    api_resp = await call_gpt4o(messages, use_json=True, model='gpt-4o')
    content = api_resp.get('choices', [{}])[0].get('message', {}).get('content', '{}')

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        logger.warning('Model returned non-JSON; wrapping raw string.')
        data = {
            'verdict': 'unclear',
            'p_fake': 0.5,
            'bias_label': 'unknown',
            'rewrites': {'left': req.headline or '', 'right': req.headline or '', 'neutral': req.headline or ''},
            'evidence': [content[:280]]
        }

    # If we built evidence, ensure response includes at least them
    if evidence_snippets and not data.get('evidence'):
        data['evidence'] = evidence_snippets

    return AnalyzeResponse(
        verdict=data.get('verdict', 'unclear'),
        p_fake=float(data.get('p_fake', 0.5)),
        bias_label=data.get('bias_label', 'unknown'),
        rewrites=RewriteOutput(**data.get('rewrites', {'left': '', 'right': '', 'neutral': ''})),
        evidence=data.get('evidence', evidence_snippets),
        citations=citations or None,
        model_used=api_resp.get('model'),
        raw=api_resp
    )

# Include router
app.include_router(api_router)

@app.on_event('shutdown')
async def shutdown_db_client():
    client.close()