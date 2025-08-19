from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime, timezone
import uuid
import os
import logging
import base64
import json
import requests

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
    import re
    from html import unescape
    text = re.sub(r'&lt;|&gt;|&amp;', lambda m: unescape(m.group(0)), html)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fetch_url_snippet(url: str, max_chars: int = 1200) -> str:
    try:
        r = requests.get(url, timeout=15, headers={'User-Agent': 'FakeFinderBot/1.0'})
        r.raise_for_status()
        text = strip_html_simple(r.text)
        return text[:max_chars]
    except Exception as e:
        logger.warning(f'URL fetch failed: {e}')
        return ''

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
    # Convert timestamp back to datetime if it's string
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

# -------- LLM Analyze --------
@api_router.post('/llm/analyze', response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, current=Depends(get_current_user)):
    if not req.headline and not req.url and not req.image_base64:
        raise HTTPException(status_code=400, detail='Provide at least one of headline, url, image_base64')

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
        'Return STRICT JSON adhering to the provided schema. Keep p_fake in [0,1]. '
        'Ensure rewrites are concise and faithful. Evidence should be up to 3 succinct, source-agnostic snippets.'
    )

    user_content: List[Dict[str, Any]] = []
    prompt_parts = []
    if req.headline:
        prompt_parts.append(f'Headline: {req.headline}')
    if req.url:
        prompt_parts.append(f'URL: {req.url}')
    if url_snippet:
        prompt_parts.append(f'URL_Snippet: {url_snippet[:800]}')

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

    return AnalyzeResponse(
        verdict=data.get('verdict', 'unclear'),
        p_fake=float(data.get('p_fake', 0.5)),
        bias_label=data.get('bias_label', 'unknown'),
        rewrites=RewriteOutput(**data.get('rewrites', {'left': '', 'right': '', 'neutral': ''})),
        evidence=data.get('evidence', []),
        model_used=api_resp.get('model'),
        raw=api_resp
    )

# Include router
app.include_router(api_router)

@app.on_event('shutdown')
async def shutdown_db_client():
    client.close()