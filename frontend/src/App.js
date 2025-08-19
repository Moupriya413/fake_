import React, { useEffect, useMemo, useRef, useState } from 'react';
import './App.css';
import axios from 'axios';
import { BrowserRouter } from 'react-router-dom';
// shadcn ui components
import { Button } from './components/ui/button';
import { Input } from './components/ui/input';
import { Textarea } from './components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './components/ui/tabs';
import { Label } from './components/ui/label';
import { Progress } from './components/ui/progress';
import { Badge } from './components/ui/badge';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from './components/ui/dialog';
import { useToast } from './hooks/use-toast';
import { ShieldCheck, AlertTriangle, Globe, Image as ImageIcon, Upload, LogIn, UserPlus } from 'lucide-react';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL; // Do not hardcode
const API = `${BACKEND_URL}/api`;

const api = axios.create({ baseURL: API });

function useAuth() {
  const [token, setToken] = useState(() => localStorage.getItem('ff_token') || '');
  useEffect(() => {
    if (token) localStorage.setItem('ff_token', token);
  }, [token]);

  useEffect(() => {
    const id = api.interceptors.request.use((config) => {
      if (token) config.headers.Authorization = `Bearer ${token}`;
      return config;
    });
    return () => api.interceptors.request.eject(id);
  }, [token]);

  return { token, setToken };
}

function Header({ onOpenAuth }) {
  return (
    <header className="ff-header">
      <div className="ff-container">
        <div className="brand">
          <ShieldCheck size={24} />
          <span>FakeFinder</span>
        </div>
        <div className="actions">
          <Button variant="outline" onClick={onOpenAuth}>
            <LogIn className="mr-2" size={16} /> Login / Sign up
          </Button>
        </div>
      </div>
    </header>
  );
}

function AuthDialog({ open, onOpenChange, onAuthed }) {
  const { toast } = useToast();
  const [mode, setMode] = useState('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);

  const submit = async () => {
    try {
      setLoading(true);
      const endpoint = mode === 'login' ? '/auth/login' : '/auth/signup';
      const { data } = await api.post(endpoint, { email, password });
      onAuthed(data.access_token);
      toast({ title: 'Success', description: `You are ${mode}ed in.` });
      onOpenChange(false);
    } catch (e) {
      toast({ title: 'Auth failed', description: e?.response?.data?.detail || 'Please try again', variant: 'destructive' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="auth-dialog">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {mode === 'login' ? <LogIn size={18} /> : <UserPlus size={18} />} {mode === 'login' ? 'Login' : 'Create an account'}
          </DialogTitle>
        </DialogHeader>
        <div className="grid gap-4 mt-2">
          <div className="grid gap-2">
            <Label htmlFor="email">Email</Label>
            <Input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@news.com" />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="password">Password</Label>
            <Input id="password" type="password" value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••••" />
          </div>
          <div className="flex items-center justify-between">
            <Button disabled={loading} onClick={submit} className="w-full">
              {loading ? 'Please wait…' : mode === 'login' ? 'Login' : 'Sign up'}
            </Button>
          </div>
          <div className="text-sm text-muted mt-1 text-center">
            {mode === 'login' ? (
              <span>New here? <button className="link" onClick={() => setMode('signup')}>Create an account</button></span>
            ) : (
              <span>Already have an account? <button className="link" onClick={() => setMode('login')}>Login</button></span>
            )}
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function Analyzer({ token }) {
  const { toast } = useToast();
  const [tab, setTab] = useState('headline');
  const [headline, setHeadline] = useState('');
  const [url, setUrl] = useState('');
  const [imageDataUrl, setImageDataUrl] = useState('');
  const [preview, setPreview] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  const onFile = async (file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      toast({ title: 'Invalid file', description: 'Please select an image file', variant: 'destructive' });
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setImageDataUrl(reader.result);
    reader.readAsDataURL(file);
    setPreview(URL.createObjectURL(file));
  };

  const submit = async () => {
    try {
      setLoading(true);
      setResult(null);
      const payload = { headline: headline || undefined, url: url || undefined, image_base64: imageDataUrl || undefined };
      const { data } = await api.post('/llm/analyze', payload, token ? undefined : undefined);
      setResult(data);
    } catch (e) {
      toast({ title: 'Analyze failed', description: e?.response?.data?.detail || 'Unexpected error', variant: 'destructive' });
    } finally {
      setLoading(false);
    }
  };

  const VerdictIcon = useMemo(() => {
    if (!result) return ShieldCheck;
    if (result.verdict === 'fake') return AlertTriangle;
    return ShieldCheck;
  }, [result]);

  return (
    <section className="analyzer-section">
      <div className="ff-container">
        <div className="hero">
          <h1>Verify news in seconds</h1>
          <p>AI-powered checks with web context, bias detection, and perspective rewrites. Paste a headline, a URL, or drop an image — we handle all three.</p>
        </div>

        <Card className="glass">
          <CardHeader>
            <CardTitle>Input</CardTitle>
          </CardHeader>
          <CardContent>
            <Tabs value={tab} onValueChange={setTab}>
              <TabsList>
                <TabsTrigger value="headline">Headline</TabsTrigger>
                <TabsTrigger value="url">URL</TabsTrigger>
                <TabsTrigger value="image">Image</TabsTrigger>
              </TabsList>
              <TabsContent value="headline">
                <div className="grid gap-2 mt-4">
                  <Label htmlFor="headline">Headline</Label>
                  <Textarea id="headline" placeholder="Type or paste a headline…" value={headline} onChange={(e) => setHeadline(e.target.value)} rows={3} />
                </div>
              </TabsContent>
              <TabsContent value="url">
                <div className="grid gap-2 mt-4">
                  <Label htmlFor="url">Article URL</Label>
                  <Input id="url" placeholder="https://example.com/news-article" value={url} onChange={(e) => setUrl(e.target.value)} />
                  <div className="hint">We fetch and analyze the page text snippet.</div>
                </div>
              </TabsContent>
              <TabsContent value="image">
                <div className="grid gap-2 mt-4">
                  <Label>Upload Image</Label>
                  <div className="upload">
                    <Input type="file" accept="image/*" onChange={(e) => onFile(e.target.files?.[0])} />
                    <Button variant="secondary"><Upload className="mr-2" size={16} />Choose file</Button>
                  </div>
                  {preview && (
                    <div className="preview">
                      <img alt="preview" src={preview} />
                    </div>
                  )}
                </div>
              </TabsContent>
            </Tabs>
            <div className="flex justify-end mt-6">
              <Button onClick={submit} disabled={loading || (!headline && !url && !imageDataUrl)}>
                {loading ? 'Analyzing…' : 'Analyze'}
              </Button>
            </div>
          </CardContent>
        </Card>

        {result &amp;&amp; (
          <div className="results">
            <Card className="glass">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <VerdictIcon size={18} /> Verdict
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid gap-3">
                  <div className="flex items-center gap-3">
                    <span className={`verdict ${result.verdict}`}>{result.verdict.toUpperCase()}</span>
                    <Badge variant="secondary">Bias: {result.bias_label}</Badge>
                  </div>
                  <div>
                    <Label>Fake likelihood: {(result.p_fake * 100).toFixed(0)}%</Label>
                    <Progress value={Math.min(100, Math.max(0, result.p_fake * 100))} />
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="grid rewrites">
              <Card className="glass">
                <CardHeader><CardTitle>Rewrites</CardTitle></CardHeader>
                <CardContent>
                  <Tabs defaultValue="left">
                    <TabsList>
                      <TabsTrigger value="left">Left</TabsTrigger>
                      <TabsTrigger value="right">Right</TabsTrigger>
                      <TabsTrigger value="neutral">Neutral</TabsTrigger>
                    </TabsList>
                    <TabsContent value="left"><div className="rewrite">{result.rewrites?.left}</div></TabsContent>
                    <TabsContent value="right"><div className="rewrite">{result.rewrites?.right}</div></TabsContent>
                    <TabsContent value="neutral"><div className="rewrite">{result.rewrites?.neutral}</div></TabsContent>
                  </Tabs>
                </CardContent>
              </Card>

              <Card className="glass">
                <CardHeader><CardTitle>Evidence</CardTitle></CardHeader>
                <CardContent>
                  <ul className="evidence">
                    {(result.evidence || []).map((e, i) => (
                      <li key={i}>• {e}</li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

function App() {
  const [authOpen, setAuthOpen] = useState(false);
  const { token, setToken } = useAuth();

  return (
    <div className="App">
      <BrowserRouter>
        <div className="parallax parallax-1" />
        <div className="parallax parallax-2" />
        <div className="parallax parallax-3" />

        <Header onOpenAuth={() => setAuthOpen(true)} />
        <main>
          {!token &amp;&amp; (
            <div className="auth-banner">
              <p>Please login to use FakeFinder.</p>
              <Button onClick={() => setAuthOpen(true)}><LogIn className="mr-2" size={16} /> Login / Sign up</Button>
            </div>
          )}
          <Analyzer token={token} />
        </main>

        <footer className="ff-footer">
          <div className="ff-container">
            <span>Built for hackathons — Real AI, real-time.</span>
          </div>
        </footer>

        <AuthDialog open={authOpen} onOpenChange={setAuthOpen} onAuthed={setToken} />
      </BrowserRouter>
    </div>
  );
}

export default App;