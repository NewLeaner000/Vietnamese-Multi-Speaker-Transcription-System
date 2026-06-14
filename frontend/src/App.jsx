import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import { GoogleLogin } from '@react-oauth/google';
import { Upload, Play, Pause, CheckCircle, AlertCircle, Loader2, Users, Moon, Sun, Clock, FileText, Bot, MessageSquare, LogOut, History, Plus, Send, Trash2, RefreshCw, Languages, X } from 'lucide-react';
import { dictionaries } from './i18n';
import './index.css';

// Sử dụng biến môi trường VITE_API_URL khi chạy trên mạng (Vercel), nếu không có thì chạy mặc định ở Localhost
const API_BASE = import.meta.env.VITE_API_URL || `http://${window.location.hostname}:8000/api`;

function App() {
  // Theme & Auth & I18N State
  const [theme, setTheme] = useState('light');
  const [lang, setLang] = useState('vi');
  const t = (key) => dictionaries[lang]?.[key] || key;
  
  const [token, setToken] = useState(localStorage.getItem('token') || null);
  const [isLoginView, setIsLoginView] = useState(true);
  
  // Auth Form State
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState(''); // For register
  const [verificationCode, setVerificationCode] = useState('');
  const [countdown, setCountdown] = useState(0);
  const [authError, setAuthError] = useState('');
  
  // Forgot Password State
  const [showForgotPassword, setShowForgotPassword] = useState(false);
  const [resetStep, setResetStep] = useState(1); // 1: Email, 2: OTP + New Password
  const [resetEmail, setResetEmail] = useState('');
  const [resetOtp, setResetOtp] = useState('');
  const [resetNewPassword, setResetNewPassword] = useState('');
  const [resetError, setResetError] = useState('');
  const [resetMessage, setResetMessage] = useState('');

  // App State
  const [historyJobs, setHistoryJobs] = useState([]);
  const [trashJobs, setTrashJobs] = useState([]);
  const [activeTab, setActiveTab] = useState('history'); // 'history' | 'trash'
  const [selectedJobId, setSelectedJobId] = useState(null);
  
  // Job Execution State
  const [file, setFile] = useState(null);
  const [enrollmentFiles, setEnrollmentFiles] = useState([]);
  const [numSpeakers, setNumSpeakers] = useState(2);
  const [taskId, setTaskId] = useState(null);
  const [status, setStatus] = useState('IDLE'); // IDLE, UPLOADING, PROCESSING, SUMMARIZING, COMPLETED, ERROR
  const [progress, setProgress] = useState(0);
  const [progressMessage, setProgressMessage] = useState('');
  
  // Result State
  const [transcripts, setTranscripts] = useState([]);
  const [summary, setSummary] = useState(null);
  
  // Rename Speaker State
  const [editingSpeaker, setEditingSpeaker] = useState(null);
  const [newSpeakerName, setNewSpeakerName] = useState('');

  // Rename Job State
  const [editingJobId, setEditingJobId] = useState(null);
  const [newJobName, setNewJobName] = useState('');
  
  // Audio Player State
  const audioRef = useRef(null);
  const [audioError, setAudioError] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // Chat State
  const [chatMessages, setChatMessages] = useState([]);
  const [chatInput, setChatInput] = useState('');
  const [isChatting, setIsChatting] = useState(false);
  
  const fileInputRef = useRef(null);
  const pollIntervalRef = useRef(null);

  // 1. Theme Switcher Logic
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(theme === 'light' ? 'dark' : 'light');

  // 2. Lấy Lịch Sử (Khi đã có Token)
  const fetchHistory = async () => {
    if (!token) return;
    try {
      const res = await fetch(`${API_BASE}/audio/jobs`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setHistoryJobs(data.data);
      }
    } catch (e) {
      console.error(e);
    }
  };



  const fetchTrash = async () => {
    if (!token) return;
    try {
      const res = await fetch(`${API_BASE}/audio/jobs/trash`, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        const data = await res.json();
        setTrashJobs(data.data);
      }
    } catch (e) {
      console.error(e);
    }
  };

  useEffect(() => {
    if (token) {
      localStorage.setItem('token', token);
      if (activeTab === 'history') fetchHistory();
      else fetchTrash();
    } else {
      localStorage.removeItem('token');
    }
  }, [token, activeTab]);

  useEffect(() => {
    if (countdown > 0) {
      const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
      return () => clearTimeout(timer);
    }
  }, [countdown]);

  const handleSendCode = async () => {
    if (!email) {
      setAuthError('Vui lòng nhập Email trước khi gửi mã!');
      return;
    }
    setAuthError('');
    try {
      const res = await fetch(`${API_BASE}/auth/send-verification-code`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Không thể gửi mã xác thực');
      
      setCountdown(60);
      setAuthError(data.message || 'Mã xác thực đã được gửi tới email của bạn!');
    } catch (err) {
      setAuthError(err.message);
    }
  };

  // 3. Auth Handlers
  const handleAuth = async (e) => {
    e.preventDefault();
    setAuthError('');
    try {
      if (isLoginView) {
        // LOGIN
        const formData = new URLSearchParams();
        formData.append('username', username);
        formData.append('password', password);
        
        const res = await fetch(`${API_BASE}/auth/login`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: formData
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Login failed');
        setToken(data.access_token);
      } else {
        // REGISTER
        const res = await fetch(`${API_BASE}/auth/register`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, email, password, verification_code: verificationCode })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Register failed');
        // Auto login after register
        setIsLoginView(true);
        setAuthError('Đăng ký thành công! Đang đăng nhập...');
        setTimeout(() => handleAuth(e), 1000); // Trigger login
      }
    } catch (err) {
      setAuthError(err.message);
    }
  };

  const handleGoogleSuccess = async (credentialResponse) => {
    try {
      const res = await fetch(`${API_BASE}/auth/google-login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ credential: credentialResponse.credential })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Google Login failed');
      setToken(data.access_token);
    } catch (err) {
      setAuthError(err.message);
    }
  };

  const handleForgotPassword = async (e) => {
    e.preventDefault();
    setResetError('');
    setResetMessage('');
    try {
      if (resetStep === 1) {
        const res = await fetch(`${API_BASE}/auth/forgot-password`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email: resetEmail })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed to send OTP');
        setResetMessage(data.message);
        setResetStep(2);
      } else {
        const res = await fetch(`${API_BASE}/auth/reset-password`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            email: resetEmail, 
            verification_code: resetOtp,
            new_password: resetNewPassword 
          })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed to reset password');
        setResetMessage(data.message);
        setTimeout(() => {
          setShowForgotPassword(false);
          setResetStep(1);
          setResetEmail('');
          setResetOtp('');
          setResetNewPassword('');
          setIsLoginView(true);
        }, 2000);
      }
    } catch (err) {
      setResetError(err.message);
    }
  };

  const handleLogout = async () => {
    try {
      if (token) {
        await fetch(`${API_URL}/api/auth/logout`, {
          method: 'POST',
          headers: { 'Authorization': `Bearer ${token}` }
        });
      }
    } catch (e) {
      console.error("Lỗi khi đăng xuất khỏi máy chủ", e);
    }
    setToken(null);
    setHistoryJobs([]);
    setTrashJobs([]);
    setSelectedJobId(null);
    setStatus('IDLE');
    setChatMessages([]);
    if (audioRef.current) {
      audioRef.current.pause();
      setIsPlaying(false);
    }
  };

  // 4. File Upload Handlers
  const handleDragOver = (e) => e.preventDefault();
  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };
  const handleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file || !token) return;
    setStatus('UPLOADING');
    setProgress(0);
    setProgressMessage('Đang tải file lên...');
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('num_speakers', numSpeakers.toString());
    
    // Đính kèm các file mẫu giọng nói nếu có
    if (enrollmentFiles && enrollmentFiles.length > 0) {
      Array.from(enrollmentFiles).forEach(f => {
        formData.append('enrollment_files', f);
      });
    }

    try {
      const response = await fetch(`${API_BASE}/audio/upload`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` },
        body: formData,
      });
      if (!response.ok) throw new Error('Upload failed');
      
      const data = await response.json();
      setSelectedJobId(data.job_id);
      setTaskId(data.celery_task_id);
      setChatMessages([]);
      setStatus('PROCESSING');
      fetchHistory(); // Refresh sidebar
    } catch (error) {
      setStatus('ERROR');
      setProgressMessage('Lỗi tải file lên máy chủ. Có thể Token đã hết hạn.');
    }
  };

  // 5. Tiến trình Polling cho Job mới chạy
  useEffect(() => {
    if (status === 'PROCESSING' && taskId) {
      pollIntervalRef.current = setInterval(async () => {
        try {
          const res = await fetch(`${API_BASE}/audio/progress/${taskId}`);
          const data = await res.json();
          
          if (data.status === 'ĐANG CHẠY') {
            setProgress(data.percent);
            setProgressMessage(`Đang phân tích âm thanh... ${data.percent}%`);
          } else if (data.status === 'HOÀN THÀNH') {
            clearInterval(pollIntervalRef.current);
            setProgress(100);
            setStatus('SUMMARIZING');
          } else if (data.status === 'FAILED' || data.status === 'FAILURE') {
            clearInterval(pollIntervalRef.current);
            setStatus('ERROR');
            setProgressMessage('Lỗi hệ thống: Xử lý thất bại.');
          }
        } catch (error) {
          console.error(error);
        }
      }, 1000);
    }
    return () => clearInterval(pollIntervalRef.current);
  }, [status, taskId]);

  // 6. Lấy Transcript và Ra lệnh Summarize
  useEffect(() => {
    if (status === 'SUMMARIZING' && selectedJobId) {
      const fetchTranscriptsAndSummarize = async () => {
        try {
          setProgressMessage('Đang chạy AI Tóm tắt (Qwen)...');
          const tRes = await fetch(`${API_BASE}/audio/jobs/${selectedJobId}/transcripts`, {
            headers: { 'Authorization': `Bearer ${token}` }
          });
          const tData = await tRes.json();
          if (tData.data) setTranscripts(tData.data);

          await fetch(`${API_BASE}/audio/summarize/${selectedJobId}`, { 
            method: 'POST',
            headers: { 'Authorization': `Bearer ${token}` }
          });
          
          const summaryPoll = setInterval(async () => {
            const sRes = await fetch(`${API_BASE}/audio/jobs/${selectedJobId}/summary`, {
              headers: { 'Authorization': `Bearer ${token}` }
            });
            const sData = await sRes.json();
            if (sData.status === 'completed') {
              clearInterval(summaryPoll);
              setSummary(sData.data);
              setStatus('COMPLETED');
            }
          }, 3000);
        } catch (error) {
          setStatus('ERROR');
          setProgressMessage('Lỗi khi chạy tóm tắt.');
        }
      };
      fetchTranscriptsAndSummarize();
    }
  }, [status, selectedJobId]);

  // 7. Load lại dữ liệu khi người dùng Click vào một Job cũ trong Sidebar
  // 7. Load lại dữ liệu khi người dùng Click vào một Job cũ trong Sidebar
  const loadPastJob = async (job) => {
    setStatus('LOADING_PAST_JOB');
    setSelectedJobId(job.id);
    setFile(null); // Không cần file gốc khi xem lịch sử
    
    // Nếu Job đang xử lý dở dang, ta lấy lại task_id để resume thanh tiến độ
    if (job.status === 'pending' || job.status === 'processing') {
      setStatus(job.status.toUpperCase());
      setTaskId(job.celery_task_id);
      return; // Không fetch transcript vội vì chưa xong
    }
    
    try {
      // 1. Fetch transcript
      const tRes = await fetch(`${API_BASE}/audio/jobs/${job.id}/transcripts`, { headers: { 'Authorization': `Bearer ${token}` } });
      const tData = await tRes.json();
      setTranscripts(tData.data || []);
      
      // 2. Fetch summary
      const sRes = await fetch(`${API_BASE}/audio/jobs/${job.id}/summary`, { headers: { 'Authorization': `Bearer ${token}` } });
      const sData = await sRes.json();
      
      setChatMessages([]);
      setIsPlaying(false);
      setCurrentTime(0);

      // Nếu đã có Transcripts nhưng chưa có bản Tóm tắt (hoặc đang lỗi), tự động ép chạy lại Tóm tắt AI
      if (tData.data && tData.data.length > 0 && sData.status !== 'completed') {
        setStatus('SUMMARIZING');
      } else {
        setSummary(sData.data);
        setStatus('COMPLETED');
      }
    } catch (e) {
      console.error(e);
      setStatus('ERROR');
      setProgressMessage("Không thể tải lịch sử.");
    }
  };

  // Audio Handlers
  const handleTimeUpdate = () => {
    if (audioRef.current) setCurrentTime(audioRef.current.currentTime);
  };
  const handleLoadedMetadata = () => {
    if (audioRef.current) setDuration(audioRef.current.duration);
  };
  const handlePlayPause = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setIsPlaying(!isPlaying);
  };
  const handleSeek = (e) => {
    const bar = e.currentTarget;
    const clickX = e.clientX - bar.getBoundingClientRect().left;
    const newTime = (clickX / bar.offsetWidth) * duration;
    if (audioRef.current) audioRef.current.currentTime = newTime;
  };
  const jumpToTime = (timeInSeconds) => {
    if (audioRef.current) {
       audioRef.current.currentTime = timeInSeconds;
       audioRef.current.play();
       setIsPlaying(true);
    }
  };

  // Chat Handlers
  const handleSendChat = async (e) => {
    e.preventDefault();
    if (!chatInput.trim() || !selectedJobId || !token || isChatting) return;
    const msg = chatInput.trim();
    setChatInput('');
    setChatMessages(prev => [...prev, { role: 'user', content: msg }]);
    setIsChatting(true);
    try {
      const res = await fetch(`${API_BASE}/audio/chat/${selectedJobId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
        body: JSON.stringify({ message: msg })
      });
      const data = await res.json();
      if (res.ok) {
        setChatMessages(prev => [...prev, { role: 'ai', content: data.data }]);
      } else {
         setChatMessages(prev => [...prev, { role: 'ai', content: "Lỗi kết nối AI: " + (data.detail || "Unknown error") }]);
      }
    } catch(err) {
       setChatMessages(prev => [...prev, { role: 'ai', content: "Lỗi kết nối AI." }]);
    }
    setIsChatting(false);
  };



  // Format Utils
  const formatTime = (seconds) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = Math.floor(seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  };

  const getSpeakerColor = (speaker) => {
    if (!speaker) return '#3b82f6';
    const hash = speaker.split('').reduce((acc, char) => char.charCodeAt(0) + acc, 0);
    const colors = ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#ec4899'];
    return colors[hash % colors.length];
  };

  const handleRenameSpeaker = async (oldName, newName) => {
    if (!newName.trim() || oldName === newName) {
      setEditingSpeaker(null);
      return;
    }
    try {
      const res = await fetch(`${API_BASE}/audio/jobs/${selectedJobId}/rename_speaker`, {
        method: 'PUT',
        headers: { 
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json' 
        },
        body: JSON.stringify({ old_name: oldName, new_name: newName })
      });
      if (res.ok) {
        setTranscripts(prev => prev.map(t => t.speaker === oldName ? { ...t, speaker: newName } : t));
      }
    } catch (e) {
      console.error(e);
    }
    setEditingSpeaker(null);
  };

  const handleRenameJob = async (jobId, newName) => {
    if (!newName.trim()) {
      setEditingJobId(null);
      return;
    }
    try {
      const res = await fetch(`${API_BASE}/audio/jobs/${jobId}/rename_job`, {
        method: 'PUT',
        headers: { 
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json' 
        },
        body: JSON.stringify({ new_filename: newName })
      });
      if (res.ok) {
        setHistoryJobs(prev => prev.map(j => j.id === jobId ? { ...j, filename: newName } : j));
      }
    } catch (e) {
      console.error(e);
    }
    setEditingJobId(null);
  };

  const handleMoveToTrash = async (jobId) => {
    try {
      const res = await fetch(`${API_BASE}/audio/jobs/${jobId}/trash`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        if (selectedJobId === jobId) {
          setSelectedJobId(null);
          setStatus('IDLE');
        }
        fetchHistory();
        fetchTrash();
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleRestore = async (jobId) => {
    try {
      const res = await fetch(`${API_BASE}/audio/jobs/${jobId}/restore`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        fetchHistory();
        fetchTrash();
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleHardDelete = async (jobId) => {
    if(!window.confirm("Bạn có chắc chắn muốn xóa vĩnh viễn file này? Không thể khôi phục sau khi xóa.")) return;
    try {
      const res = await fetch(`${API_BASE}/audio/jobs/${jobId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });
      if (res.ok) {
        fetchTrash();
      }
    } catch (e) {
      console.error(e);
    }
  };

  // ==========================================
  // VIEW: AUTHENTICATION (LOGIN / REGISTER)
  // ==========================================
  if (!token) {
    return (
      <div className="app-container" style={{ alignItems: 'center', justifyContent: 'center' }}>
        <div className="upload-card fade-in" style={{ maxWidth: '400px', padding: '2rem' }}>
          <MessageSquare color="var(--accent-primary)" size={48} style={{ margin: '0 auto 1rem' }} />
          <h2 style={{ marginBottom: '1.5rem', textAlign: 'center' }}>{isLoginView ? t('loginTitle') : t('registerTitle')}</h2>
          
          <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.5rem' }}>
            <GoogleLogin
              onSuccess={handleGoogleSuccess}
              onError={() => { setAuthError('Google Login Failed'); }}
              useOneTap
            />
          </div>

          <div style={{ textAlign: 'center', marginBottom: '1rem', color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
            hoặc đăng nhập bằng tài khoản
          </div>

          <form onSubmit={handleAuth} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <input 
              type="text" placeholder={t('username')} required
              value={username} onChange={e => setUsername(e.target.value)}
              style={{ padding: '0.8rem', borderRadius: '6px', border: '1px solid var(--border-color)', background: 'var(--bg-hover)', color: 'var(--text-primary)' }}
            />
            {!isLoginView && (
              <>
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <input 
                    type="email" placeholder={t('email')} required
                    value={email} onChange={e => setEmail(e.target.value)}
                    style={{ flexGrow: 1, padding: '0.8rem', borderRadius: '6px', border: '1px solid var(--border-color)', background: 'var(--bg-hover)', color: 'var(--text-primary)' }}
                  />
                  <button 
                    type="button" 
                    className="btn-secondary" 
                    onClick={handleSendCode}
                    disabled={countdown > 0 || !email}
                    style={{ whiteSpace: 'nowrap' }}
                  >
                    {countdown > 0 ? `Chờ ${countdown}s` : 'Gửi mã'}
                  </button>
                </div>
                <input 
                  type="text" placeholder="Nhập mã xác thực (6 số)" required
                  value={verificationCode} onChange={e => setVerificationCode(e.target.value)}
                  style={{ padding: '0.8rem', borderRadius: '6px', border: '1px solid var(--border-color)', background: 'var(--bg-hover)', color: 'var(--text-primary)' }}
                />
              </>
            )}
            <input 
              type="password" placeholder={t('password')} required
              value={password} onChange={e => setPassword(e.target.value)}
              style={{ padding: '0.8rem', borderRadius: '6px', border: '1px solid var(--border-color)', background: 'var(--bg-hover)', color: 'var(--text-primary)' }}
            />
            
            {authError && <p style={{ color: 'var(--error)', fontSize: '0.85rem' }}>{authError}</p>}
            
            <button type="submit" className="btn-primary" style={{ justifyContent: 'center', marginTop: '0.5rem' }}>
              {isLoginView ? t('loginBtn') : t('registerBtn')}
            </button>
          </form>
          
          <div style={{ marginTop: '1.5rem', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
            {isLoginView ? t('noAccount') + " " : t('haveAccount') + " "}
            <span style={{ color: 'var(--accent-primary)', cursor: 'pointer', fontWeight: 500 }} onClick={() => { setIsLoginView(!isLoginView); setAuthError(''); }}>
              {isLoginView ? t('registerBtn') : t('loginBtn')}
            </span>
            {isLoginView && (
              <div style={{ marginTop: '0.5rem' }}>
                <span style={{ color: 'var(--text-secondary)', cursor: 'pointer', fontWeight: 500 }} onClick={() => setShowForgotPassword(true)}>
                  Quên mật khẩu?
                </span>
              </div>
            )}
          </div>
        </div>

        {/* FORGOT PASSWORD MODAL */}
        {showForgotPassword && (
          <div className="modal-overlay" style={{
            position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, 
            backgroundColor: 'rgba(0,0,0,0.5)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 1000
          }}>
            <div className="modal-content fade-in" style={{
              background: 'var(--bg-panel)', padding: '2rem', borderRadius: '12px', width: '90%', maxWidth: '400px',
              border: '1px solid var(--border-color)', boxShadow: '0 10px 25px rgba(0,0,0,0.1)', position: 'relative'
            }}>
              <button 
                onClick={() => { setShowForgotPassword(false); setResetStep(1); setResetError(''); setResetMessage(''); }} 
                style={{ position: 'absolute', top: '1rem', right: '1rem', background: 'none', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer' }}
              ><X size={20}/></button>
              
              <h3 style={{ marginBottom: '1rem' }}>Khôi phục mật khẩu</h3>
              
              <form onSubmit={handleForgotPassword} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {resetStep === 1 ? (
                  <>
                    <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Nhập email bạn đã đăng ký, chúng tôi sẽ gửi mã OTP gồm 6 chữ số.</p>
                    <input 
                      type="email" placeholder="Email của bạn" required
                      value={resetEmail} onChange={e => setResetEmail(e.target.value)}
                      style={{ padding: '0.8rem', borderRadius: '6px', border: '1px solid var(--border-color)', background: 'var(--bg-hover)', color: 'var(--text-primary)' }}
                    />
                  </>
                ) : (
                  <>
                    <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>Mã khôi phục đã được gửi tới <b>{resetEmail}</b></p>
                    <input 
                      type="text" placeholder="Mã OTP 6 số" required
                      autoComplete="one-time-code"
                      value={resetOtp} onChange={e => setResetOtp(e.target.value)}
                      style={{ padding: '0.8rem', borderRadius: '6px', border: '1px solid var(--border-color)', background: 'var(--bg-hover)', color: 'var(--text-primary)' }}
                    />
                    <input 
                      type="password" placeholder="Mật khẩu mới" required
                      autoComplete="new-password"
                      value={resetNewPassword} onChange={e => setResetNewPassword(e.target.value)}
                      style={{ padding: '0.8rem', borderRadius: '6px', border: '1px solid var(--border-color)', background: 'var(--bg-hover)', color: 'var(--text-primary)' }}
                    />
                  </>
                )}
                
                {resetError && <p style={{ color: 'var(--error)', fontSize: '0.85rem' }}>{resetError}</p>}
                {resetMessage && <p style={{ color: 'var(--success)', fontSize: '0.85rem' }}>{resetMessage}</p>}
                
                <button type="submit" className="btn-primary" style={{ justifyContent: 'center' }}>
                  {resetStep === 1 ? 'Gửi mã khôi phục' : 'Xác nhận đổi mật khẩu'}
                </button>
              </form>
            </div>
          </div>
        )}
      </div>
    );
  }

  // ==========================================
  // VIEW: MAIN WORKSPACE (SIDEBAR + CONTENT)
  // ==========================================
  return (
    <div className="app-container">
      {/* Top Navigation Bar */}
      <nav className="top-nav">
        <div className="logo-area">
          <MessageSquare color="var(--accent-primary)" />
          <span>ViMeet <span style={{fontWeight: 400, color: 'var(--text-secondary)'}}>Workspace</span></span>
        </div>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <button className="theme-toggle" onClick={() => setLang(lang === 'vi' ? 'en' : 'vi')} title="Ngôn ngữ / Language">
            <Languages size={20} />
            <span style={{ fontSize: '0.75rem', fontWeight: 600, marginLeft: '4px' }}>{lang.toUpperCase()}</span>
          </button>
          <button className="theme-toggle" onClick={toggleTheme}>
            {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
          </button>
          <button className="btn-secondary" onClick={handleLogout} style={{ border: 'none', padding: '0.5rem', color: 'var(--error)' }} title="Đăng xuất">
            <LogOut size={20} />
          </button>
        </div>
      </nav>

      <main className="main-content">
        
        {/* LEFT SIDEBAR: History & Trash */}
        <div className="sidebar" style={{
          width: '280px',
          minWidth: '280px',
          flexShrink: 0,
          borderRight: '1px solid var(--border-color)',
          background: 'var(--bg-panel)',
          display: 'flex',
          flexDirection: 'column',
          zIndex: 5
        }}>
          <div style={{ padding: '1rem' }}>
            <button className="btn-primary" style={{ width: '100%', justifyContent: 'center' }} onClick={() => { setStatus('IDLE'); setSelectedJobId(null); setFile(null); }}>
              <Plus size={18} /> {t('newMeeting')}
            </button>
          </div>
          
          <div style={{ display: 'flex', padding: '0 1rem', marginBottom: '0.5rem', gap: '0.5rem' }}>
            <button 
              onClick={() => setActiveTab('history')}
              style={{ flex: 1, padding: '0.5rem', borderRadius: '6px', border: 'none', background: activeTab === 'history' ? 'var(--bg-hover)' : 'transparent', color: activeTab === 'history' ? 'var(--text-primary)' : 'var(--text-secondary)', fontWeight: 600, cursor: 'pointer', fontSize: '0.8rem', transition: 'all 0.2s' }}
            >
              {t('history')}
            </button>
            <button 
              onClick={() => setActiveTab('trash')}
              style={{ flex: 1, padding: '0.5rem', borderRadius: '6px', border: 'none', background: activeTab === 'trash' ? 'var(--bg-hover)' : 'transparent', color: activeTab === 'trash' ? 'var(--text-primary)' : 'var(--text-secondary)', fontWeight: 600, cursor: 'pointer', fontSize: '0.8rem', transition: 'all 0.2s' }}
            >
              {t('trash')}
            </button>
          </div>
          
          <div style={{ flexGrow: 1, overflowY: 'auto', padding: '0 0.5rem', scrollbarGutter: 'stable' }}>
            {activeTab === 'history' && historyJobs.map(job => (
              <div 
                key={job.id} 
                onClick={() => { if (editingJobId !== job.id) loadPastJob(job); }}
                style={{
                  padding: '0.75rem',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  marginBottom: '0.25rem',
                  background: selectedJobId === job.id ? 'var(--bg-hover)' : 'transparent',
                  borderLeft: selectedJobId === job.id ? '3px solid var(--accent-primary)' : '3px solid transparent',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  position: 'relative'
                }}
                className="job-item"
              >
                <History size={16} color="var(--text-secondary)" style={{minWidth: '16px'}} />
                <div style={{ overflow: 'hidden', flexGrow: 1 }}>
                  {editingJobId === job.id ? (
                    <input 
                      type="text" 
                      autoFocus
                      value={newJobName}
                      onChange={(e) => setNewJobName(e.target.value)}
                      onBlur={() => handleRenameJob(job.id, newJobName)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') handleRenameJob(job.id, newJobName);
                        if (e.key === 'Escape') setEditingJobId(null);
                      }}
                      onClick={(e) => e.stopPropagation()}
                      style={{ width: '100%', fontSize: '0.9rem', padding: '0.1rem', background: 'var(--bg-secondary)', color: 'var(--text-primary)', border: '1px solid var(--border-color)', borderRadius: '4px' }}
                    />
                  ) : (
                    <div 
                      onClick={(e) => {
                        e.stopPropagation();
                        setEditingJobId(job.id);
                      }}
                      title={t('rename')}
                      style={{ fontSize: '0.9rem', fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', color: 'var(--text-primary)' }}
                    >
                      {job.filename}
                    </div>
                  )}
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                    {t('status_' + job.status) || job.status}
                  </div>
                </div>
                <button 
                  onClick={(e) => { e.stopPropagation(); handleMoveToTrash(job.id); }}
                  className="trash-btn"
                  title={t('delete')}
                  style={{ background: 'transparent', border: 'none', color: 'var(--text-secondary)', cursor: 'pointer', padding: '0.2rem', display: 'flex', alignItems: 'center' }}
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
            {activeTab === 'history' && historyJobs.length === 0 && (
              <p style={{ padding: '1rem', textAlign: 'center', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>{t('emptyHistory')}</p>
            )}

            {activeTab === 'trash' && trashJobs.map(job => (
              <div 
                key={job.id} 
                style={{
                  padding: '0.75rem',
                  borderRadius: '6px',
                  marginBottom: '0.25rem',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.75rem',
                  border: '1px solid var(--border-color)',
                  background: 'var(--bg-secondary)'
                }}
              >
                <div style={{ overflow: 'hidden', flexGrow: 1 }}>
                  <div style={{ fontSize: '0.9rem', fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', color: 'var(--text-secondary)', textDecoration: 'line-through' }}>
                    {job.filename}
                  </div>
                </div>
                <button 
                  onClick={() => handleRestore(job.id)}
                  title="Khôi phục"
                  style={{ background: 'transparent', border: 'none', color: 'var(--accent-primary)', cursor: 'pointer', padding: '0.2rem', display: 'flex', alignItems: 'center' }}
                >
                  <RefreshCw size={14} />
                </button>
                <button 
                  onClick={() => handleHardDelete(job.id)}
                  title="Xóa vĩnh viễn"
                  style={{ background: 'transparent', border: 'none', color: 'var(--error)', cursor: 'pointer', padding: '0.2rem', display: 'flex', alignItems: 'center' }}
                >
                  <Trash2 size={14} />
                </button>
              </div>
            ))}
            {activeTab === 'trash' && trashJobs.length === 0 && (
              <div style={{ padding: '2rem 1rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
                <Trash2 size={32} style={{ margin: '0 auto 1rem', opacity: 0.5 }} />
                <p style={{ fontSize: '0.85rem' }}>Thùng rác trống.</p>
              </div>
            )}
          </div>
        </div>

        {/* MAIN WORKSPACE PANEL */}
        <div style={{ flexGrow: 1, position: 'relative', display: 'flex' }}>
          
          {/* VIEW 1: UPLOAD & PROGRESS */}
          {(status === 'IDLE' || status === 'UPLOADING' || status === 'PROCESSING' || status === 'ERROR') && (
            <div className="upload-wrapper fade-in" style={{ width: '100%' }}>
              <div className="upload-card">
                <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>{t('uploadTitle')}</h2>
                <p style={{ color: 'var(--text-secondary)', marginBottom: '2rem' }}>{t('uploadSubtitle')}</p>
                
                {(status === 'IDLE' || status === 'ERROR') ? (
                  <>
                    <div 
                      onDragOver={handleDragOver}
                      onDrop={handleDrop}
                      onClick={() => fileInputRef.current?.click()}
                      style={{
                        border: '2px dashed var(--border-color)',
                        borderRadius: '8px',
                        padding: '3rem',
                        cursor: 'pointer',
                        background: 'var(--bg-hover)',
                        marginBottom: '1.5rem'
                      }}
                    >
                      <Upload size={32} color="var(--text-secondary)" style={{ margin: '0 auto 1rem' }} />
                      <p style={{ fontWeight: 500 }}>{file ? file.name : t('uploadTitle')}</p>
                      <input type="file" ref={fileInputRef} style={{ display: 'none' }} accept="audio/*" onChange={handleFileSelect}/>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', marginBottom: '2rem', flexWrap: 'wrap' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'var(--bg-hover)', padding: '0.5rem 1rem', borderRadius: '6px' }}>
                        <Users size={16} color="var(--text-secondary)" />
                        <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>{t('numSpeakers')}:</span>
                        <input 
                          type="number"
                          value={numSpeakers}
                          onChange={(e) => setNumSpeakers(e.target.value === '' ? '' : parseInt(e.target.value))}
                          onBlur={(e) => {
                            if (!e.target.value || parseInt(e.target.value) < 1) setNumSpeakers(1);
                          }}
                          style={{ background: 'transparent', border: '1px solid var(--border-color)', color: 'var(--text-primary)', width: '80px', padding: '0.4rem', borderRadius: '4px', textAlign: 'center' }}
                        />
                      </div>
                      
                      {/* Tải file Enrollment */}
                      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'var(--bg-hover)', padding: '0.5rem 1rem', borderRadius: '6px' }}>
                          <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>{t('enrollmentLabel')}:</span>
                          <input 
                            type="file" 
                            multiple 
                            accept="audio/*,.zip" 
                            onChange={(e) => {
                              if (e.target.files) {
                                setEnrollmentFiles(prev => [...prev, ...Array.from(e.target.files)]);
                              }
                            }}
                            style={{ fontSize: '0.85rem', color: 'var(--text-primary)', maxWidth: '200px' }}
                          />
                        </div>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontStyle: 'italic', paddingLeft: '0.5rem' }}>
                          {t('enrollmentNote')}
                        </span>
                        
                        {/* Danh sách file đã chọn */}
                        {enrollmentFiles.length > 0 && (
                          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.5rem' }}>
                            {enrollmentFiles.map((f, idx) => (
                              <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', background: 'var(--bg-secondary)', padding: '0.2rem 0.5rem', borderRadius: '4px', border: '1px solid var(--border-color)' }}>
                                <span style={{ fontSize: '0.8rem', color: 'var(--text-primary)' }}>{f.name}</span>
                                <button 
                                  onClick={() => setEnrollmentFiles(prev => prev.filter((_, i) => i !== idx))}
                                  style={{ background: 'transparent', border: 'none', color: 'var(--danger-color)', cursor: 'pointer', padding: 0, display: 'flex', alignItems: 'center' }}
                                  title="Xóa file này"
                                >
                                  ×
                                </button>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>

                    <button className="btn-primary" style={{ width: '100%', justifyContent: 'center' }} onClick={handleUpload} disabled={!file}>
                      {t('startProcess')}
                    </button>

                    {status === 'ERROR' && (
                      <div style={{ color: 'var(--error)', marginTop: '1rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
                        <AlertCircle size={16} /> <span style={{ fontSize: '0.9rem' }}>{progressMessage}</span>
                      </div>
                    )}
                  </>
                ) : (
                  <div style={{ padding: '2rem 0' }}>
                    <Loader2 size={40} color="var(--accent-primary)" style={{ animation: 'spin 1s linear infinite', margin: '0 auto 1.5rem' }} />
                    <h3 style={{ fontSize: '1.1rem', marginBottom: '0.5rem' }}>{t('processing')}</h3>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>{progressMessage}</p>
                    <div className="progress-container">
                      <div className="progress-fill" style={{ width: `${progress}%` }}></div>
                    </div>
                  </div>
                )}
              </div>
              <style>{`@keyframes spin { 100% { transform: rotate(360deg); } }`}</style>
            </div>
          )}

          {/* VIEW 2: LOADING PAST JOB */}
          {status === 'LOADING_PAST_JOB' && (
            <div className="upload-wrapper fade-in" style={{ width: '100%' }}>
              <Loader2 size={40} color="var(--accent-primary)" style={{ animation: 'spin 1s linear infinite' }} />
            </div>
          )}

          {/* VIEW 3: RESULTS (Notta Style) */}
          {(status === 'COMPLETED' || status === 'SUMMARIZING') && (
            <>
              {/* Left Column: Full Transcript */}
                <div className="left-pane fade-in" style={{ paddingLeft: '4rem', paddingRight: '4rem' }}>
                  <div style={{ maxWidth: '800px', margin: '0 auto' }}>
                    <h1 style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>
                      {file ? file.name : (historyJobs.find(j => j.id === selectedJobId)?.filename || t('transcriptTab'))}
                    </h1>
                    <p style={{ color: 'var(--text-secondary)', marginBottom: '3rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <Clock size={16} /> {t('completed')}
                    </p>

                  <div className="transcript-list">
                    {transcripts.map((tr, idx) => (
                      <div className="transcript-block" key={idx} onClick={() => jumpToTime(tr.start)} style={{cursor: 'pointer'}}>
                        <div className="speaker-col">
                          <div className="speaker-avatar" style={{ background: getSpeakerColor(tr.speaker) }}>
                            {tr.speaker.charAt(0).toUpperCase()}
                          </div>
                          {editingSpeaker === tr.speaker ? (
                            <input 
                              type="text" 
                              autoFocus
                              value={newSpeakerName}
                              onChange={(e) => setNewSpeakerName(e.target.value)}
                              onBlur={() => handleRenameSpeaker(tr.speaker, newSpeakerName)}
                              onKeyDown={(e) => {
                                if (e.key === 'Enter') handleRenameSpeaker(tr.speaker, newSpeakerName);
                                if (e.key === 'Escape') setEditingSpeaker(null);
                              }}
                              onClick={(e) => e.stopPropagation()}
                              style={{ width: '80px', fontSize: '0.8rem', padding: '0.1rem', background: 'var(--bg-secondary)', color: 'var(--text-primary)', border: '1px solid var(--border-color)', borderRadius: '4px', textAlign: 'center' }}
                            />
                          ) : (
                            <span 
                              className="speaker-name" 
                              onClick={(e) => {
                                e.stopPropagation();
                                setEditingSpeaker(tr.speaker);
                              setNewSpeakerName(tr.speaker);
                            }}
                            title={t('rename')}
                            style={{ cursor: 'text' }}
                          >
                              {tr.speaker.toUpperCase()}
                            </span>
                          )}
                          <span className="speaker-time">{formatTime(tr.start)}</span>
                        </div>
                        <div className="text-col" style={{ background: (currentTime >= tr.start && currentTime < tr.end) ? 'var(--bg-hover)' : 'transparent', borderRadius: '4px', padding: '0.25rem' }}>
                          {tr.text}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Right Column: AI Summary & Chatbot */}
              <div className="right-pane fade-in" style={{ display: 'flex', flexDirection: 'column' }}>
                <h2 style={{ fontSize: '1.25rem', marginBottom: '1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Bot size={20} color="var(--accent-primary)" /> Ani Assistant
                </h2>
                
                {status === 'SUMMARIZING' ? (
                  <div style={{ textAlign: 'center', padding: '3rem 0', color: 'var(--text-secondary)' }}>
                    <Loader2 size={32} style={{ animation: 'spin 1s linear infinite', margin: '0 auto 1rem' }} />
                    <p>{t('processing')}</p>
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', flexGrow: 1, overflow: 'hidden' }}>
                    {/* Summary Card */}
                    <div className="summary-card" style={{ flexShrink: 0, maxHeight: '40%', overflowY: 'auto', marginBottom: '1rem' }}>
                      <div className="summary-title"><FileText size={18} /> {t('summaryTab')}</div>
                      <div className="markdown-content" style={{ color: 'var(--text-primary)', fontSize: '0.95rem' }}>
                        {summary ? <ReactMarkdown>{summary}</ReactMarkdown> : t('emptyHistory')}
                      </div>
                    </div>

                    {/* Chat Messages */}
                    <div className="chat-messages" style={{ flexGrow: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: '1rem', padding: '0.5rem 0' }}>
                      {chatMessages.length === 0 && (
                        <div style={{ textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '2rem' }}>
                          {t('chatPlaceholder')}
                        </div>
                      )}
                      {chatMessages.map((msg, idx) => (
                        <div key={idx} style={{ alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '85%', background: msg.role === 'user' ? 'var(--accent-primary)' : 'var(--bg-hover)', color: msg.role === 'user' ? 'white' : 'var(--text-primary)', padding: '0.75rem 1rem', borderRadius: '8px', fontSize: '0.95rem', whiteSpace: 'pre-line' }}>
                          {msg.content}
                        </div>
                      ))}
                      {isChatting && (
                        <div style={{ alignSelf: 'flex-start', background: 'var(--bg-hover)', padding: '0.75rem 1rem', borderRadius: '8px', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                          <Loader2 size={16} className="spin" color="var(--text-secondary)" /> <span style={{fontSize: '0.9rem', color: 'var(--text-secondary)'}}>{t('processing')}</span>
                        </div>
                      )}
                    </div>

                    {/* Chat Input */}
                    <form onSubmit={handleSendChat} style={{ display: 'flex', gap: '0.5rem', marginTop: '1rem', borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
                      <input 
                        type="text" 
                        placeholder={t('chatPlaceholder')} 
                        value={chatInput} 
                        onChange={(e) => setChatInput(e.target.value)}
                        disabled={isChatting}
                        style={{ flexGrow: 1, padding: '0.75rem 1rem', borderRadius: '24px', border: '1px solid var(--border-color)', background: 'var(--bg-hover)', color: 'var(--text-primary)' }}
                      />
                      <button type="submit" disabled={isChatting || !chatInput.trim()} className="btn-primary" style={{ borderRadius: '50%', width: '42px', height: '42px', padding: 0, justifyContent: 'center' }}>
                        <Send size={18} />
                      </button>
                    </form>
                  </div>
                )}
              </div>

              {/* Bottom Audio Player Bar */}
              {audioError ? (
                <div className="bottom-player fade-in" style={{ left: 0, justifyContent: 'center', background: 'var(--bg-hover)', color: 'var(--text-secondary)' }}>
                  <AlertCircle size={18} style={{ marginRight: '8px' }} />
                  <span style={{ fontSize: '0.9rem' }}>{t('audioAutoDeleted') || "File âm thanh gốc đã bị xóa tự động để tiết kiệm dung lượng (Văn bản vẫn được giữ lại)."}</span>
                </div>
              ) : (
                <div className="bottom-player fade-in" style={{ left: 0 }}>
                  {selectedJobId && (
                    <audio 
                      ref={audioRef}
                      src={selectedJobId ? `${API_BASE}/audio/jobs/${selectedJobId}/audio?token=${token}` : ''}
                      onTimeUpdate={handleTimeUpdate}
                      onLoadedMetadata={handleLoadedMetadata}
                      onEnded={() => setIsPlaying(false)}
                      onError={() => setAudioError(true)}
                      controls
                      style={{ width: '100%', marginBottom: '1rem', display: 'none' }}
                    />
                  )}
                  <button className="btn-primary" onClick={handlePlayPause} style={{ borderRadius: '50%', width: '48px', height: '48px', padding: 0, justifyContent: 'center', flexShrink: 0 }}>
                    {isPlaying ? <Pause fill="currentColor" size={20} /> : <Play fill="currentColor" size={20} />}
                  </button>
                  <div style={{ flexGrow: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.25rem' }}>
                      <span>{formatTime(currentTime)}</span>
                      <span>{formatTime(duration)}</span>
                    </div>
                    <div className="progress-container" onClick={handleSeek} style={{ marginTop: 0, height: '8px', cursor: 'pointer' }}>
                      <div className="progress-fill" style={{ width: `${duration ? (currentTime / duration) * 100 : 0}%`, transition: 'width 0.1s linear' }}></div>
                    </div>
                  </div>
                </div>
              )}
            </>
          )}

        </div>
      </main>
    </div>
  );
}

export default App;
