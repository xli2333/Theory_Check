import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import UploadZone from './components/UploadZone';
import LoadingState from './components/LoadingState';
import ResultDashboard from './components/ResultDashboard';
import PasswordGate from './components/PasswordGate';

// 更新 ReportData 接口以匹配后端结构
export interface ReportData {
  meta: { filename: string; detected_title: string; timestamp: string };
  summary: {
    high_risk_count: number;
    medium_risk_count: number;
    low_risk_count: number; // Added
    total_items: number;
    theory_count: number; // Added
    point_count: number; // Added
    theory_match_count: number; // Added
    point_match_count: number; // Added
    analysis: { // This structure is more detailed now
      risk_level: string;
      risk_color: string;
      risk_description: string;
      overlap_rate: number;
      total_matches: number;
      explicit_stats: { high: number; med: number; low: number };
      implicit_stats: { high: number; med: number; low: number };
      top_reference_years: Array<{ year: string; count: number }>;
      conflicting_cases: Array<{ filename: string; year: string; count: number }>;
      recommendations: Array<{ level: string; title: string; description: string; action: string }>;
    };
  };
  results: {
    explicit_theories: any[]; // Added explicit theories
    implicit_theories: any[]; // Added implicit theories
    theories: any[]; // Kept for compatibility if needed
    knowledge_points: any[];
  };
  report_type?: string; // Added report_type, optional for older data or default
}

function App() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [status, setStatus] = useState<'idle' | 'analyzing' | 'done'>('idle');
  const [report, setReport] = useState<ReportData | null>(null);
  const [progress, setProgress] = useState(0);
  const [loadingMessage, setLoadingMessage] = useState("正在初始化...");
  const [searchMode, setSearchMode] = useState<'general' | 'precision'>('general');

  // Check session storage on mount to persist login during refresh
  useEffect(() => {
    const isUnlocked = sessionStorage.getItem('fdc_unlocked');
    if (isUnlocked === 'true') {
      setIsAuthenticated(true);
    }
  }, []);

  const handleUnlock = () => {
    setIsAuthenticated(true);
    sessionStorage.setItem('fdc_unlocked', 'true');
  };

  const handleFileUpload = (file: File) => {
    setStatus('analyzing');
    setProgress(0);
    
    // Get WebSocket URL from env or default to local
    const wsUrl = import.meta.env.VITE_WS_URL || `ws://${window.location.hostname}:8000`;
    const ws = new WebSocket(`${wsUrl}/ws/${Date.now()}?mode=${searchMode}`);
    
    ws.onopen = () => {
      // 发送文件二进制数据
      file.arrayBuffer().then(buffer => {
        ws.send(buffer);
      });
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      // 过滤心跳消息，避免干扰进度条
      if (data.type === 'heartbeat') {
        return;
      }

      if (data.step === 'error') {
        alert("分析出错: " + data.message);
        setStatus('idle');
        ws.close();
        return;
      }

      if (data.step === 'done') {
        // 补全 filename，因为 WS 没传
        data.data.meta.filename = file.name;
        setReport(data.data);
        setStatus('done');
        ws.close();
      } else {
        // 更新进度和消息
        if (data.progress !== undefined) {
          setProgress(data.progress);
        }
        if (data.message) {
          setLoadingMessage(data.message);
        }
      }
    };

    ws.onerror = () => {
      alert("WebSocket 连接失败");
      setStatus('idle');
    };
  };

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center p-6 relative overflow-hidden bg-[#F8FAFC]">
      
      {isAuthenticated && (
        <motion.div 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className={`absolute top-8 left-8 z-50 flex items-center gap-3 cursor-pointer`}
          onClick={() => setStatus('idle')}
        >
          <span className="font-serif text-lg tracking-wider text-finance-ink font-bold border-l-4 border-finance-ink pl-3">
            复旦商业案例智能审查系统
          </span>
        </motion.div>
      )}

      <AnimatePresence mode='wait'>
        {!isAuthenticated ? (
           <PasswordGate key="gate" onUnlock={handleUnlock} />
        ) : (
          <>
            {status === 'idle' && (
              <UploadZone 
                key="upload" 
                onFileSelect={handleFileUpload} 
                searchMode={searchMode}
                onModeChange={setSearchMode}
              />
            )}
            
            {status === 'analyzing' && (
              <LoadingState key="loading" progress={progress} message={loadingMessage} />
            )}

            {status === 'done' && report && (
              <ResultDashboard key="result" data={report} onReset={() => setStatus('idle')} />
            )}
          </>
        )}
      </AnimatePresence>
      
    </div>
  );
}

export default App;