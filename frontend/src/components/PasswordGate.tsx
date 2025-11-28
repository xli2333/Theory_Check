import React, { useState } from 'react';
import { motion } from 'framer-motion';


interface Props {
  onUnlock: () => void;
}

export default function PasswordGate({ onUnlock }: Props) {
  const [password, setPassword] = useState('');
  const [error, setError] = useState(false);
  
  // 从环境变量获取密码，如果没有设置则默认一个（为了安全最好提醒用户设置）
  // 注意：在 Vercel 中需要设置 VITE_APP_PASSWORD
  const CORRECT_PASSWORD = import.meta.env.VITE_APP_PASSWORD || 'FDSM2025';

  const handleSubmit = (e?: React.FormEvent) => {
    e?.preventDefault();
    if (password === CORRECT_PASSWORD) {
      onUnlock();
    } else {
      setError(true);
      setTimeout(() => setError(false), 800);
    }
  };

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center p-6 bg-[#F8FAFC]">
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-xl w-full text-center"
      >
        {/* Heading aligned with UploadZone style */}
        <h1 className="font-serif text-5xl md:text-6xl text-finance-ink mb-8 leading-tight font-bold">
          FDC <br/> <span className="italic text-finance-subtle font-normal">内部访问系统</span>
        </h1>

        <form onSubmit={handleSubmit} className="relative w-full max-w-md mx-auto">
          <div className="relative group">
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="请输入访问密码"
              className="w-full bg-transparent text-center text-3xl md:text-4xl font-serif text-finance-ink placeholder-gray-200 border-none outline-none focus:ring-0 py-4 tracking-widest transition-all"
              autoFocus
            />
            {/* Custom Underline */}
            <div className={`absolute bottom-0 left-0 w-full h-0.5 transition-all duration-500 ${error ? 'bg-rose-500 scale-x-100' : 'bg-finance-ink scale-x-50 group-hover:scale-x-75 focus-within:scale-x-100'}`} />
          </div>

          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: error ? 1 : 0 }}
            className="text-rose-500 text-xs font-bold tracking-widest mt-4 uppercase"
          >
            密码错误
          </motion.p>

          <button 
            type="submit" 
            className="mt-12 opacity-0 w-0 h-0 overflow-hidden" // Hidden submit button for Enter key support
          >
            Submit
          </button>
        </form>
        
        <div className="mt-16 text-finance-subtle text-xs font-mono uppercase tracking-[0.2em] opacity-40">
           仅限内部人员访问
        </div>
      </motion.div>
    </div>
  );
}
