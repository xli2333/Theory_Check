import React, { useRef } from 'react';
import { motion } from 'framer-motion';
import { Upload, Search, Crosshair } from 'lucide-react';

interface Props {
  onFileSelect: (file: File) => void;
  searchMode: 'general' | 'precision';
  onModeChange: (mode: 'general' | 'precision') => void;
}

export default function UploadZone({ onFileSelect, searchMode, onModeChange }: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) onFileSelect(file);
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="max-w-4xl w-full text-center flex flex-col items-center"
    >
      <h1 className="font-serif text-5xl md:text-6xl text-finance-ink mb-8 leading-tight font-bold">
        案例知识点及理论 <br/> <span className="italic text-finance-subtle font-normal">查重系统</span>
      </h1>
      
      <p className="text-finance-subtle text-lg mb-10 font-light tracking-wide max-w-2xl">
        深度解析文档语义，精准比对 FDC 历史库中的<span className="font-medium text-finance-ink">学术理论框架</span>与<span className="font-medium text-finance-ink">实务知识点</span>重复率。
      </p>

      {/* Mode Selection */}
      <div className="flex gap-12 mb-16 w-full justify-center">
        <div 
          onClick={() => onModeChange('general')}
          className={`cursor-pointer px-8 py-6 rounded-xl transition-all duration-500 flex-1 max-w-[280px] flex flex-col items-center gap-3 ${
            searchMode === 'general' 
              ? 'bg-slate-900 text-white shadow-2xl scale-110' 
              : 'bg-transparent text-slate-400 hover:bg-slate-50 hover:text-slate-600'
          }`}
        >
          <Search strokeWidth={1.5} size={28} className={searchMode === 'general' ? 'text-white' : 'text-current'} />
          <div className="font-serif font-bold text-xl tracking-wide">普通查询</div>
        </div>

        <div 
          onClick={() => onModeChange('precision')}
          className={`cursor-pointer px-8 py-6 rounded-xl transition-all duration-500 flex-1 max-w-[280px] flex flex-col items-center gap-3 ${
            searchMode === 'precision' 
              ? 'bg-slate-900 text-white shadow-2xl scale-110' 
              : 'bg-transparent text-slate-400 hover:bg-slate-50 hover:text-slate-600'
          }`}
        >
          <Crosshair strokeWidth={1.5} size={28} className={searchMode === 'precision' ? 'text-white' : 'text-current'} />
          <div className="font-serif font-bold text-xl tracking-wide">精准查询</div>
        </div>
      </div>

      <div 
        onClick={() => fileInputRef.current?.click()}
        className="group cursor-pointer relative py-20 px-12 rounded-xl border border-transparent hover:border-finance-border hover:bg-white hover:shadow-float transition-all duration-500 ease-out w-full max-w-2xl bg-white/50 backdrop-blur-sm"
      >
        <div className="flex flex-col items-center gap-6 transition-transform duration-300 group-hover:-translate-y-2">
            <div className="w-16 h-16 rounded-full bg-finance-ink text-white flex items-center justify-center mb-2 shadow-xl group-hover:scale-110 transition-transform duration-500">
                <Upload strokeWidth={1.5} size={28} />
            </div>
            <div>
                <span className="text-xl font-medium text-finance-ink block mb-2 font-serif">
                   {searchMode === 'general' ? '上传文件进行普通查询' : '上传文件进行精准查询'}
                </span>
                <span className="text-sm text-finance-subtle font-mono opacity-70">支持最大 50MB · 仅限 .pdf 格式</span>
            </div>
        </div>
        <input 
            type="file" 
            ref={fileInputRef}
            onChange={handleFileChange} 
            accept=".pdf" 
            className="hidden" 
        />
      </div>
    </motion.div>
  );
}
