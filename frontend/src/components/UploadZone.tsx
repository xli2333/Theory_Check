import React, { useRef } from 'react';
import { motion } from 'framer-motion';
import { Upload } from 'lucide-react';

interface Props {
  onFileSelect: (file: File) => void;
}

export default function UploadZone({ onFileSelect }: Props) {
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
      className="max-w-3xl w-full text-center"
    >
      <h1 className="font-serif text-5xl md:text-6xl text-finance-ink mb-8 leading-tight font-bold">
        案例知识点及理论 <br/> <span className="italic text-finance-subtle font-normal">查重系统</span>
      </h1>
      
      <p className="text-finance-subtle text-lg mb-12 font-light tracking-wide">
        深度解析文档语义，精准比对 FDC 历史库中的<span className="font-medium text-finance-ink">学术理论框架</span>与<span className="font-medium text-finance-ink">实务知识点</span>重复率。
      </p>

      <div 
        onClick={() => fileInputRef.current?.click()}
        className="group cursor-pointer relative py-24 px-12 rounded-xl border border-transparent hover:border-finance-border hover:bg-white hover:shadow-float transition-all duration-500 ease-out"
      >
        <div className="flex flex-col items-center gap-6 transition-transform duration-300 group-hover:-translate-y-2">
            <div className="w-16 h-16 rounded-full bg-finance-ink text-white flex items-center justify-center mb-2 shadow-xl group-hover:scale-110 transition-transform duration-500">
                <Upload strokeWidth={1.5} size={28} />
            </div>
            <div>
                <span className="text-xl font-medium text-finance-ink block mb-2 font-serif">点击上传文件</span>
                <span className="text-sm text-finance-subtle font-mono opacity-70">支持最大 20MB · 仅限 .pdf 格式</span>
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
