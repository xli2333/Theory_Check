import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronRight, ArrowLeft, ShieldCheck, Printer, Loader2, FileText } from 'lucide-react'; // Added FileText icon
import axios from 'axios';
import type { ReportData } from '../App';
import SummaryPanel from './SummaryPanel';

interface Props {
  data: ReportData;
  onReset: () => void;
}

export default function ResultDashboard({ data, onReset }: Props) {
  const [activeTab, setActiveTab] = useState<'summary' | 'theory'>('summary');
  const [isExporting, setIsExporting] = useState(false);

  // Modified handleDownload to accept reportType
  const handleDownload = async (reportType: 'dashboard' | 'paper') => {
    setIsExporting(true);
    try {
      const dataToSend = { ...data, report_type: reportType }; // Pass report_type to backend
      // Get API URL from env or default to local
      const apiUrl = import.meta.env.VITE_API_URL || `http://${window.location.hostname}:8000`;
      const response = await axios.post(`${apiUrl}/api/export`, dataToSend, {
        responseType: 'blob', // 关键：接收二进制流
      });

      // 创建下载链接
      const url = window.URL.createObjectURL(new Blob([response.data], { type: 'text/html' }));
      const link = document.createElement('a');
      link.href = url;
      // Use filename from headers if available, otherwise construct
      const contentDisposition = response.headers['content-disposition'];
      let filename = `FDC_Report_${data.meta.filename.replace('.pdf', '')}.html`;
      if (contentDisposition) {
        const filenameMatch = contentDisposition.match(/filename="([^"]+)"/);
        if (filenameMatch && filenameMatch[1]) {
          filename = filenameMatch[1];
        }
      }
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      
      // 清理
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Export failed", error);
      alert("报告生成失败，请检查后端服务。");
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      className="w-full max-w-7xl h-full flex flex-col gap-10 pt-12 pb-12"
    >
      {/* 头部摘要 */}
      <header className="flex justify-between items-end border-b border-finance-border pb-8">
        <div>
          <button onClick={onReset} className="flex items-center text-finance-subtle hover:text-finance-ink text-sm mb-6 transition-colors group">
            <ArrowLeft size={16} className="mr-1 group-hover:-translate-x-1 transition-transform" /> 返回上传页
          </button>
          <h1 className="font-serif text-4xl text-finance-ink mb-3 font-bold tracking-tight">{data.meta.detected_title}</h1>
          <p className="text-finance-subtle font-mono text-xs uppercase tracking-widest opacity-80">
            {data.meta.filename} · 检测时间: {new Date(data.meta.timestamp).toLocaleString('zh-CN')}
          </p>
        </div>
        
        <div className="flex gap-12 text-right">
          <div>
            <span className="block text-xs font-bold text-finance-subtle uppercase tracking-wider mb-2">精准重合项</span>
            <span className="text-5xl font-mono font-light text-finance-alert">
              {String(data.summary.analysis?.explicit_stats?.high || 0).padStart(2, '0')} {/* Updated to use analysis stats */}
            </span>
          </div>
          <div>
             <span className="block text-xs font-bold text-finance-subtle uppercase tracking-wider mb-2">提取概念总数</span>
             <span className="text-5xl font-mono font-light text-finance-ink">
              {String(data.summary.total_items).padStart(2, '0')}
            </span>
          </div>
        </div>
      </header>

      {/* 核心内容区 */}
      <div className="flex-1 grid grid-cols-12 gap-10 min-h-[600px]">
        
        {/* 左侧控制栏 */}
        <div className="col-span-3 flex flex-col gap-3">
            <button
                onClick={() => setActiveTab('summary')}
                className={`text-left px-5 py-4 rounded-lg transition-all border ${activeTab === 'summary' ? 'bg-finance-ink text-white border-finance-ink shadow-lg' : 'text-finance-subtle border-transparent hover:bg-white hover:border-gray-100'}`}
            >
                <span className="block text-xs uppercase tracking-wider opacity-60 mb-1">Overview</span>
                <span className="text-lg font-serif font-bold tracking-wide">智能总结</span>
            </button>

            <button
                onClick={() => setActiveTab('theory')}
                className={`text-left px-5 py-4 rounded-lg transition-all border ${activeTab === 'theory' ? 'bg-finance-ink text-white border-finance-ink shadow-lg' : 'text-finance-subtle border-transparent hover:bg-white hover:border-gray-100'}`}
            >
                <span className="block text-xs uppercase tracking-wider opacity-60 mb-1">Details</span>
                <span className="text-lg font-serif font-bold tracking-wide">理论架构查重</span>
            </button>

            <div className="mt-auto pt-8 border-t border-gray-100 flex flex-col gap-3"> {/* Changed to flex col for multiple buttons */}
                <button 
                    onClick={() => handleDownload('dashboard')} // Pass reportType: 'dashboard'
                    disabled={isExporting}
                    className="w-full border border-finance-ink text-finance-ink py-3 px-4 rounded-lg flex items-center justify-center gap-2 hover:bg-finance-ink hover:text-white transition-all duration-300 font-medium text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {isExporting ? (<Loader2 size={16} className="animate-spin" />) : (<Printer size={16} />)}
                    {isExporting ? "正在生成..." : "导出仪表盘报告"} {/* Updated button text */}
                </button>
                 <button 
                    onClick={() => handleDownload('paper')} // New button for paper report
                    disabled={isExporting}
                    className="w-full border border-slate-400 text-slate-700 py-3 px-4 rounded-lg flex items-center justify-center gap-2 hover:bg-slate-700 hover:text-white transition-all duration-300 font-medium text-sm disabled:opacity-50 disabled:cursor-not-allowed"
                >
                    {isExporting ? (<Loader2 size={16} className="animate-spin" />) : (<FileText size={16} />)}
                    {isExporting ? "正在生成..." : "导出文字版报告"}
                </button>
            </div>
        </div>

        {/* 右侧内容 */}
        <div className="col-span-9 bg-white rounded-xl shadow-sm border border-finance-border overflow-hidden flex flex-col min-h-[600px]">
            <div className="flex-1 overflow-y-auto p-8">
                <AnimatePresence mode='wait'>
                    {activeTab === 'summary' ? (
                        <motion.div
                            key="summary"
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                        >
                            <SummaryPanel summary={data.summary} />
                        </motion.div>
                    ) : (
                        <motion.div
                            key="theory"
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                            className="space-y-12"
                        >
                            {/* Explicit Section */}
                            {(data.results.explicit_theories && data.results.explicit_theories.length > 0) && (
                                <div>
                                    <div className="flex items-center justify-between mb-6 border-l-4 border-finance-ink pl-4">
                                        <h3 className="text-xl font-serif font-bold text-finance-ink">显性提及</h3>
                                        <span className="text-xs text-finance-subtle font-mono bg-gray-100 px-2 py-1 rounded">Explicit Mentions</span>
                                    </div>
                                    <div className="space-y-4">
                                        {data.results.explicit_theories.map((item: any, idx: number) => (
                                            <RiskItemRow key={`exp-${idx}`} item={item} />
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Implicit Section */}
                            {(data.results.implicit_theories && data.results.implicit_theories.length > 0) && (
                                <div>
                                    <div className="flex items-center justify-between mb-6 border-l-4 border-slate-400 pl-4">
                                        <h3 className="text-xl font-serif font-bold text-slate-700">隐性逻辑</h3>
                                        <span className="text-xs text-finance-subtle font-mono bg-gray-100 px-2 py-1 rounded">Implicit Logic</span>
                                    </div>
                                    <div className="space-y-4">
                                        {data.results.implicit_theories.map((item: any, idx: number) => (
                                            <RiskItemRow key={`imp-${idx}`} item={item} />
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Fallback / Empty State */}
                            {((!data.results.explicit_theories || data.results.explicit_theories.length === 0) && 
                              (!data.results.implicit_theories || data.results.implicit_theories.length === 0) &&
                              (!data.results.theories || data.results.theories.length === 0)) && (
                                <div className="text-center py-32 text-finance-subtle">
                                    <ShieldCheck size={64} strokeWidth={1} className="mx-auto mb-6 opacity-20" />
                                    <p className="text-lg font-serif text-finance-ink">未检测到重复风险</p>
                                    <p className="text-sm mt-2 opacity-60">该赛道内容原创度极高</p>
                                </div>
                            )}
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>

      </div>
    </motion.div>
  );
}

function RiskItemRow({ item }: { item: any }) {
    // 默认都不展开，点击展开再展开
    const [expanded, setExpanded] = useState(false);

    // 根据 risk_level 设置样式
    const getRiskStyle = (level: string) => {
        switch(level) {
            case '高度重合':
                return 'text-rose-700 bg-rose-50 border-rose-100';
            case '次重合':
                return 'text-amber-700 bg-amber-50 border-amber-100';
            case '重合':
                return 'text-blue-700 bg-blue-50 border-blue-100';
            default:
                return 'text-rose-700 bg-rose-50 border-rose-100';
        }
    };

    return (
        <div className={`border rounded-lg transition-all duration-300 ${expanded ? 'border-finance-border shadow-md bg-white' : 'border-transparent hover:bg-[#F8FAFC]'}`}>
            <div
                onClick={() => setExpanded(!expanded)}
                className="flex items-center p-5 cursor-pointer gap-6"
            >
                <span className={`px-3 py-1 rounded text-xs font-bold border whitespace-nowrap ${getRiskStyle(item.risk_level)}`}>
                    {item.risk_level}
                </span>

                <div className="flex-1 min-w-0">
                    <div className="flex flex-col gap-1">
                        <div className="flex items-center gap-3">
                            <span className="font-bold text-finance-ink text-lg font-serif truncate">{item.db_term}</span>
                            <span className="text-xs bg-gray-100 px-2 py-0.5 rounded text-finance-subtle">FDC 库标准词</span>
                        </div>
                        <p className="text-xs text-finance-subtle">
                            文中涉及变体: {item.matched_terms ? item.matched_terms.join(', ') : item.new_term}
                        </p>
                    </div>
                </div>

                <div className="flex items-center gap-2 text-xs text-finance-subtle">
                    <span>{item.evidence.length} 处历史来源</span>
                    <ChevronRight size={20} className={`transition-transform duration-300 ${expanded ? 'rotate-90' : ''}`} />
                </div>
            </div>

            <AnimatePresence>
                {expanded && (
                    <motion.div 
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: 'auto', opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        className="overflow-hidden"
                    >
                        <div className="p-6 pt-0 border-t border-gray-100 grid grid-cols-1 md:grid-cols-2 gap-8 bg-gray-50/50">
                            <div className="pt-6">
                                <p className="text-xs font-bold text-finance-subtle uppercase mb-3 tracking-wider">当前文稿语境</p>
                                <p className="text-sm font-serif text-finance-ink leading-loose text-justify bg-white p-4 rounded border border-gray-100 shadow-sm whitespace-pre-wrap mb-4">
                                    {item.new_context}
                                </p>

                                {/* AI 判定理由 Card */}
                                <div className="p-4 bg-slate-100/50 rounded border border-slate-200">
                                    <p className="text-xs font-bold text-slate-500 uppercase mb-2 flex items-center gap-2">
                                        <ShieldCheck size={12} /> AI 判定理由
                                    </p>
                                    <p className="text-xs text-slate-600 leading-relaxed">
                                        {item.rationale || "AI未提供详细判定理由。"}
                                    </p>
                                </div>
                            </div>
                            <div className="pt-6 md:border-l md:border-gray-200 md:pl-8">
                                <p className="text-xs font-bold text-finance-subtle uppercase mb-3 tracking-wider">历史案例库证据链 ({item.evidence.length})</p>
                                <div className="space-y-4 max-h-[400px] overflow-y-auto pr-2">
                                    {item.evidence.map((ev: any, i: number) => (
                                        <div key={i} className="group">
                                            <div className="flex items-baseline justify-between mb-1">
                                                <p className="text-sm font-bold text-finance-ink">{ev.filename}</p>
                                                <span className="text-xs font-mono text-finance-subtle bg-white border px-1 rounded">{ev.year}</span>
                                            </div>
                                            <p className="text-xs text-finance-subtle italic leading-relaxed border-l-2 border-gray-300 pl-3 group-hover:border-finance-ink transition-colors">
                                                "{ev.context.substring(0, 150)}..."
                                            </p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    )
}