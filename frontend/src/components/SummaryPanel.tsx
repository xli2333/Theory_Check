
import { motion } from 'framer-motion';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip as RechartsTooltip, ResponsiveContainer, Cell, CartesianGrid
} from 'recharts';
import { Calendar, Layers, FileText } from 'lucide-react';

interface Props {
  summary: any;
}



export default function SummaryPanel({ summary }: Props) {
  const analysis = summary.analysis || {};
  const riskLevel = analysis.risk_level || '分析中';
  
  // Data for Charts
  const yearData = (analysis.top_reference_years || []).map((y: any) => ({
    name: y.year,
    count: y.count
  })).sort((a: any, b: any) => parseInt(a.name) - parseInt(b.name));

  const caseData = analysis.conflicting_cases || [];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: { 
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 10 },
    visible: { opacity: 1, y: 0 }
  };

  // Dynamic Text Colors based on Risk
  const getRiskColor = (level: string) => {
      if (level === '高风险') return 'text-rose-700';
      if (level === '中等风险') return 'text-orange-600';
      if (level === '低风险') return 'text-emerald-700';
      if (level === '逻辑风险') return 'text-amber-700'; // Keep for legacy/implicit if needed
      if (level === '中度风险') return 'text-orange-600'; // Keep for legacy
      return 'text-emerald-700';
  };

  const expStats = analysis.explicit_stats || { high: 0, med: 0, low: 0 };
  const impStats = analysis.implicit_stats || { high: 0, med: 0, low: 0 };

  return (
    <motion.div 
      className="space-y-12 pb-12"
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      {/* 1. Executive Summary */}
      <motion.div variants={itemVariants} className="flex flex-col gap-8">
        <div>
           <h2 className="text-xs font-bold tracking-widest text-slate-400 uppercase mb-4">
            分析结论摘要
          </h2>
          <div className="flex flex-col md:flex-row gap-12 items-start">
             <div className="flex-1">
                 <div className="flex items-baseline gap-4 mb-4">
                     <h1 className={`text-6xl font-serif font-bold tracking-tight ${getRiskColor(riskLevel)}`}>
                        {riskLevel}
                     </h1>
                 </div>
                 <p className="text-xl font-serif text-slate-600 leading-relaxed max-w-2xl">
                    {analysis.risk_description || "系统正在分析文稿的原创性..."}
                 </p>
             </div>
             
             {/* Key Stats Grid - Updated for Explicit/Implicit Breakdown */}
             <div className="grid grid-cols-2 gap-8 min-w-[300px]">
                 {/* Explicit Stats */}
                 <div className="space-y-3">
                    <h4 className="text-[10px] uppercase tracking-widest text-slate-400 font-bold border-b border-slate-100 pb-2">显性框架 (Explicit)</h4>
                    <div className="grid grid-cols-3 gap-2">
                        <div className="text-center">
                            <div className="text-2xl font-bold text-slate-900">{expStats.high}</div>
                            <div className="text-[9px] text-slate-400">高度</div>
                        </div>
                        <div className="text-center">
                            <div className="text-2xl font-bold text-slate-900">{expStats.med}</div>
                            <div className="text-[9px] text-slate-400">次重</div>
                        </div>
                        <div className="text-center">
                             <div className="text-2xl font-bold text-slate-900">{expStats.low}</div>
                             <div className="text-[9px] text-slate-400">重合</div>
                        </div>
                    </div>
                 </div>

                 {/* Implicit Stats */}
                 <div className="space-y-3">
                    <h4 className="text-[10px] uppercase tracking-widest text-slate-400 font-bold border-b border-slate-100 pb-2">隐性逻辑 (Implicit)</h4>
                    <div className="grid grid-cols-3 gap-2">
                        <div className="text-center">
                            <div className="text-2xl font-bold text-slate-900">{impStats.high}</div>
                            <div className="text-[9px] text-slate-400">高度</div>
                        </div>
                        <div className="text-center">
                             <div className="text-2xl font-bold text-slate-900">{impStats.med}</div>
                             <div className="text-[9px] text-slate-400">次重</div>
                        </div>
                        <div className="text-center">
                             <div className="text-2xl font-bold text-slate-900">{impStats.low}</div>
                             <div className="text-[9px] text-slate-400">重合</div>
                        </div>
                    </div>
                 </div>
             </div>
          </div>
        </div>
      </motion.div>

      {/* 2. Charts Section */}
      <motion.div variants={itemVariants} className="grid grid-cols-1 lg:grid-cols-3 gap-12 border-t border-slate-100 pt-12">
         <div className="lg:col-span-2">
            <h3 className="text-xs font-bold tracking-widest text-slate-400 uppercase mb-6 flex items-center gap-2">
                <Calendar size={14} />
                历史引用趋势分布
            </h3>
            <div className="h-64 w-full bg-slate-50/50 rounded-sm p-4">
                {yearData.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={yearData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                            <XAxis 
                                dataKey="name" 
                                axisLine={false} 
                                tickLine={false} 
                                tick={{fontSize: 12, fill: '#64748b', fontFamily: 'monospace'}} 
                                dy={10}
                            />
                            <YAxis 
                                axisLine={false} 
                                tickLine={false} 
                                tick={{fontSize: 12, fill: '#64748b', fontFamily: 'monospace'}} 
                            />
                            <RechartsTooltip 
                                cursor={{fill: '#f1f5f9'}}
                                contentStyle={{ backgroundColor: '#0f172a', border: 'none', borderRadius: '0px', color: '#fff' }}
                                itemStyle={{ color: '#fff', fontFamily: 'monospace' }}
                            />
                            <Bar dataKey="count" fill="#334155" radius={[2, 2, 0, 0]} barSize={40}>
                                {yearData.map((_entry: any, index: number) => (
                                    <Cell key={`cell-${index}`} fill="#334155" />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                ) : (
                    <div className="h-full flex items-center justify-center text-slate-400 text-sm font-serif italic">暂无历史引用数据</div>
                )}
            </div>
         </div>

         <div className="lg:col-span-1">
             <h3 className="text-xs font-bold tracking-widest text-slate-400 uppercase mb-6 flex items-center gap-2">
                <Layers size={14} />
                提取概念总数
            </h3>
            <div className="bg-slate-50 rounded-sm p-8 flex flex-col justify-center h-64 border border-slate-100">
                <span className="text-7xl font-light text-slate-900 mb-2 block">{summary.total_items || 0}</span>
                <span className="text-sm text-slate-500 font-serif italic">从文稿中提取的理论概念</span>
            </div>
         </div>
      </motion.div>

      {/* 3. Conflicting Cases Table */}
      <motion.div variants={itemVariants} className="border-t border-slate-100 pt-12">
        <h3 className="text-xs font-bold tracking-widest text-slate-400 uppercase mb-8 flex items-center gap-2">
            <FileText size={14} />
            重复度最高的历史案例 (TOP 10)
        </h3>
        
        <div className="overflow-hidden">
            <table className="w-full text-left border-collapse">
                <thead>
                    <tr className="border-b border-slate-900">
                        <th className="py-4 pr-6 text-xs font-bold text-slate-900 uppercase tracking-wider w-16">#</th>
                        <th className="py-4 px-6 text-xs font-bold text-slate-900 uppercase tracking-wider">案例标题</th>
                        <th className="py-4 px-6 text-xs font-bold text-slate-900 uppercase tracking-wider w-32">年份</th>
                        <th className="py-4 pl-6 text-xs font-bold text-slate-900 uppercase tracking-wider w-32 text-right">重复点</th>
                    </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                    {caseData.length > 0 ? (
                        caseData.map((item: any, index: number) => (
                            <tr key={index} className="group hover:bg-slate-50 transition-colors">
                                <td className="py-5 pr-6 text-sm text-slate-400 font-mono">
                                    {String(index + 1).padStart(2, '0')}
                                </td>
                                <td className="py-5 px-6">
                                    <div className="font-serif text-slate-900 text-lg group-hover:text-black transition-colors">{item.filename.replace('.pdf', '')}</div>
                                </td>
                                <td className="py-5 px-6">
                                    <span className="text-slate-500 text-sm font-mono">
                                        {item.year}
                                    </span>
                                </td>
                                <td className="py-5 pl-6 text-right">
                                    <span className={`font-mono text-lg font-bold ${item.count >= 3 ? 'text-rose-700' : 'text-slate-700'}`}>
                                        {item.count}
                                    </span>
                                </td>
                            </tr>
                        ))
                    ) : (
                        <tr>
                            <td colSpan={4} className="py-16 text-center text-slate-400 font-serif italic text-lg">
                                未发现显著的历史案例重复。
                            </td>
                        </tr>
                    )}
                </tbody>
            </table>
        </div>
      </motion.div>
    </motion.div>
  );
}
