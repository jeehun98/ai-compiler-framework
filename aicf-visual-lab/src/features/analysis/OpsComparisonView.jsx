import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, CartesianGrid } from 'recharts';
import { Zap, TrendingUp, Activity } from 'lucide-react';

export default function OpsComparisonView({ opData }) {
  // 차트 데이터 가공
  const chartData = opData.variants.map(v => ({
    name: v.id,
    label: v.name,
    latency: v.metrics.metrics["gpu__time_duration.sum"].val,
    throughput: v.metrics.metrics["sm__throughput.avg.pct_of_peak_sustained_elapsed"].val
  }));

  return (
    <div className="space-y-8 animate-in fade-in duration-700">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-[#1e293b] p-6 rounded-[2rem] border border-slate-800">
          <div className="text-slate-500 text-[10px] font-black uppercase mb-2">Target Op</div>
          <div className="text-2xl font-black text-white">{opData.label}</div>
        </div>
        <div className="bg-[#1e293b] p-6 rounded-[2rem] border border-slate-800">
          <div className="text-slate-500 text-[10px] font-black uppercase mb-2">Best Performer</div>
          <div className="text-2xl font-black text-emerald-400 flex items-center gap-2">
            <Zap size={20} className="fill-current"/> {chartData[0]?.label}
          </div>
        </div>
        <div className="bg-[#1e293b] p-6 rounded-[2rem] border border-slate-800">
          <div className="text-slate-500 text-[10px] font-black uppercase mb-2">Avg. SM Throughput</div>
          <div className="text-2xl font-black text-blue-400">
            {(chartData.reduce((acc, curr) => acc + curr.throughput, 0) / chartData.length).toFixed(1)}%
          </div>
        </div>
      </div>

      <div className="bg-[#1e293b] border border-slate-800 rounded-[3rem] p-10 shadow-2xl">
        <h3 className="text-white font-black text-xl mb-8 flex items-center gap-3">
          <TrendingUp className="text-emerald-500" /> Latency Comparison (us)
        </h3>
        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
              <XAxis dataKey="label" stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} />
              <YAxis stroke="#64748b" fontSize={11} tickLine={false} axisLine={false} />
              <Tooltip 
                cursor={{fill: '#ffffff05'}} 
                contentStyle={{backgroundColor: '#0f172a', border: '1px solid #334155', borderRadius: '16px'}}
              />
              <Bar dataKey="latency" radius={[10, 10, 0, 0]} barSize={50}>
                {chartData.map((entry, index) => (
                  <Cell key={index} fill={index === 0 ? '#10b981' : '#334155'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}