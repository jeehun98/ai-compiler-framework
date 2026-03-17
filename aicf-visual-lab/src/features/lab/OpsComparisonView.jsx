import React, { useMemo, useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  CartesianGrid,
} from 'recharts';
import {
  Zap,
  TrendingUp,
  Activity,
  Layers,
  Cpu,
  ChevronRight,
  BarChart3,
} from 'lucide-react';

const METRIC_OPTIONS = {
  latency: {
    key: 'latency',
    label: 'Latency',
    title: 'Latency Comparison',
    unit: 'us',
    icon: TrendingUp,
    color: 'text-emerald-500',
    lowerIsBetter: true,
    formatter: (v) => `${Number(v).toFixed(3)} us`,
  },
  throughput: {
    key: 'throughput',
    label: 'SM Throughput',
    title: 'SM Throughput Comparison',
    unit: '%',
    icon: Activity,
    color: 'text-blue-500',
    lowerIsBetter: false,
    formatter: (v) => `${Number(v).toFixed(1)}%`,
  },
  warpsActive: {
    key: 'warpsActive',
    label: 'Warp Active',
    title: 'Warp Active Comparison',
    unit: '%',
    icon: BarChart3,
    color: 'text-violet-500',
    lowerIsBetter: false,
    formatter: (v) => `${Number(v).toFixed(1)}%`,
  },
};

export default function OpsComparisonView({ opData }) {
  const [activeMetric, setActiveMetric] = useState('latency');

  const chartData = useMemo(() => {
    return opData.variants.map((v) => ({
      id: v.id,
      label: v.name,
      tag: v.tag,
      description: v.description,
      algorithm: v.algorithm,
      blueprint: v.blueprint,
      features: v.features ?? [],
      insights: v.insights ?? [],
      latency: v.metrics?.metrics?.['gpu__time_duration.sum']?.val ?? 0,
      throughput:
        v.metrics?.metrics?.['sm__throughput.avg.pct_of_peak_sustained_elapsed']?.val ?? 0,
      warpsActive:
        v.metrics?.metrics?.['sm__warps_active.avg.pct_of_peak_sustained_active']?.val ?? 0,
    }));
  }, [opData]);

  const activeMetricMeta = METRIC_OPTIONS[activeMetric];

  const bestPerformer = useMemo(() => {
    if (!chartData.length) return null;

    const sorted = [...chartData].sort((a, b) => {
      const av = a[activeMetricMeta.key] ?? 0;
      const bv = b[activeMetricMeta.key] ?? 0;
      return activeMetricMeta.lowerIsBetter ? av - bv : bv - av;
    });

    return sorted[0];
  }, [chartData, activeMetricMeta]);

  const avgThroughput = useMemo(() => {
    if (!chartData.length) return 0;
    return chartData.reduce((acc, curr) => acc + curr.throughput, 0) / chartData.length;
  }, [chartData]);

  const avgLatency = useMemo(() => {
    if (!chartData.length) return 0;
    return chartData.reduce((acc, curr) => acc + curr.latency, 0) / chartData.length;
  }, [chartData]);

  const avgWarpsActive = useMemo(() => {
    if (!chartData.length) return 0;
    return chartData.reduce((acc, curr) => acc + curr.warpsActive, 0) / chartData.length;
  }, [chartData]);

  const ActiveMetricIcon = activeMetricMeta.icon;

  return (
    <div className="space-y-8 animate-in fade-in duration-700">
      {/* Top Summary */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="bg-[#1e293b] p-6 rounded-[2rem] border border-slate-800">
          <div className="text-slate-500 text-[10px] font-black uppercase mb-2">
            Target Op
          </div>
          <div className="text-2xl font-black text-white">{opData.label}</div>
          <div className="text-sm text-slate-400 mt-2">{opData.category}</div>
        </div>

        <div className="bg-[#1e293b] p-6 rounded-[2rem] border border-slate-800">
          <div className="text-slate-500 text-[10px] font-black uppercase mb-2">
            Best {activeMetricMeta.label}
          </div>
          <div className="text-2xl font-black text-emerald-400 flex items-center gap-2">
            <Zap size={20} className="fill-current" />
            {bestPerformer?.label ?? '-'}
          </div>
          <div className="text-sm text-slate-400 mt-2">
            {bestPerformer
              ? activeMetricMeta.formatter(bestPerformer[activeMetricMeta.key])
              : '-'}
          </div>
        </div>

        <div className="bg-[#1e293b] p-6 rounded-[2rem] border border-slate-800">
          <div className="text-slate-500 text-[10px] font-black uppercase mb-2">
            Avg. Latency
          </div>
          <div className="text-2xl font-black text-emerald-400">
            {avgLatency.toFixed(3)} us
          </div>
        </div>

        <div className="bg-[#1e293b] p-6 rounded-[2rem] border border-slate-800">
          <div className="text-slate-500 text-[10px] font-black uppercase mb-2">
            Avg. SM / Warp
          </div>
          <div className="text-lg font-black text-blue-400">
            {avgThroughput.toFixed(1)}%
          </div>
          <div className="text-sm text-slate-400 mt-1">
            Warp Active {avgWarpsActive.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Comparison Summary */}
      {opData.comparisonSummary && (
        <div className="bg-[#1e293b] border border-slate-800 rounded-[2rem] p-8">
          <div className="text-slate-500 text-[10px] font-black uppercase mb-3">
            Comparison Summary
          </div>
          <p className="text-slate-200 text-base leading-7">
            {opData.comparisonSummary}
          </p>
        </div>
      )}

      {/* Metric Toggle + Chart */}
      <div className="bg-[#1e293b] border border-slate-800 rounded-[3rem] p-10 shadow-2xl">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6 mb-8">
          <h3 className="text-white font-black text-xl flex items-center gap-3">
            <ActiveMetricIcon className={activeMetricMeta.color} />
            {activeMetricMeta.title} ({activeMetricMeta.unit})
          </h3>

          <div className="flex flex-wrap gap-3">
            {Object.entries(METRIC_OPTIONS).map(([key, meta]) => {
              const Icon = meta.icon;
              const isActive = activeMetric === key;

              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => setActiveMetric(key)}
                  className={`px-4 py-2 rounded-2xl border text-sm font-bold transition-all ${
                    isActive
                      ? 'bg-slate-100 text-slate-900 border-slate-100'
                      : 'bg-slate-900/60 text-slate-300 border-slate-700 hover:border-slate-500'
                  }`}
                >
                  <span className="flex items-center gap-2">
                    <Icon size={16} />
                    {meta.label}
                  </span>
                </button>
              );
            })}
          </div>
        </div>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="#334155"
                vertical={false}
              />
              <XAxis
                dataKey="label"
                stroke="#64748b"
                fontSize={11}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                stroke="#64748b"
                fontSize={11}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip
                cursor={{ fill: '#ffffff05' }}
                contentStyle={{
                  backgroundColor: '#0f172a',
                  border: '1px solid #334155',
                  borderRadius: '16px',
                }}
                formatter={(value) => [
                  activeMetricMeta.formatter(value),
                  activeMetricMeta.label,
                ]}
              />
              <Bar
                dataKey={activeMetricMeta.key}
                radius={[10, 10, 0, 0]}
                barSize={54}
              >
                {chartData.map((entry) => (
                  <Cell
                    key={entry.id}
                    fill={entry.id === bestPerformer?.id ? '#10b981' : '#334155'}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="mt-6 text-sm text-slate-400">
          {activeMetric === 'latency' && (
            <span>낮을수록 더 빠른 커널이다.</span>
          )}
          {activeMetric === 'throughput' && (
            <span>높을수록 SM 자원 활용 비율이 높다.</span>
          )}
          {activeMetric === 'warpsActive' && (
            <span>높을수록 더 많은 warp가 동시에 활성 상태에 있다.</span>
          )}
        </div>
      </div>

      {/* Variant Detail Cards */}
      <div className="space-y-6">
        {chartData.map((variant) => (
          <div
            key={variant.id}
            className="bg-[#1e293b] border border-slate-800 rounded-[2rem] p-8"
          >
            <div className="flex flex-col xl:flex-row xl:items-start xl:justify-between gap-6 mb-6">
              <div>
                <div className="flex items-center gap-3 mb-3 flex-wrap">
                  <div className="text-white text-2xl font-black">
                    {variant.label}
                  </div>
                  <span className="px-3 py-1 rounded-full text-xs font-bold bg-slate-800 text-slate-300 border border-slate-700">
                    {variant.tag}
                  </span>
                  {bestPerformer?.id === variant.id && (
                    <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                      Best {activeMetricMeta.label}
                    </span>
                  )}
                </div>
                <p className="text-slate-300 leading-7 max-w-4xl">
                  {variant.description}
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 min-w-[300px]">
                <div className="bg-slate-900/60 rounded-2xl p-4 border border-slate-800">
                  <div className="text-[10px] uppercase font-black text-slate-500 mb-2">
                    Latency
                  </div>
                  <div className="text-emerald-400 font-black text-xl">
                    {variant.latency.toFixed(3)}
                  </div>
                  <div className="text-slate-500 text-xs">us</div>
                </div>

                <div className="bg-slate-900/60 rounded-2xl p-4 border border-slate-800">
                  <div className="text-[10px] uppercase font-black text-slate-500 mb-2">
                    SM Throughput
                  </div>
                  <div className="text-blue-400 font-black text-xl">
                    {variant.throughput.toFixed(1)}
                  </div>
                  <div className="text-slate-500 text-xs">%</div>
                </div>

                <div className="bg-slate-900/60 rounded-2xl p-4 border border-slate-800">
                  <div className="text-[10px] uppercase font-black text-slate-500 mb-2">
                    Warp Active
                  </div>
                  <div className="text-violet-400 font-black text-xl">
                    {variant.warpsActive.toFixed(1)}
                  </div>
                  <div className="text-slate-500 text-xs">%</div>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              {/* Algorithm */}
              <div className="bg-slate-900/50 rounded-[1.5rem] p-6 border border-slate-800">
                <div className="flex items-center gap-2 text-white font-black mb-4">
                  <Cpu size={18} className="text-cyan-400" />
                  {variant.algorithm?.title ?? 'Algorithm'}
                </div>
                <p className="text-slate-300 leading-7 mb-4">
                  {variant.algorithm?.logic}
                </p>
                <div className="space-y-2">
                  {(variant.algorithm?.strategy ?? []).map((item, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-2 text-sm text-slate-300"
                    >
                      <ChevronRight size={16} className="text-slate-500 mt-[2px]" />
                      <span>{item}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Blueprint */}
              <div className="bg-slate-900/50 rounded-[1.5rem] p-6 border border-slate-800">
                <div className="flex items-center gap-2 text-white font-black mb-4">
                  <Layers size={18} className="text-amber-400" />
                  Blueprint
                </div>

                <div className="grid grid-cols-2 gap-4 text-sm mb-4">
                  <div>
                    <div className="text-slate-500 uppercase text-[10px] font-black mb-1">
                      Memory Access
                    </div>
                    <div className="text-slate-200">
                      {variant.blueprint?.mem_access}
                    </div>
                  </div>
                  <div>
                    <div className="text-slate-500 uppercase text-[10px] font-black mb-1">
                      Instruction
                    </div>
                    <div className="text-slate-200">
                      {variant.blueprint?.instruction}
                    </div>
                  </div>
                  <div>
                    <div className="text-slate-500 uppercase text-[10px] font-black mb-1">
                      Vector Width
                    </div>
                    <div className="text-slate-200">
                      {variant.blueprint?.vector_width}
                    </div>
                  </div>
                  <div>
                    <div className="text-slate-500 uppercase text-[10px] font-black mb-1">
                      Alignment
                    </div>
                    <div className="text-slate-200">
                      {variant.blueprint?.alignment_req}
                    </div>
                  </div>
                </div>

                <div>
                  <div className="text-slate-500 uppercase text-[10px] font-black mb-2">
                    Code Snippet
                  </div>
                  <pre className="bg-[#020617] text-emerald-300 text-xs rounded-2xl p-4 overflow-x-auto border border-slate-800 whitespace-pre-wrap">
                    {variant.blueprint?.code_snippet}
                  </pre>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mt-6">
              <div className="bg-slate-900/50 rounded-[1.5rem] p-6 border border-slate-800">
                <div className="flex items-center gap-2 text-white font-black mb-4">
                  <Activity size={18} className="text-emerald-400" />
                  Features
                </div>
                <div className="flex flex-wrap gap-2">
                  {variant.features.map((feature, idx) => (
                    <span
                      key={idx}
                      className="px-3 py-2 rounded-full text-sm bg-slate-800 text-slate-200 border border-slate-700"
                    >
                      {feature}
                    </span>
                  ))}
                </div>
              </div>

              <div className="bg-slate-900/50 rounded-[1.5rem] p-6 border border-slate-800">
                <div className="flex items-center gap-2 text-white font-black mb-4">
                  <Zap size={18} className="text-yellow-400" />
                  Insights
                </div>
                <div className="space-y-2">
                  {variant.insights.map((item, idx) => (
                    <div
                      key={idx}
                      className="flex items-start gap-2 text-sm text-slate-300"
                    >
                      <ChevronRight size={16} className="text-slate-500 mt-[2px]" />
                      <span>{item}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}