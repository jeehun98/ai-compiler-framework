import React from 'react';
import { useParams } from 'react-router-dom';
import OpsComparisonView from './OpsComparisonView';
import KernelDetailView from './KernelDetailView';
import { allAnalysisConfigs } from '../../data/analysis/configs';

export default function AnalysisLayout() {
  const { opId, kernelId } = useParams();
  
  const opData = allAnalysisConfigs[opId];
  if (!opData) return <div className="p-20 text-slate-500 italic">Select an operator to analyze.</div>;

  if (kernelId) {
    const kernelData = opData.variants.find(v => v.id === kernelId);
    return <KernelDetailView kernelData={kernelData} />;
  }

  return <OpsComparisonView opData={opData} />;
}