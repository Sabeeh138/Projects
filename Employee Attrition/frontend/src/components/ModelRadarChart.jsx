// frontend/src/components/ModelRadarChart.jsx
import React, { useState } from 'react';
import {
    Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer, Legend, Tooltip
} from 'recharts';
import { motion } from 'framer-motion';

export default function ModelRadarChart({ metrics }) {
    const [selectedModel, setSelectedModel] = useState(null);

    if (!metrics || Object.keys(metrics).length === 0) {
        return (
            <div className="bg-white/80 backdrop-blur-lg p-6 rounded-xl shadow-lg border border-white/20 dark:bg-gray-800/80 h-full flex items-center justify-center">
                <p className="text-gray-500">No metrics available</p>
            </div>
        );
    }

    // Get top 5 models by AUC
    const topModels = Object.keys(metrics)
        .sort((a, b) => metrics[b].roc_auc - metrics[a].roc_auc)
        .slice(0, 5);

    // Build radar data - one object per metric with all models
    const radarData = [
        {
            metric: 'AUC',
            ...Object.fromEntries(topModels.map(name => [name, metrics[name].roc_auc]))
        },
        {
            metric: 'Precision',
            ...Object.fromEntries(topModels.map(name => [name, metrics[name].classification_report["1"]?.precision || 0]))
        },
        {
            metric: 'Recall',
            ...Object.fromEntries(topModels.map(name => [name, metrics[name].classification_report["1"]?.recall || 0]))
        },
        {
            metric: 'F1-Score',
            ...Object.fromEntries(topModels.map(name => [name, metrics[name].classification_report["1"]?.["f1-score"] || 0]))
        },
        {
            metric: 'Accuracy',
            ...Object.fromEntries(topModels.map(name => [name, metrics[name].classification_report.accuracy || 0]))
        }
    ];

    const colors = ['#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#3b82f6'];

    return (
        <div className="bg-white/80 backdrop-blur-lg p-6 rounded-xl shadow-lg border border-white/20 dark:bg-gray-800/80 h-full">
            <h4 className="text-lg font-bold mb-4 bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                Model Comparison
            </h4>
            
            <div style={{ width: '100%', height: 320 }}>
                <ResponsiveContainer>
                    <RadarChart data={radarData} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
                        <PolarGrid stroke="#e0e7ff" />
                        <PolarAngleAxis 
                            dataKey="metric" 
                            tick={{ fill: '#6b7280', fontSize: 12, fontWeight: 600 }}
                        />
                        <PolarRadiusAxis 
                            angle={90} 
                            domain={[0, 1]} 
                            tick={{ fill: '#9ca3af', fontSize: 10 }}
                        />
                        <Tooltip 
                            contentStyle={{ 
                                backgroundColor: 'rgba(255, 255, 255, 0.95)', 
                                border: '2px solid #e0e7ff',
                                borderRadius: '8px',
                                fontSize: '12px'
                            }}
                            formatter={(value) => (value * 100).toFixed(1) + '%'}
                        />
                        {topModels.map((modelName, index) => (
                            <Radar
                                key={modelName}
                                name={modelName}
                                dataKey={modelName}
                                stroke={colors[index]}
                                fill={colors[index]}
                                fillOpacity={selectedModel === modelName ? 0.6 : 0.2}
                                strokeWidth={selectedModel === modelName ? 3 : 2}
                                onMouseEnter={() => setSelectedModel(modelName)}
                                onMouseLeave={() => setSelectedModel(null)}
                            />
                        ))}
                    </RadarChart>
                </ResponsiveContainer>
            </div>

            {/* Model Legend */}
            <div className="mt-4 grid grid-cols-2 gap-2">
                {topModels.map((modelName, index) => (
                    <motion.div
                        key={modelName}
                        whileHover={{ scale: 1.05 }}
                        onMouseEnter={() => setSelectedModel(modelName)}
                        onMouseLeave={() => setSelectedModel(null)}
                        className={`flex items-center gap-2 p-2 rounded-lg cursor-pointer transition-all ${
                            selectedModel === modelName 
                                ? 'bg-gradient-to-r from-purple-100 to-pink-100 dark:from-purple-900 dark:to-pink-900' 
                                : 'bg-gray-50 dark:bg-gray-700'
                        }`}
                    >
                        <div 
                            className="w-3 h-3 rounded-full" 
                            style={{ backgroundColor: colors[index] }}
                        />
                        <span className="text-xs font-semibold text-gray-700 dark:text-gray-200 truncate">
                            {modelName}
                        </span>
                    </motion.div>
                ))}
            </div>
        </div>
    );
}
