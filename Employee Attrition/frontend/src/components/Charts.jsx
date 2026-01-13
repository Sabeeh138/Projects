// frontend/src/components/Charts.jsx
import React, { useState } from 'react';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    Tooltip,
    ResponsiveContainer,
    Legend,
    CartesianGrid,
    Cell
} from 'recharts';
import { motion, AnimatePresence } from 'framer-motion';

export default function Charts({ metrics }) {
    const [selectedModel, setSelectedModel] = useState(null);

    // Convert metrics object → Recharts format
    const data = Object.keys(metrics).map(modelName => ({
        model: modelName,
        auc: metrics[modelName].roc_auc
    })).sort((a, b) => b.auc - a.auc);

    const colors = ['#8b5cf6', '#ec4899', '#6366f1', '#14b8a6', '#f59e0b', '#ef4444', '#10b981', '#3b82f6'];

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            return (
                <div className="bg-white dark:bg-gray-800 p-4 rounded-lg shadow-xl border-2 border-indigo-200">
                    <p className="font-bold text-indigo-600">{payload[0].payload.model}</p>
                    <p className="text-sm">AUC: <span className="font-bold">{(payload[0].value * 100).toFixed(2)}%</span></p>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="space-y-6">

            {/* --------- ROC-AUC BAR CHART --------- */}
            <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-gradient-to-br from-white to-indigo-50 rounded-2xl shadow-lg p-6 border border-indigo-200 dark:from-gray-800 dark:to-gray-900 dark:border-gray-700"
            >
                <h3 className="text-xl font-bold mb-4 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                    Model Performance Comparison
                </h3>

                <div style={{ width: "100%", height: 350 }}>
                    <ResponsiveContainer>
                        <BarChart data={data} onMouseMove={(e) => e?.activeLabel && setSelectedModel(e.activeLabel)}>
                            <defs>
                                {data.map((entry, index) => (
                                    <linearGradient key={index} id={`gradient${index}`} x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor={colors[index % colors.length]} stopOpacity={0.9}/>
                                        <stop offset="100%" stopColor={colors[index % colors.length]} stopOpacity={0.6}/>
                                    </linearGradient>
                                ))}
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e0e7ff" />
                            <XAxis 
                                dataKey="model" 
                                angle={-15} 
                                textAnchor="end" 
                                height={80}
                                tick={{ fontSize: 12 }}
                            />
                            <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
                            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(99, 102, 241, 0.1)' }} />
                            <Bar dataKey="auc" radius={[10, 10, 0, 0]} animationDuration={1000}>
                                {data.map((entry, index) => (
                                    <Cell 
                                        key={`cell-${index}`} 
                                        fill={`url(#gradient${index})`}
                                        stroke={colors[index % colors.length]}
                                        strokeWidth={2}
                                    />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </motion.div>

            {/* --------- STRUCTURED METRICS DETAILS --------- */}
            <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-gradient-to-br from-white to-pink-50 rounded-2xl shadow-lg p-6 border border-pink-200 dark:from-gray-800 dark:to-gray-900 dark:border-gray-700"
            >
                <h3 className="text-xl font-bold mb-6 bg-gradient-to-r from-pink-600 to-rose-600 bg-clip-text text-transparent">
                    Detailed Model Metrics
                </h3>

                <div className="space-y-4">
                    {Object.keys(metrics).sort((a, b) => metrics[b].roc_auc - metrics[a].roc_auc).map((modelName, index) => {
                        const m = metrics[modelName];
                        const [expanded, setExpanded] = useState(false);

                        return (
                            <motion.div
                                key={modelName}
                                initial={{ opacity: 0, x: -20 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: index * 0.1 }}
                                whileHover={{ scale: 1.02 }}
                                className="p-5 rounded-xl bg-white/80 backdrop-blur-sm border-2 border-indigo-200 shadow-lg hover:shadow-xl transition-all dark:bg-gray-800/80 dark:border-gray-600"
                            >
                                <div className="flex justify-between items-center mb-3">
                                    <h4 className="text-lg font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
                                        {index === 0 && 'Best: '}{modelName}
                                    </h4>
                                    <motion.button
                                        whileHover={{ scale: 1.1 }}
                                        whileTap={{ scale: 0.9 }}
                                        onClick={() => setExpanded(!expanded)}
                                        className="text-2xl"
                                    >
                                        {expanded ? '▼' : '▶'}
                                    </motion.button>
                                </div>

                                {/* AUC Score with Progress Bar */}
                                <div className="mb-4">
                                    <div className="flex justify-between items-center mb-2">
                                        <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">ROC-AUC Score</span>
                                        <span className="text-xl font-bold text-indigo-600 dark:text-indigo-400">
                                            {(m.roc_auc * 100).toFixed(2)}%
                                        </span>
                                    </div>
                                    <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden dark:bg-gray-700">
                                        <motion.div
                                            initial={{ width: 0 }}
                                            animate={{ width: `${m.roc_auc * 100}%` }}
                                            transition={{ duration: 1, delay: index * 0.1 }}
                                            className="h-full bg-gradient-to-r from-indigo-500 to-purple-600 rounded-full"
                                        />
                                    </div>
                                </div>

                                <AnimatePresence>
                                    {expanded && (
                                        <motion.div
                                            initial={{ opacity: 0, height: 0 }}
                                            animate={{ opacity: 1, height: 'auto' }}
                                            exit={{ opacity: 0, height: 0 }}
                                            transition={{ duration: 0.3 }}
                                            className="space-y-4"
                                        >
                                            {/* Confusion Matrix - Visual Grid */}
                                            <div>
                                                <h5 className="font-semibold text-gray-700 dark:text-gray-300 mb-3">Confusion Matrix</h5>
                                                <div className="grid grid-cols-2 gap-2">
                                                    <div className="p-4 bg-gradient-to-br from-green-100 to-emerald-200 rounded-lg text-center">
                                                        <div className="text-2xl font-bold text-green-700">{m.confusion_matrix[0][0]}</div>
                                                        <div className="text-xs text-green-600 font-semibold">True Negative</div>
                                                    </div>
                                                    <div className="p-4 bg-gradient-to-br from-red-100 to-rose-200 rounded-lg text-center">
                                                        <div className="text-2xl font-bold text-red-700">{m.confusion_matrix[0][1]}</div>
                                                        <div className="text-xs text-red-600 font-semibold">False Positive</div>
                                                    </div>
                                                    <div className="p-4 bg-gradient-to-br from-orange-100 to-amber-200 rounded-lg text-center">
                                                        <div className="text-2xl font-bold text-orange-700">{m.confusion_matrix[1][0]}</div>
                                                        <div className="text-xs text-orange-600 font-semibold">False Negative</div>
                                                    </div>
                                                    <div className="p-4 bg-gradient-to-br from-blue-100 to-indigo-200 rounded-lg text-center">
                                                        <div className="text-2xl font-bold text-blue-700">{m.confusion_matrix[1][1]}</div>
                                                        <div className="text-xs text-blue-600 font-semibold">True Positive</div>
                                                    </div>
                                                </div>
                                            </div>

                                            {/* Classification Metrics */}
                                            <div>
                                                <h5 className="font-semibold text-gray-700 dark:text-gray-300 mb-3">
                                                    Performance Metrics
                                                </h5>
                                                <div className="grid grid-cols-2 gap-3">
                                                    {["0", "1"].map(cls => (
                                                        <div key={cls} className="p-3 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-gray-700 dark:to-gray-600 rounded-lg">
                                                            <div className="text-xs font-semibold text-gray-600 dark:text-gray-300 mb-2">
                                                                Class {cls} {cls === "0" ? "(Stay)" : "(Leave)"}
                                                            </div>
                                                            <div className="space-y-1 text-xs">
                                                                <div className="flex justify-between">
                                                                    <span>Precision:</span>
                                                                    <span className="font-bold">{(m.classification_report[cls]["precision"] * 100).toFixed(1)}%</span>
                                                                </div>
                                                                <div className="flex justify-between">
                                                                    <span>Recall:</span>
                                                                    <span className="font-bold">{(m.classification_report[cls]["recall"] * 100).toFixed(1)}%</span>
                                                                </div>
                                                                <div className="flex justify-between">
                                                                    <span>F1-Score:</span>
                                                                    <span className="font-bold">{(m.classification_report[cls]["f1-score"] * 100).toFixed(1)}%</span>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        </motion.div>
                                    )}
                                </AnimatePresence>
                            </motion.div>
                        );
                    })}
                </div>
            </motion.div>

        </div>
    );
}
