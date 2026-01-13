// frontend/src/components/FeatureImportance.jsx
import React, { useEffect, useState } from 'react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, Cell } from 'recharts';
import { motion } from 'framer-motion';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export default function FeatureImportance() {
    const [data, setData] = useState(null);
    const [viewMode, setViewMode] = useState('chart'); // 'chart' or 'list'

    useEffect(() => {
        async function load() {
            try {
                const res = await axios.get(`${API_BASE}/feature-importance`);
                if (res.data.source === 'shap') {
                    setData(res.data.data.shap_top.slice(0, 10).map(d => ({ 
                        name: d.feature, 
                        value: d.mean_abs_shap,
                        shortName: d.feature.length > 20 ? d.feature.substring(0, 17) + '...' : d.feature
                    })));
                } else if (res.data.source === 'feature_importances') {
                    setData(res.data.data.slice(0, 10).map(d => ({ 
                        name: d.feature, 
                        value: d.importance,
                        shortName: d.feature.length > 20 ? d.feature.substring(0, 17) + '...' : d.feature
                    })));
                }
            } catch (err) {
                console.error("Failed to load feature importance", err);
            }
        }
        load();
    }, []);

    if (!data) {
        return (
            <div className="bg-white/80 backdrop-blur-lg p-6 rounded-xl shadow-lg border border-white/20 dark:bg-gray-800/80 h-full flex items-center justify-center">
                <div className="animate-spin rounded-full h-12 w-12 border-t-4 border-b-4 border-pink-600"></div>
            </div>
        );
    }

    const colors = ['#ec4899', '#f43f5e', '#f97316', '#f59e0b', '#eab308', '#84cc16', '#22c55e', '#10b981', '#14b8a6', '#06b6d4'];

    const CustomTooltip = ({ active, payload }) => {
        if (active && payload && payload.length) {
            return (
                <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-xl border-2 border-pink-200">
                    <p className="font-bold text-pink-600 text-sm">{payload[0].payload.name}</p>
                    <p className="text-xs">Importance: <span className="font-bold">{payload[0].value.toFixed(4)}</span></p>
                </div>
            );
        }
        return null;
    };

    return (
        <div className="bg-white/80 backdrop-blur-lg p-6 rounded-xl shadow-lg border border-white/20 dark:bg-gray-800/80 h-full">
            <div className="flex justify-between items-center mb-4">
                <h4 className="text-lg font-bold bg-gradient-to-r from-pink-600 to-rose-600 bg-clip-text text-transparent">
                    Top Features
                </h4>
                <div className="flex gap-2">
                    <motion.button
                        whileHover={{ scale: 1.1 }}
                        whileTap={{ scale: 0.9 }}
                        onClick={() => setViewMode('chart')}
                        className={`px-3 py-1 rounded-lg text-sm font-semibold transition-all ${
                            viewMode === 'chart' 
                                ? 'bg-gradient-to-r from-pink-500 to-rose-600 text-white' 
                                : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
                        }`}
                    >
                        Chart
                    </motion.button>
                    <motion.button
                        whileHover={{ scale: 1.1 }}
                        whileTap={{ scale: 0.9 }}
                        onClick={() => setViewMode('list')}
                        className={`px-3 py-1 rounded-lg text-sm font-semibold transition-all ${
                            viewMode === 'list' 
                                ? 'bg-gradient-to-r from-pink-500 to-rose-600 text-white' 
                                : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
                        }`}
                    >
                        List
                    </motion.button>
                </div>
            </div>

            {viewMode === 'chart' ? (
                <div style={{ width: '100%', height: 350 }}>
                    <ResponsiveContainer>
                        <BarChart data={data} layout="vertical" margin={{ left: 10, right: 10 }}>
                            <defs>
                                {data.map((entry, index) => (
                                    <linearGradient key={index} id={`colorGradient${index}`} x1="0" y1="0" x2="1" y2="0">
                                        <stop offset="0%" stopColor={colors[index]} stopOpacity={0.8}/>
                                        <stop offset="100%" stopColor={colors[index]} stopOpacity={1}/>
                                    </linearGradient>
                                ))}
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e0e7ff" />
                            <XAxis type="number" tick={{ fontSize: 11 }} />
                            <YAxis 
                                type="category" 
                                dataKey="shortName" 
                                width={120}
                                tick={{ fontSize: 11 }}
                            />
                            <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(236, 72, 153, 0.1)' }} />
                            <Bar dataKey="value" radius={[0, 8, 8, 0]} animationDuration={1000}>
                                {data.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={`url(#colorGradient${index})`} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            ) : (
                <div className="space-y-2 max-h-[350px] overflow-y-auto">
                    {data.map((item, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, x: -20 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: index * 0.05 }}
                            className="p-3 bg-gradient-to-r from-pink-50 to-rose-50 dark:from-gray-700 dark:to-gray-600 rounded-lg"
                        >
                            <div className="flex justify-between items-center mb-1">
                                <span className="text-sm font-semibold text-gray-700 dark:text-gray-200 truncate flex-1">
                                    {index + 1}. {item.name}
                                </span>
                                <span className="text-sm font-bold text-pink-600 dark:text-pink-400 ml-2">
                                    {item.value.toFixed(4)}
                                </span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-600">
                                <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${(item.value / data[0].value) * 100}%` }}
                                    transition={{ duration: 0.8, delay: index * 0.05 }}
                                    className="h-full rounded-full"
                                    style={{ background: colors[index] }}
                                />
                            </div>
                        </motion.div>
                    ))}
                </div>
            )}
        </div>
    );
}
