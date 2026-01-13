// frontend/src/components/ShapSummary.jsx
import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000';

export default function ShapSummary() {
    const [shap, setShap] = useState(null);

    useEffect(() => {
        async function load() {
            try {
                const res = await axios.get(`${API_BASE}/shap-summary`);
                setShap(res.data);
            } catch (err) {
                console.error("Failed to load SHAP summary", err);
            }
        }
        load();
    }, []);

    if (!shap) {
        return (
            <div className="bg-white/80 backdrop-blur-lg p-6 rounded-xl shadow-lg border border-white/20 dark:bg-gray-800/80 flex items-center justify-center h-32">
                <div className="animate-spin rounded-full h-10 w-10 border-t-4 border-b-4 border-indigo-600"></div>
            </div>
        );
    }

    const maxValue = Math.max(...shap.shap_top.map(r => r.mean_abs_shap));

    return (
        <div className="bg-white/80 backdrop-blur-lg p-6 rounded-xl shadow-lg border border-white/20 dark:bg-gray-800/80">
            <div className="mb-4">
                <h4 className="text-lg font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent mb-2">
                    SHAP Feature Impact
                </h4>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                    Model: <span className="font-bold text-indigo-600 dark:text-indigo-400">{shap.model}</span>
                </p>
            </div>

            <div className="space-y-2 max-h-96 overflow-y-auto pr-2 custom-scrollbar">
                {shap.shap_top.slice(0, 15).map((r, i) => (
                    <motion.div
                        key={r.feature}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.03 }}
                        whileHover={{ scale: 1.02, x: 5 }}
                        className="p-3 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-gray-700 dark:to-gray-600 rounded-lg cursor-pointer"
                    >
                        <div className="flex justify-between items-center mb-2">
                            <span className="text-xs font-semibold text-gray-700 dark:text-gray-200 truncate flex-1 pr-2">
                                {i + 1}. {r.feature}
                            </span>
                            <span className="text-xs font-bold text-indigo-600 dark:text-indigo-400">
                                {Number(r.mean_abs_shap).toFixed(4)}
                            </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-600 overflow-hidden">
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${(r.mean_abs_shap / maxValue) * 100}%` }}
                                transition={{ duration: 0.8, delay: i * 0.03 }}
                                className="h-full rounded-full bg-gradient-to-r from-indigo-500 to-purple-600"
                            />
                        </div>
                    </motion.div>
                ))}
            </div>
        </div>
    );
}
